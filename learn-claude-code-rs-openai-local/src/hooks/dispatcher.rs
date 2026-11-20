use std::{collections::BTreeMap, sync::Arc, time::Instant};

use async_trait::async_trait;
use tracing::{Instrument, error, info, info_span, instrument};

use super::{
    config::{HookBackend, HookConfig, HookEvent, HookMatcher, HookRule},
    runner::{HookContext, HookResponse, HookRunner},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HookDecision {
    Allow,
    Deny,
    Modify,
    Noop,
    Error,
}

impl Default for HookDecision {
    fn default() -> Self {
        Self::Allow
    }
}

impl HookDecision {
    fn as_str(self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::Deny => "deny",
            Self::Modify => "modify",
            Self::Noop => "noop",
            Self::Error => "error",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct HookDispatchResult {
    pub(crate) decision: HookDecision,
    pub(crate) reason: Option<String>,
    pub(crate) modified_arguments: Option<String>,
    pub(crate) modified_prompt: Option<String>,
    pub(crate) reminders: Vec<String>,
}

#[async_trait]
pub(crate) trait NativeHook: Send + Sync {
    async fn execute(&self, context: HookContext) -> anyhow::Result<HookResponse>;
}

#[derive(Default)]
pub(crate) struct NativeHookRegistry {
    hooks: BTreeMap<String, Arc<dyn NativeHook>>,
}

impl NativeHookRegistry {
    pub(crate) fn register(&mut self, name: impl Into<String>, hook: impl NativeHook + 'static) {
        self.hooks.insert(name.into(), Arc::new(hook));
    }

    pub(crate) fn get(&self, name: &str) -> Option<Arc<dyn NativeHook>> {
        self.hooks.get(name).cloned()
    }
}

#[derive(Clone)]
pub(crate) struct HookDispatcher {
    rules: Arc<Vec<HookRule>>,
    native: Arc<NativeHookRegistry>,
}

impl HookDispatcher {
    pub(crate) fn from_config(config: HookConfig) -> Self {
        Self {
            rules: Arc::new(config.rules),
            native: Arc::new(NativeHookRegistry::default()),
        }
    }

    pub(crate) fn with_native(mut self, native: NativeHookRegistry) -> Self {
        self.native = Arc::new(native);
        self
    }

    pub(crate) fn disabled() -> Self {
        Self::from_config(HookConfig { rules: Vec::new() })
    }

    fn matches_rule(rule: &HookRule, event: HookEvent, tool_name: Option<&str>) -> bool {
        if rule.event != event {
            return false;
        }
        match &rule.matcher {
            HookMatcher::All => true,
            HookMatcher::Glob(pattern) => tool_name
                .map(|name| wildmatch::WildMatch::new(pattern).matches(name))
                .unwrap_or(false),
            HookMatcher::Regex(pattern) => {
                let Ok(regex) = regex::Regex::new(pattern) else {
                    return false;
                };
                tool_name.map(|name| regex.is_match(name)).unwrap_or(false)
            }
        }
    }

    #[instrument(
        name = "hook.dispatch",
        skip(self, context),
        fields(hook.event = %context.event, hook.tool_name = ?context.tool_name)
    )]
    pub(crate) async fn dispatch(&self, mut context: HookContext) -> HookDispatchResult {
        let event = context.event.parse().unwrap_or(HookEvent::SessionStart);
        let mut aggregate = HookDispatchResult {
            decision: HookDecision::Allow,
            ..Default::default()
        };

        for rule in self
            .rules
            .iter()
            .filter(|rule| Self::matches_rule(rule, event, context.tool_name.as_deref()))
        {
            let started = Instant::now();
            let span = info_span!(
                "hook.rule",
                hook.event = %context.event,
                hook.name = %rule.name,
                hook.id = %rule.id,
                hook.tool_name = ?context.tool_name,
                hook.decision = tracing::field::Empty,
                hook.exit_code = tracing::field::Empty,
                hook.duration_ms = tracing::field::Empty,
                hook.stdout = tracing::field::Empty,
                hook.stderr = tracing::field::Empty,
            );
            let outcome = async {
                match &rule.backend {
                    HookBackend::Native { name } => {
                        if let Some(native) = self.native.get(name) {
                            native.execute(context.clone()).await.map(|response| {
                                (
                                    HookResponseMetadata::Native,
                                    0i64,
                                    String::new(),
                                    String::new(),
                                    response,
                                )
                            })
                        } else {
                            Err(anyhow::anyhow!("Native hook '{}' is not registered", name))
                        }
                    }
                    HookBackend::External { .. } => {
                        match HookRunner::run_external(rule, &rule.backend, &context).await {
                            Ok((output, response)) => Ok((
                                HookResponseMetadata::External,
                                output.exit_code,
                                output.stdout,
                                output.stderr,
                                response,
                            )),
                            Err(error) => Err(error),
                        }
                    }
                }
            }
            .instrument(span.clone())
            .await;

            let duration_ms = started.elapsed().as_millis().try_into().unwrap_or(u64::MAX);
            let (decision, reason) = match outcome {
                Ok((_metadata, exit_code, stdout, stderr, response)) => {
                    let response_decision = response.decision.as_deref().unwrap_or("");
                    let decision = if exit_code != 0 {
                        match rule.failure_mode {
                            super::config::FailureMode::FailOpen => HookDecision::Allow,
                            super::config::FailureMode::FailClosed => HookDecision::Deny,
                        }
                    } else {
                        match response_decision {
                            "allow" => HookDecision::Allow,
                            "deny" => HookDecision::Deny,
                            "modify" => HookDecision::Modify,
                            "error" => HookDecision::Error,
                            _ => HookDecision::Noop,
                        }
                    };
                    if let Some(arguments) = response.modified_arguments {
                        aggregate.modified_arguments = Some(arguments);
                        if context.tool_input.is_none() {
                            context.tool_input = aggregate.modified_arguments.clone();
                        }
                    }
                    if let Some(prompt) = response.modified_prompt {
                        aggregate.modified_prompt = Some(prompt);
                    }
                    if let Some(reminder) = response.reminder {
                        aggregate.reminders.push(reminder);
                    }
                    if let Some(reason) = response.reason {
                        aggregate.reason = Some(reason);
                    }
                    span.record("hook.exit_code", exit_code);
                    span.record("hook.stdout", stdout);
                    span.record("hook.stderr", stderr);
                    span.record("hook.duration_ms", duration_ms);
                    span.record("hook.decision", decision.as_str());
                    info!(
                        parent: &span,
                        hook.id = %rule.id,
                        hook.decision = decision.as_str(),
                        hook.exit_code = exit_code,
                        hook.duration_ms = duration_ms,
                        "Hook rule completed"
                    );
                    (decision, aggregate.reason.clone())
                }
                Err(error) => {
                    let decision = match rule.failure_mode {
                        super::config::FailureMode::FailOpen => HookDecision::Allow,
                        super::config::FailureMode::FailClosed => HookDecision::Deny,
                    };
                    span.record("hook.exit_code", -1i64);
                    span.record("hook.duration_ms", duration_ms);
                    span.record("hook.decision", HookDecision::Error.as_str());
                    error!(
                        parent: &span,
                        hook.id = %rule.id,
                        hook.decision = HookDecision::Error.as_str(),
                        error = %error,
                        "Hook rule failed"
                    );
                    (decision, Some(error.to_string()))
                }
            };

            if decision == HookDecision::Deny
                || decision == HookDecision::Error && aggregate.decision != HookDecision::Deny
            {
                aggregate.decision = decision;
            } else if aggregate.decision == HookDecision::Allow
                && (decision == HookDecision::Modify || decision == HookDecision::Noop)
            {
                aggregate.decision = decision;
            }
            if let Some(reason) = reason {
                aggregate.reason = Some(reason);
            }
        }

        aggregate
    }

    pub(crate) async fn session_start(&self) -> HookDispatchResult {
        self.dispatch(HookContext {
            event: HookEvent::SessionStart.as_name().to_owned(),
            tool_name: None,
            tool_input: None,
            tool_output: None,
            prompt: None,
            subagent_name: None,
        })
        .await
    }

    pub(crate) async fn session_end(&self) -> HookDispatchResult {
        self.dispatch(HookContext {
            event: HookEvent::SessionEnd.as_name().to_owned(),
            tool_name: None,
            tool_input: None,
            tool_output: None,
            prompt: None,
            subagent_name: None,
        })
        .await
    }
}

#[derive(Debug, Clone, Copy)]
enum HookResponseMetadata {
    Native,
    External,
}

impl std::str::FromStr for HookEvent {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "SessionStart" => Ok(Self::SessionStart),
            "SessionEnd" => Ok(Self::SessionEnd),
            "UserPromptSubmit" => Ok(Self::UserPromptSubmit),
            "PreToolUse" => Ok(Self::PreToolUse),
            "PostToolUse" => Ok(Self::PostToolUse),
            "SubagentStart" => Ok(Self::SubagentStart),
            "SubagentStop" => Ok(Self::SubagentStop),
            _ => Err(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};
    use tracing::instrument::WithSubscriber;
    use tracing_subscriber::prelude::*;

    fn exporter_provider() -> (InMemorySpanExporter, SdkTracerProvider) {
        let exporter = InMemorySpanExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        (exporter, provider)
    }

    fn rule(event: HookEvent, command: &str, args: Vec<String>) -> HookRule {
        HookRule {
            id: "test-hook".to_owned(),
            name: "Test hook".to_owned(),
            event,
            matcher: HookMatcher::All,
            timeout_ms: 1000,
            failure_mode: super::super::config::FailureMode::FailOpen,
            backend: HookBackend::External {
                command: command.to_owned(),
                args,
                payload_format: super::super::config::HookPayloadFormat::Json,
            },
        }
    }

    #[tokio::test]
    async fn external_pre_tool_hook_can_block_and_emit_stderr() {
        let (exporter, provider) = exporter_provider();
        let tracer = provider.tracer("hook-test");
        let subscriber =
            tracing_subscriber::registry().with(tracing_opentelemetry::layer().with_tracer(tracer));
        let dispatcher = HookDispatcher::from_config(HookConfig {
                rules: vec![rule(
                    HookEvent::PreToolUse,
                    "/bin/sh",
                    vec![
                        "-c".to_owned(),
                        "read _; printf '{\"decision\":\"deny\",\"reason\":\"blocked\",\"reminder\":\"stop now\"}' >&1; printf 'diagnostic stderr' >&2".to_owned(),
                    ],
                )],
            });

        let result = dispatcher
            .dispatch(HookContext {
                event: HookEvent::PreToolUse.as_name().to_owned(),
                tool_name: Some("bash".to_owned()),
                tool_input: Some(r#"{"command":"rm -rf /"}"#.to_owned()),
                tool_output: None,
                prompt: None,
                subagent_name: None,
            })
            .with_subscriber(subscriber)
            .await;

        assert_eq!(result.decision, HookDecision::Deny);
        assert_eq!(result.reason.as_deref(), Some("blocked"));
        provider.force_flush().unwrap();
        let spans = exporter.get_finished_spans().unwrap();
        assert!(
            spans
                .iter()
                .any(|span| span.name.as_ref() == "hook.dispatch")
        );
        assert!(spans.iter().any(|span| span.name.as_ref() == "hook.rule"));
    }

    #[tokio::test]
    async fn external_hook_timeout_kills_process_and_is_observed() {
        let (exporter, provider) = exporter_provider();
        let tracer = provider.tracer("hook-test");
        let subscriber =
            tracing_subscriber::registry().with(tracing_opentelemetry::layer().with_tracer(tracer));
        let mut timed_rule = rule(
            HookEvent::PreToolUse,
            "/bin/sh",
            vec!["-c".to_owned(), "read _; sleep 10".to_owned()],
        );
        timed_rule.timeout_ms = 20;
        timed_rule.failure_mode = super::super::config::FailureMode::FailOpen;
        let dispatcher = HookDispatcher::from_config(HookConfig {
            rules: vec![timed_rule],
        });

        let result = dispatcher
            .dispatch(HookContext {
                event: HookEvent::PreToolUse.as_name().to_owned(),
                tool_name: Some("bash".to_owned()),
                tool_input: Some("{}".to_owned()),
                tool_output: None,
                prompt: None,
                subagent_name: None,
            })
            .with_subscriber(subscriber)
            .await;
        assert_eq!(result.decision, HookDecision::Allow);
        provider.force_flush().unwrap();
        assert!(
            exporter
                .get_finished_spans()
                .unwrap()
                .iter()
                .any(|span| span.name.as_ref() == "hook.rule")
        );
    }

    #[tokio::test]
    async fn post_tool_hook_adds_follow_up_reminder() {
        let (exporter, provider) = exporter_provider();
        let tracer = provider.tracer("hook-test");
        let subscriber =
            tracing_subscriber::registry().with(tracing_opentelemetry::layer().with_tracer(tracer));
        let dispatcher = HookDispatcher::from_config(HookConfig {
            rules: vec![rule(
                HookEvent::PostToolUse,
                "/bin/sh",
                vec![
                    "-c".to_owned(),
                    "read _; printf '{\"decision\":\"noop\",\"reminder\":\"verify the output\"}'"
                        .to_owned(),
                ],
            )],
        });

        let result = dispatcher
            .dispatch(HookContext {
                event: HookEvent::PostToolUse.as_name().to_owned(),
                tool_name: Some("read_file".to_owned()),
                tool_input: Some("{}".to_owned()),
                tool_output: Some("content".to_owned()),
                prompt: None,
                subagent_name: None,
            })
            .with_subscriber(subscriber)
            .await;
        assert_eq!(result.reminders, vec!["verify the output".to_owned()]);
        provider.force_flush().unwrap();
        assert!(
            exporter
                .get_finished_spans()
                .unwrap()
                .iter()
                .any(|span| span.name.as_ref() == "hook.dispatch")
        );
    }
}
