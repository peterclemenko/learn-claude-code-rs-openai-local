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
        let mut native = NativeHookRegistry::default();
        native.register("noop", NoopNativeHook);
        Self {
            rules: Arc::new(config.rules),
            native: Arc::new(NativeHookRegistry::default()),
        }
        .with_native(native)
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
        fields(
            hook.event = %context.event,
            hook.tool_name = tracing::field::Empty,
            hook.decision = tracing::field::Empty,
            hook.duration_ms = tracing::field::Empty,
        )
    )]
    pub(crate) async fn dispatch(&self, mut context: HookContext) -> HookDispatchResult {
        let dispatch_started = Instant::now();
        if let Some(tool_name) = &context.tool_name {
            tracing::Span::current().record("hook.tool_name", tool_name.as_str());
        }
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
                                    0i64,
                                    String::new(),
                                    String::new(),
                                    started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
                                    false,
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
                                output.exit_code,
                                output.stdout,
                                output.stderr,
                                output.duration_ms,
                                output.timed_out,
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
                Ok((exit_code, stdout, stderr, hook_duration_ms, timed_out, response)) => {
                    let response_decision = response.decision.as_deref().unwrap_or("");
                    let decision = if timed_out || exit_code != 0 {
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
                    span.record("hook.stdout", stdout.as_str());
                    span.record("hook.stderr", stderr.as_str());
                    span.record("hook.duration_ms", hook_duration_ms);
                    span.record("hook.timed_out", timed_out);
                    span.record("hook.decision", decision.as_str());
                    info!(
                        parent: &span,
                        hook.id = %rule.id,
                        hook.decision = decision.as_str(),
                        hook.exit_code = exit_code,
                        hook.duration_ms = hook_duration_ms,
                        hook.timed_out = timed_out,
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
                    span.record("hook.timed_out", false);
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

        let duration_ms = dispatch_started
            .elapsed()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX);
        let span = tracing::Span::current();
        span.record("hook.decision", aggregate.decision.as_str());
        span.record("hook.duration_ms", duration_ms);
        info!(
            hook.event = %context.event,
            hook.tool_name = ?context.tool_name,
            hook.decision = aggregate.decision.as_str(),
            hook.duration_ms = duration_ms,
            "Hook dispatch completed"
        );
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

struct NoopNativeHook;

#[async_trait]
impl NativeHook for NoopNativeHook {
    async fn execute(&self, _context: HookContext) -> anyhow::Result<HookResponse> {
        Ok(HookResponse::default())
    }
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
mod otel_hook_tests {
    use super::*;
    use opentelemetry::Value;
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider, SpanData};
    use tracing::instrument::WithSubscriber;
    use tracing_subscriber::prelude::*;

    fn exporter_provider() -> (InMemorySpanExporter, SdkTracerProvider) {
        let exporter = InMemorySpanExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        (exporter, provider)
    }

    fn otel_only_subscriber(
        provider: &SdkTracerProvider,
    ) -> impl tracing::Subscriber + Send + Sync + 'static {
        tracing_subscriber::registry()
            .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("otel-hook-tests")))
    }

    fn flush_and_shutdown(
        exporter: &InMemorySpanExporter,
        provider: &SdkTracerProvider,
    ) -> Vec<SpanData> {
        provider.force_flush().unwrap();
        let spans = exporter.get_finished_spans().unwrap();
        provider.shutdown().unwrap();
        assert!(exporter.is_shutdown_called());
        spans
    }

    fn attribute<'a>(span: &'a SpanData, name: &str) -> Option<&'a Value> {
        span.attributes
            .iter()
            .find(|attribute| attribute.key.as_str() == name)
            .map(|attribute| &attribute.value)
    }

    fn assert_string_attribute(span: &SpanData, name: &str, expected: &str) {
        let actual = attribute(span, name);
        assert!(
            matches!(actual, Some(Value::String(value)) if value.as_str() == expected),
            "attribute {name} expected {expected:?}, got {actual:?}"
        );
    }

    fn assert_present(span: &SpanData, name: &str) {
        assert!(
            attribute(span, name).is_some(),
            "missing OTel attribute {name}"
        );
    }

    fn dispatch_span<'a>(spans: &'a [SpanData], event: &str, tool_name: &str) -> &'a SpanData {
        spans
            .iter()
            .find(|span| {
                span.name.as_ref() == "hook.dispatch"
                    && matches!(
                        attribute(span, "hook.event"),
                        Some(Value::String(value)) if value.as_str() == event
                    )
                    && matches!(
                        attribute(span, "hook.tool_name"),
                        Some(Value::String(value)) if value.as_str() == tool_name
                    )
            })
            .expect("hook.dispatch span with expected event and tool name")
    }

    fn external_span<'a>(spans: &'a [SpanData], stderr: &str) -> &'a SpanData {
        spans
            .iter()
            .find(|span| {
                span.name.as_ref() == "hook.run_external"
                    && matches!(
                        attribute(span, "hook.stderr"),
                        Some(Value::String(value)) if value.as_str() == stderr
                    )
            })
            .expect("external hook runner span")
    }

    fn timed_out_external_span(spans: &[SpanData]) -> &SpanData {
        spans
            .iter()
            .find(|span| {
                span.name.as_ref() == "hook.run_external"
                    && matches!(attribute(span, "hook.timed_out"), Some(Value::Bool(true)))
            })
            .expect("timed-out external hook runner span")
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
    async fn otel_pre_tool_dispatch_records_allow_block_and_modify() {
        let (exporter, provider) = exporter_provider();
        let dispatch = |response: serde_json::Value, stderr: &str| {
            let response = response.to_string();
            let script = format!("read _; printf '%s' '{response}'; printf '%s' '{stderr}' >&2");
            HookDispatcher::from_config(HookConfig {
                rules: vec![rule(
                    HookEvent::PreToolUse,
                    "/bin/sh",
                    vec!["-c".to_owned(), script],
                )],
            })
        };
        let context = |tool_name: &str| HookContext {
            event: HookEvent::PreToolUse.as_name().to_owned(),
            tool_name: Some(tool_name.to_owned()),
            tool_input: Some(r#"{"value":1}"#.to_owned()),
            tool_output: None,
            prompt: None,
            subagent_name: None,
        };

        let allow = dispatch(serde_json::json!({"decision":"allow"}), "allow stderr")
            .dispatch(context("bash"))
            .with_subscriber(otel_only_subscriber(&provider))
            .await;
        let block = dispatch(
            serde_json::json!({"decision":"deny","reason":"blocked"}),
            "block stderr",
        )
        .dispatch(context("read_file"))
        .with_subscriber(otel_only_subscriber(&provider))
        .await;
        let modify = dispatch(
            serde_json::json!({
                "decision": "modify",
                "modified_arguments": r#"{"value":2}"#,
            }),
            "modify stderr",
        )
        .dispatch(context("todo_write"))
        .with_subscriber(otel_only_subscriber(&provider))
        .await;

        assert_eq!(allow.decision, HookDecision::Allow);
        assert_eq!(block.decision, HookDecision::Deny);
        assert_eq!(block.reason.as_deref(), Some("blocked"));
        assert_eq!(modify.decision, HookDecision::Modify);
        assert_eq!(modify.modified_arguments.as_deref(), Some(r#"{"value":2}"#));

        let spans = flush_and_shutdown(&exporter, &provider);
        for (tool_name, decision) in [
            ("bash", "allow"),
            ("read_file", "deny"),
            ("todo_write", "modify"),
        ] {
            let dispatch_span = dispatch_span(&spans, "PreToolUse", tool_name);
            assert_string_attribute(dispatch_span, "hook.decision", decision);
            assert_present(dispatch_span, "hook.duration_ms");
        }
        let runner = external_span(&spans, "modify stderr");
        assert_present(runner, "hook.exit_code");
        assert_present(runner, "hook.duration_ms");
        assert_string_attribute(runner, "hook.stderr", "modify stderr");
    }

    #[tokio::test]
    async fn otel_external_hook_records_exit_status_and_captured_streams() {
        let (exporter, provider) = exporter_provider();
        let response = r#"{"decision":"deny","reason":"nonzero exit"}"#;
        let script =
            format!("read _; printf '%s' '{response}'; printf '%s' 'captured stderr' >&2; exit 7");
        let dispatcher = HookDispatcher::from_config(HookConfig {
            rules: vec![rule(
                HookEvent::PreToolUse,
                "/bin/sh",
                vec!["-c".to_owned(), script],
            )],
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
            .with_subscriber(otel_only_subscriber(&provider))
            .await;

        assert_eq!(result.decision, HookDecision::Allow);
        let spans = flush_and_shutdown(&exporter, &provider);
        let runner = external_span(&spans, "captured stderr");
        assert!(matches!(
            attribute(runner, "hook.exit_code"),
            Some(Value::I64(7))
        ));
        assert_string_attribute(runner, "hook.stdout", response);
        assert_string_attribute(runner, "hook.stderr", "captured stderr");
        assert!(matches!(
            attribute(runner, "hook.timed_out"),
            Some(Value::Bool(false))
        ));
        assert_present(runner, "hook.duration_ms");
    }

    #[tokio::test]
    async fn otel_external_hook_records_timeout() {
        let (exporter, provider) = exporter_provider();
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
            .with_subscriber(otel_only_subscriber(&provider))
            .await;
        assert_eq!(result.decision, HookDecision::Allow);
        let spans = flush_and_shutdown(&exporter, &provider);
        let runner = timed_out_external_span(&spans);
        assert!(matches!(
            attribute(runner, "hook.exit_code"),
            Some(Value::I64(-1))
        ));
        assert!(matches!(
            attribute(runner, "hook.timed_out"),
            Some(Value::Bool(true))
        ));
        assert_present(runner, "hook.duration_ms");
    }

    #[tokio::test]
    async fn otel_post_tool_use_records_external_hook_and_reminder() {
        let (exporter, provider) = exporter_provider();
        let dispatcher = HookDispatcher::from_config(HookConfig {
            rules: vec![rule(
                HookEvent::PostToolUse,
                "/bin/sh",
                vec![
                    "-c".to_owned(),
                    "read _; printf '{\"decision\":\"noop\",\"reminder\":\"verify the output\"}'; printf 'post stderr' >&2"
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
            .with_subscriber(otel_only_subscriber(&provider))
            .await;
        assert_eq!(result.reminders, vec!["verify the output".to_owned()]);
        let spans = flush_and_shutdown(&exporter, &provider);
        let dispatch = dispatch_span(&spans, "PostToolUse", "read_file");
        assert_string_attribute(dispatch, "hook.decision", "noop");
        assert_present(dispatch, "hook.duration_ms");
        let runner = external_span(&spans, "post stderr");
        assert!(matches!(
            attribute(runner, "hook.exit_code"),
            Some(Value::I64(0))
        ));
        assert_string_attribute(
            runner,
            "hook.stdout",
            r#"{"decision":"noop","reminder":"verify the output"}"#,
        );
        assert_string_attribute(runner, "hook.stderr", "post stderr");
    }
}
