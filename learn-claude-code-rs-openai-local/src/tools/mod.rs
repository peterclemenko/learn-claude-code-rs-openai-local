mod bash;
mod file;
mod memory;
mod system_prompt;
mod todo;

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::Result;
use async_openai::types::chat::{ChatCompletionTool, ChatCompletionTools, FunctionObject};
use async_trait::async_trait;
use tracing::{Instrument, info, instrument};

use crate::{
    error_recovery::{ErrorKind, ErrorRecoveryEngine, RecoveryAction},
    hooks::{HookContext, HookDecision, HookDispatcher, HookEvent},
    permission::{PermissionBehavior, PermissionManager},
    types::toon_table,
};

pub use bash::BashTool;
pub use file::ReadFileTool;
pub use memory::MemoryTool;
pub use system_prompt::SelectSystemPromptTool;
pub use todo::TodoTool;

#[async_trait]
pub trait ToonTool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn parameters_signature(&self) -> &'static str;
    fn result_signature(&self) -> &'static str;
    async fn execute(&self, payload: &str) -> Result<String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolExecutionMetrics {
    pub tool: String,
    pub duration_ms: u64,
    pub input_bytes: usize,
    pub estimated_input_tokens: usize,
    pub output_bytes: usize,
    pub estimated_output_tokens: usize,
    pub succeeded: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolExecution {
    pub output: String,
    pub metrics: ToolExecutionMetrics,
}

fn estimated_tokens(bytes: usize) -> usize {
    bytes.div_ceil(4)
}

pub struct ToolRegistry {
    tools: BTreeMap<&'static str, Arc<dyn ToonTool>>,
    permissions: Option<Arc<Mutex<PermissionManager>>>,
    hooks: Arc<HookDispatcher>,
    recovery: Arc<ErrorRecoveryEngine>,
    failure_attempts: Mutex<BTreeMap<String, u32>>,
}

impl ToolRegistry {
    #[cfg(test)]
    pub fn new() -> Self {
        Self::with_recovery(Arc::new(ErrorRecoveryEngine::default()))
    }

    #[cfg(test)]
    fn with_recovery(recovery: Arc<ErrorRecoveryEngine>) -> Self {
        Self {
            tools: BTreeMap::new(),
            permissions: None,
            hooks: Arc::new(HookDispatcher::disabled()),
            recovery,
            failure_attempts: Mutex::new(BTreeMap::new()),
        }
    }

    pub(crate) fn with_permissions_and_hooks(
        permissions: Arc<Mutex<PermissionManager>>,
        hooks: Arc<HookDispatcher>,
        recovery: Arc<ErrorRecoveryEngine>,
    ) -> Self {
        Self {
            tools: BTreeMap::new(),
            permissions: Some(permissions),
            hooks,
            recovery,
            failure_attempts: Mutex::new(BTreeMap::new()),
        }
    }

    pub fn register(&mut self, tool: impl ToonTool + 'static) {
        self.tools.insert(tool.name(), Arc::new(tool));
    }

    pub fn openai_tools(&self) -> Vec<ChatCompletionTools> {
        self.tools
            .values()
            .map(|tool| {
                ChatCompletionTools::Function(ChatCompletionTool {
                    function: FunctionObject {
                        name: tool.name().to_owned(),
                        description: Some(format!(
                            "{} Input signature: {} Output signature: {}",
                            tool.description(),
                            tool.parameters_signature(),
                            tool.result_signature()
                        )),
                        parameters: None,
                        strict: None,
                    },
                })
            })
            .collect()
    }

    #[instrument(name = "tool.dispatch", skip(self, payload), fields(tool = name))]
    pub async fn dispatch(&self, name: &str, payload: &str) -> Result<String> {
        Ok(self.dispatch_with_metrics(name, payload).await?.output)
    }

    pub async fn dispatch_with_metrics(&self, name: &str, payload: &str) -> Result<ToolExecution> {
        let Some(tool) = self.tools.get(name) else {
            let span = tracing::info_span!(
                "tool.execution",
                tool = name,
                input_bytes = payload.len(),
                estimated_input_tokens = estimated_tokens(payload.len()),
                output_bytes = tracing::field::Empty,
                estimated_output_tokens = tracing::field::Empty,
                duration_ms = tracing::field::Empty,
                succeeded = tracing::field::Empty,
                error.kind = tracing::field::Empty,
                error.recoverable = tracing::field::Empty,
                recovery.action = tracing::field::Empty,
                recovery.attempt = tracing::field::Empty,
                recovery.outcome = tracing::field::Empty,
            );
            return self
                .handle_tool_error(name, payload, format!("Unknown tool '{name}'"), 0, &span)
                .instrument(span.clone())
                .await;
        };
        let mut effective_payload = payload.to_owned();
        let pre_hook = self
            .hooks
            .dispatch(HookContext {
                event: HookEvent::PreToolUse.as_name().to_owned(),
                tool_name: Some(name.to_owned()),
                tool_input: Some(payload.to_owned()),
                tool_output: None,
                prompt: None,
                subagent_name: None,
                recovery: None,
            })
            .await;
        if pre_hook.decision == HookDecision::Deny {
            let reason = pre_hook
                .reason
                .unwrap_or_else(|| "Denied by PreToolUse hook".to_owned());
            info!(tool = name, reason = %reason, "Tool call denied by hook");
            let output = format!("PreToolUse denied tool '{name}': {reason}");
            let metrics = ToolExecutionMetrics {
                tool: name.to_owned(),
                duration_ms: 0,
                input_bytes: payload.len(),
                estimated_input_tokens: estimated_tokens(payload.len()),
                output_bytes: output.len(),
                estimated_output_tokens: estimated_tokens(output.len()),
                succeeded: false,
            };
            return Ok(ToolExecution { output, metrics });
        }
        if let Some(arguments) = pre_hook.modified_arguments {
            effective_payload = arguments;
            info!(tool = name, "Tool input modified by PreToolUse hook");
        }

        let (allowed, denial_reason) = if let Some(permissions) = &self.permissions {
            let mut permissions = permissions.lock().map_err(|error| {
                anyhow::anyhow!("Permission manager mutex is poisoned: {error}")
            })?;
            let (behavior, reason) = permissions.check(name, &effective_payload);
            match behavior {
                PermissionBehavior::Allow => (true, None),
                PermissionBehavior::Deny => {
                    info!(tool = name, reason = %reason, "Tool call denied by permission rules");
                    (false, Some(reason))
                }
                PermissionBehavior::Ask => {
                    let allowed = permissions.ask_user(name, &effective_payload)?;
                    if !allowed {
                        info!(tool = name, "Tool call denied by user");
                    }
                    (allowed, (!allowed).then(|| "Denied by user".to_owned()))
                }
            }
        } else {
            (true, None)
        };

        let input_bytes = effective_payload.len();
        let span = tracing::info_span!(
            "tool.execution",
            tool = name,
            input_bytes,
            estimated_input_tokens = estimated_tokens(input_bytes),
            output_bytes = tracing::field::Empty,
            estimated_output_tokens = tracing::field::Empty,
            duration_ms = tracing::field::Empty,
            succeeded = tracing::field::Empty,
            error.kind = tracing::field::Empty,
            error.recoverable = tracing::field::Empty,
            recovery.action = tracing::field::Empty,
            recovery.attempt = tracing::field::Empty,
            recovery.outcome = tracing::field::Empty,
        );
        let started_at = Instant::now();
        let result = if allowed {
            tool.execute(&effective_payload)
                .instrument(span.clone())
                .await
        } else {
            Ok(format!(
                "Permission denied for tool '{name}': {}",
                denial_reason.unwrap_or_else(|| "No reason provided".to_owned())
            ))
        };
        let duration_ms = started_at
            .elapsed()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX);
        let output = match result {
            Ok(output) => {
                if allowed {
                    self.reset_failure_attempts(name)?;
                }
                output
            }
            Err(tool_error) => {
                return self
                    .handle_tool_error(
                        name,
                        &effective_payload,
                        tool_error.to_string(),
                        duration_ms,
                        &span,
                    )
                    .instrument(span.clone())
                    .await;
            }
        };
        let post_hook = self
            .hooks
            .dispatch(HookContext {
                event: HookEvent::PostToolUse.as_name().to_owned(),
                tool_name: Some(name.to_owned()),
                tool_input: Some(effective_payload.clone()),
                tool_output: Some(output.clone()),
                prompt: None,
                subagent_name: None,
                recovery: None,
            })
            .await;
        let mut output = output;
        for reminder in post_hook.reminders {
            output.push_str("\n\nHook reminder (TOON):\n");
            output.push_str(&reminder);
        }
        let output_bytes = output.len();
        let estimated_output_tokens = estimated_tokens(output_bytes);
        let metrics = ToolExecutionMetrics {
            tool: name.to_owned(),
            duration_ms,
            input_bytes,
            estimated_input_tokens: estimated_tokens(input_bytes),
            output_bytes,
            estimated_output_tokens,
            succeeded: allowed,
        };
        span.record("output_bytes", metrics.output_bytes);
        span.record("estimated_output_tokens", metrics.estimated_output_tokens);
        span.record("duration_ms", metrics.duration_ms);
        span.record("succeeded", metrics.succeeded);
        info!(
            parent: &span,
            tool = name,
            duration_ms = metrics.duration_ms,
            input_bytes = metrics.input_bytes,
            estimated_input_tokens = metrics.estimated_input_tokens,
            output_bytes = metrics.output_bytes,
            estimated_output_tokens = metrics.estimated_output_tokens,
            succeeded = metrics.succeeded,
            "Tool execution usage"
        );
        Ok(ToolExecution { output, metrics })
    }

    fn next_failure_attempt(&self, name: &str) -> Result<u32> {
        let mut attempts = self
            .failure_attempts
            .lock()
            .map_err(|error| anyhow::anyhow!("Tool recovery attempt state is poisoned: {error}"))?;
        let attempt = attempts.entry(name.to_owned()).or_default();
        *attempt = attempt.saturating_add(1);
        Ok(*attempt)
    }

    fn reset_failure_attempts(&self, name: &str) -> Result<()> {
        self.failure_attempts
            .lock()
            .map_err(|error| anyhow::anyhow!("Tool recovery attempt state is poisoned: {error}"))?
            .remove(name);
        Ok(())
    }

    async fn handle_tool_error(
        &self,
        name: &str,
        payload: &str,
        message: String,
        duration_ms: u64,
        span: &tracing::Span,
    ) -> Result<ToolExecution> {
        let attempt = self.next_failure_attempt(name)?;
        let kind = ErrorKind::classify_tool_error(&message);
        let action = self.recovery.tool_action(kind, attempt);
        let outcome = action.outcome();
        let context_toon = toon_table(
            "recovery_context",
            &["tool", "error_kind", "attempt", "action", "outcome"],
            &[
                name.to_owned(),
                kind.as_str().to_owned(),
                attempt.to_string(),
                action.as_str().to_owned(),
                outcome.to_owned(),
            ],
        );
        let diagnostics = crate::hooks::RecoveryDiagnostics {
            error_kind: kind.as_str().to_owned(),
            recoverable: kind.is_recoverable(),
            recovery_action: action.as_str().to_owned(),
            attempt,
            outcome: outcome.to_owned(),
            message: message.clone(),
            context_toon: context_toon.clone(),
        };
        span.record("error.kind", kind.as_str());
        span.record("error.recoverable", kind.is_recoverable());
        span.record("recovery.action", action.as_str());
        span.record("recovery.attempt", i64::from(attempt));
        span.record("recovery.outcome", outcome);
        self.recovery
            .notify(
                &self.hooks,
                HookEvent::Error,
                Some(name.to_owned()),
                diagnostics.clone(),
            )
            .await;
        self.recovery
            .notify(
                &self.hooks,
                HookEvent::Recovery,
                Some(name.to_owned()),
                diagnostics.clone(),
            )
            .await;

        let mut output = toon_table("tool_error", &["error"], &[message.clone()]);
        output.push('\n');
        output.push_str(&context_toon);
        let post_hook = self
            .hooks
            .dispatch(HookContext {
                event: HookEvent::PostToolUse.as_name().to_owned(),
                tool_name: Some(name.to_owned()),
                tool_input: Some(payload.to_owned()),
                tool_output: Some(output.clone()),
                prompt: None,
                subagent_name: None,
                recovery: Some(diagnostics),
            })
            .await;
        for reminder in post_hook.reminders {
            output.push_str("\n\nHook reminder (TOON):\n");
            output.push_str(&reminder);
        }

        let metrics = ToolExecutionMetrics {
            tool: name.to_owned(),
            duration_ms,
            input_bytes: payload.len(),
            estimated_input_tokens: estimated_tokens(payload.len()),
            output_bytes: output.len(),
            estimated_output_tokens: estimated_tokens(output.len()),
            succeeded: false,
        };
        span.record("output_bytes", metrics.output_bytes);
        span.record("estimated_output_tokens", metrics.estimated_output_tokens);
        span.record("duration_ms", metrics.duration_ms);
        span.record("succeeded", false);
        info!(
            parent: span,
            tool = name,
            duration_ms = metrics.duration_ms,
            input_bytes = metrics.input_bytes,
            estimated_input_tokens = metrics.estimated_input_tokens,
            output_bytes = metrics.output_bytes,
            estimated_output_tokens = metrics.estimated_output_tokens,
            succeeded = false,
            "Tool execution usage"
        );

        if action == RecoveryAction::Escalate {
            anyhow::bail!("Tool '{name}' recovery exhausted after {attempt} failures: {message}");
        }
        Ok(ToolExecution { output, metrics })
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fmt::Write as _,
        fs,
        sync::{Arc, Mutex},
        time::SystemTime,
    };

    use anyhow::{Result, bail};
    use async_openai::types::chat::ChatCompletionTools;
    use async_trait::async_trait;
    use tracing::{Event, Subscriber, field::Visit, instrument::WithSubscriber};
    use tracing_subscriber::{Layer, layer::Context, prelude::*};

    use super::{BashTool, ReadFileTool, TodoTool, ToolExecutionMetrics, ToolRegistry, ToonTool};
    use crate::{
        error_recovery::{ErrorRecoveryConfig, ErrorRecoveryEngine},
        hooks::HookDispatcher,
        permission::PermissionManager,
        types::TodoState,
    };

    fn assert_usage(metrics: &ToolExecutionMetrics, name: &str, input: &str, output: &str) {
        assert_eq!(metrics.tool, name);
        assert!(metrics.succeeded);
        assert_eq!(metrics.input_bytes, input.len());
        assert_eq!(metrics.estimated_input_tokens, input.len().div_ceil(4));
        assert_eq!(metrics.output_bytes, output.len());
        assert_eq!(metrics.estimated_output_tokens, output.len().div_ceil(4));
    }

    fn builtin_registry() -> (ToolRegistry, Arc<Mutex<TodoState>>) {
        let state = Arc::new(Mutex::new(TodoState::default()));
        let mut registry = ToolRegistry::new();
        registry.register(BashTool);
        registry.register(ReadFileTool);
        registry.register(TodoTool::new(Arc::clone(&state)));
        (registry, state)
    }

    #[test]
    fn openai_definitions_advertise_toon_signatures_without_json_schema() {
        let mut registry = ToolRegistry::new();
        registry.register(BashTool);
        registry.register(ReadFileTool);
        registry.register(TodoTool::new(Arc::new(Mutex::new(TodoState::default()))));
        let tools = registry.openai_tools();

        assert_eq!(tools.len(), 3);
        assert!(matches!(tools[0], ChatCompletionTools::Function(_)));
        let ChatCompletionTools::Function(function_tool) = &tools[0] else {
            panic!("Expected a function tool at index 0");
        };
        assert!(function_tool.function.parameters.is_none());
        assert!(
            function_tool
                .function
                .description
                .as_deref()
                .unwrap()
                .contains("bash[1]{command}:")
        );
    }

    #[tokio::test]
    async fn registry_reports_unknown_tools() {
        let registry = ToolRegistry::new();
        let result = registry.dispatch_with_metrics("missing", "").await.unwrap();
        assert!(result.output.contains("Unknown tool 'missing'"));
        assert!(result.output.contains("recovery_context[1]"));
        assert!(!result.metrics.succeeded);
    }

    #[tokio::test]
    async fn permission_denial_prevents_tool_execution() {
        crate::test_telemetry::run_async_test_case(
            "tools::tests::permission_denial_prevents_tool_execution",
            async {
                let permissions_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .unwrap()
                    .join("config/permissions");
                let permissions = Arc::new(Mutex::new(
                    PermissionManager::load(permissions_dir).unwrap(),
                ));
                permissions
                    .lock()
                    .unwrap()
                    .handle_command("/profile strict")
                    .unwrap();
                let mut registry = ToolRegistry::with_permissions_and_hooks(
                    permissions,
                    Arc::new(HookDispatcher::disabled()),
                    Arc::new(ErrorRecoveryEngine::default()),
                );
                registry.register(BashTool);

                let execution = registry
                    .dispatch_with_metrics("bash", r#"{"command":"printf 'should-not-run'"}"#)
                    .await
                    .unwrap();

                assert!(execution.output.contains("Permission denied"));
                assert!(!execution.output.contains("should-not-run"));
                assert!(!execution.metrics.succeeded);
            },
        )
        .await;
    }

    #[tokio::test]
    async fn bash_executes_command_and_reports_execution_usage() {
        let (registry, _) = builtin_registry();
        let bash_input = r#"{"command":"printf 'bash-tool-ok'"}"#;
        let bash = registry
            .dispatch_with_metrics("bash", bash_input)
            .await
            .unwrap();
        assert!(bash.output.contains("bash-tool-ok"));
        assert!(bash.output.contains("0,"));
        assert_usage(&bash.metrics, "bash", bash_input, &bash.output);
    }

    #[tokio::test]
    async fn read_file_returns_contents_and_reports_execution_usage() {
        let (registry, _) = builtin_registry();
        let file_path = std::env::temp_dir().join(format!(
            "learn-claude-code-read-file-test-{}-{}.txt",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let file_content = "file-tool-content\n";
        fs::write(&file_path, file_content).unwrap();
        let file_input = serde_json::json!({"path": file_path}).to_string();
        let file = registry
            .dispatch_with_metrics("read_file", &file_input)
            .await
            .unwrap();
        assert!(file.output.contains(file_content.trim_end()));
        assert_usage(&file.metrics, "read_file", &file_input, &file.output);
        fs::remove_file(file_path).unwrap();
    }

    #[tokio::test]
    async fn todo_updates_shared_state_and_reports_execution_usage() {
        let (registry, state) = builtin_registry();
        let todo_input =
            r#"{"tasks":[{"id":1,"description":"Run tool tests","status":"Completed"}]}"#;
        let todo = registry
            .dispatch_with_metrics("todo_write", todo_input)
            .await
            .unwrap();
        assert!(todo.output.contains("1,Run tool tests,Completed"));
        assert_eq!(state.lock().unwrap().tasks.len(), 1);
        assert_usage(&todo.metrics, "todo_write", todo_input, &todo.output);
    }

    #[tokio::test]
    async fn registry_reports_usage_for_future_tools_and_failed_executions() {
        let mut registry = ToolRegistry::new();
        registry.register(ProbeTool {
            fail: false,
            message: "probe failure",
        });
        let input = "probe input";
        let output = registry
            .dispatch_with_metrics("probe", input)
            .await
            .unwrap();
        assert_eq!(output.output, "probe output");
        assert_usage(&output.metrics, "probe", input, "probe output");

        let mut registry = ToolRegistry::new();
        registry.register(ProbeTool {
            fail: true,
            message: "probe failure",
        });
        let failure = registry
            .dispatch_with_metrics("probe", input)
            .await
            .unwrap();
        assert!(!failure.metrics.succeeded);
        assert!(failure.output.contains("tool_error[1]{error}"));
    }

    #[tokio::test]
    async fn tool_errors_feed_toon_correction_then_escalate_after_budget() {
        let mut recovery_config = ErrorRecoveryConfig::default();
        recovery_config.tool.max_attempts = 1;
        let recovery = Arc::new(ErrorRecoveryEngine::new(recovery_config).unwrap());
        let mut registry = ToolRegistry::with_recovery(recovery);
        registry.register(ProbeTool {
            fail: true,
            message: "probe failure",
        });

        let feedback = registry
            .dispatch_with_metrics("probe", "bad input")
            .await
            .unwrap();
        assert!(!feedback.metrics.succeeded);
        assert!(feedback.output.contains("tool_error[1]{error}"));
        assert!(feedback.output.contains("recovery_context[1]"));
        assert!(feedback.output.contains("feedback_injected"));

        let escalated = registry.dispatch_with_metrics("probe", "bad input").await;
        assert!(escalated.is_err());

        let mut parse_config = ErrorRecoveryConfig::default();
        parse_config.parse.max_attempts = 1;
        let mut registry =
            ToolRegistry::with_recovery(Arc::new(ErrorRecoveryEngine::new(parse_config).unwrap()));
        registry.register(ProbeTool {
            fail: true,
            message: "Failed to parse tool arguments",
        });
        let feedback = registry
            .dispatch_with_metrics("probe", "malformed arguments")
            .await
            .unwrap();
        assert!(feedback.output.contains("parse"));
        assert!(feedback.output.contains("feedback_injected"));
    }

    #[tokio::test]
    async fn dispatch_emits_tool_usage_in_a_tracing_span() {
        let _subscriber_lock = crate::test_telemetry::lock_test_subscriber();
        let captured = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry().with(CaptureLayer(Arc::clone(&captured)));
        let dispatch = tracing::Dispatch::new(subscriber);
        let mut registry = ToolRegistry::new();
        registry.register(ProbeTool {
            fail: false,
            message: "probe failure",
        });

        registry
            .dispatch_with_metrics("probe", "test input")
            .with_subscriber(dispatch)
            .await
            .unwrap();

        let captured = captured.lock().unwrap();
        assert!(
            captured
                .iter()
                .any(|entry| entry.contains("tool.execution"))
        );
        assert!(
            captured.iter().any(|entry| {
                entry.contains("Tool execution usage")
                    && entry.contains("estimated_input_tokens")
                    && entry.contains("estimated_output_tokens")
                    && entry.contains("duration_ms")
            }),
            "Captured tracing records: {captured:?}"
        );
    }

    #[derive(Clone)]
    struct CaptureLayer(Arc<Mutex<Vec<String>>>);

    impl<S: Subscriber> Layer<S> for CaptureLayer {
        fn on_new_span(
            &self,
            attributes: &tracing::span::Attributes<'_>,
            _id: &tracing::span::Id,
            _context: Context<'_, S>,
        ) {
            self.0
                .lock()
                .unwrap()
                .push(attributes.metadata().name().to_owned());
        }

        fn on_event(&self, event: &Event<'_>, _context: Context<'_, S>) {
            let mut fields = CapturedFields::default();
            event.record(&mut fields);
            self.0.lock().unwrap().push(fields.0);
        }
    }

    #[derive(Default)]
    struct CapturedFields(String);

    impl Visit for CapturedFields {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            let _ = write!(self.0, "{}={value:?};", field.name());
        }
    }

    struct ProbeTool {
        fail: bool,
        message: &'static str,
    }

    #[async_trait]
    impl ToonTool for ProbeTool {
        fn name(&self) -> &'static str {
            "probe"
        }

        fn description(&self) -> &'static str {
            "Test-only probe tool."
        }

        fn parameters_signature(&self) -> &'static str {
            "probe[1]{input}:"
        }

        fn result_signature(&self) -> &'static str {
            "probe_result[1]{output}:"
        }

        async fn execute(&self, _payload: &str) -> Result<String> {
            if self.fail {
                bail!("{}", self.message);
            }
            Ok("probe output".to_owned())
        }
    }
}
