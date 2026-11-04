mod bash;
mod file;
mod todo;

use std::{collections::BTreeMap, sync::Arc, time::Instant};

use anyhow::Result;
use async_openai::types::chat::{ChatCompletionTool, ChatCompletionTools, FunctionObject};
use async_trait::async_trait;
use tracing::{Instrument, error, info, instrument};

use crate::types::toon_table;

pub use bash::BashTool;
pub use file::ReadFileTool;
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

#[derive(Default)]
pub struct ToolRegistry {
    tools: BTreeMap<&'static str, Arc<dyn ToonTool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
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
        let tool = self
            .tools
            .get(name)
            .with_context(|| format!("Unknown tool '{name}'"))?;

        let input_bytes = payload.len();
        let span = tracing::info_span!(
            "tool.execution",
            tool = name,
            input_bytes,
            estimated_input_tokens = estimated_tokens(input_bytes),
            output_bytes = tracing::field::Empty,
            estimated_output_tokens = tracing::field::Empty,
            duration_ms = tracing::field::Empty,
            succeeded = tracing::field::Empty,
        );
        let started_at = Instant::now();
        let result = tool.execute(payload).instrument(span.clone()).await;
        let duration_ms = started_at
            .elapsed()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX);
        let (output, succeeded) = match result {
            Ok(output) => (output, true),
            Err(tool_error) => {
                error!(parent: &span, tool = name, error = %tool_error, "Tool execution failed");
                (
                    toon_table("tool_error", &["error"], &[tool_error.to_string()]),
                    false,
                )
            }
        };
        let output_bytes = output.len();
        let estimated_output_tokens = estimated_tokens(output_bytes);
        let metrics = ToolExecutionMetrics {
            tool: name.to_owned(),
            duration_ms,
            input_bytes,
            estimated_input_tokens: estimated_tokens(input_bytes),
            output_bytes,
            estimated_output_tokens,
            succeeded,
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
}

use anyhow::Context;

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
    use crate::types::TodoState;

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
        let result = registry.dispatch("missing", "").await;
        assert!(result.is_err());
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
        registry.register(ProbeTool { fail: false });
        let input = "probe input";
        let output = registry
            .dispatch_with_metrics("probe", input)
            .await
            .unwrap();
        assert_eq!(output.output, "probe output");
        assert_usage(&output.metrics, "probe", input, "probe output");

        let mut registry = ToolRegistry::new();
        registry.register(ProbeTool { fail: true });
        let failure = registry
            .dispatch_with_metrics("probe", input)
            .await
            .unwrap();
        assert!(!failure.metrics.succeeded);
        assert!(failure.output.contains("tool_error[1]{error}"));
    }

    #[tokio::test]
    async fn dispatch_emits_tool_usage_in_a_tracing_span() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry().with(CaptureLayer(Arc::clone(&captured)));
        let dispatch = tracing::Dispatch::new(subscriber);
        let mut registry = ToolRegistry::new();
        registry.register(ProbeTool { fail: false });

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
                bail!("probe failure");
            }
            Ok("probe output".to_owned())
        }
    }
}
