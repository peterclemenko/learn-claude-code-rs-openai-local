mod bash;
mod file;

use std::{collections::BTreeMap, sync::Arc};

use anyhow::Result;
use async_openai::types::chat::{ChatCompletionTool, ChatCompletionTools, FunctionObject};
use async_trait::async_trait;
use tracing::{error, instrument};

use crate::types::toon_table;

pub use bash::BashTool;
pub use file::ReadFileTool;

#[async_trait]
pub trait ToonTool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn parameters_signature(&self) -> &'static str;
    fn result_signature(&self) -> &'static str;
    async fn execute(&self, payload: &str) -> Result<String>;
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
        let tool = self
            .tools
            .get(name)
            .with_context(|| format!("Unknown tool '{name}'"))?;

        match tool.execute(payload).await {
            Ok(output) => Ok(output),
            Err(tool_error) => {
                error!(tool = name, error = %tool_error, "Tool execution failed");
                Ok(toon_table(
                    "tool_error",
                    &["error"],
                    &[tool_error.to_string()],
                ))
            }
        }
    }
}

use anyhow::Context;

#[cfg(test)]
mod tests {
    use async_openai::types::chat::ChatCompletionTools;

    use super::{BashTool, ReadFileTool, ToolRegistry};

    #[test]
    fn openai_definitions_advertise_toon_signatures_without_json_schema() {
        let mut registry = ToolRegistry::new();
        registry.register(BashTool);
        registry.register(ReadFileTool);
        let tools = registry.openai_tools();

        assert_eq!(tools.len(), 2);
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
}
