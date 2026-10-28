// src/tools/registry.rs
use crate::tools::Tool;
use async_openai::types::{ChatCompletionTool, ChatCompletionToolType, FunctionObject};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: impl Tool + 'static) {
        let arc = Arc::new(tool);
        self.tools.insert(arc.name().to_string(), arc);
    }

    /// Convert registered tools into OpenAI-compatible `ChatCompletionTool` structs
    pub fn to_openai_tools(&self) -> Vec<ChatCompletionTool> {
        self.tools
            .values()
            .map(|t| ChatCompletionTool {
                r#type: ChatCompletionToolType::Function,
                function: FunctionObject {
                    name: t.name().to_string(),
                    description: Some(t.description().to_string()),
                    parameters: Some(t.parameters()),
                    strict: Some(false),
                },
            })
            .collect()
    }

    pub async fn dispatch(&self, name: &str, arguments: &str) -> Result<String, String> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| format!("Tool not found: {}", name))?;
        let parsed_args: Value = serde_json::from_str(arguments)
            .map_err(|e| format!("Invalid JSON arguments for tool {}: {}", name, e))?;

        tool.execute(parsed_args)
            .await
            .map_err(|e| format!("Tool execution failed: {}", e))
    }
}
