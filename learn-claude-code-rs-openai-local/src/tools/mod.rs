// src/tools/mod.rs
use async_trait::async_trait;
use serde_json::Value;

#[async_trait]
pub trait Tool: Send + Sync {
    /// Unique name of the tool (e.g., "read_file", "write_file", "bash")
    fn name(&self) -> &'static str;

    /// Human-readable description for system prompt/model context
    fn description(&self) -> &'static str;

    /// JSON schema defining the tool parameters (compatible with OpenAI function calling)
    fn parameters(&self) -> Value;

    /// Execute the tool given raw JSON arguments
    async fn execute(
        &self,
        args: Value,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
}
