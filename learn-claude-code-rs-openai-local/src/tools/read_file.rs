// src/tools/read_file.rs
use crate::tools::Tool;
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::fs;

pub struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &'static str {
        "read_file"
    }

    fn description(&self) -> &'static str {
        "Read the contents of a file at the specified path."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The relative or absolute path to the file to read."
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(
        &self,
        args: Value,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let path = args["path"].as_str().ok_or("Missing 'path' argument")?;
        let content = fs::read_to_string(path).await?;
        Ok(content)
    }
}
