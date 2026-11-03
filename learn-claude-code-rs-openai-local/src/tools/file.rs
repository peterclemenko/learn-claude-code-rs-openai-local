use anyhow::Result;
use async_trait::async_trait;
use tokio::fs;
use tracing::instrument;

use crate::types::{parse_toon_field, toon_table};

use super::ToonTool;

pub struct ReadFileTool;

#[async_trait]
impl ToonTool for ReadFileTool {
    fn name(&self) -> &'static str {
        "read_file"
    }

    fn description(&self) -> &'static str {
        "Read a UTF-8 file and return its path and contents."
    }

    fn parameters_signature(&self) -> &'static str {
        "read_file[1]{path}:"
    }

    fn result_signature(&self) -> &'static str {
        "read_result[1]{path,content}:"
    }

    #[instrument(name = "tool.file.read", skip(self, payload))]
    async fn execute(&self, payload: &str) -> Result<String> {
        let path = parse_toon_field(payload, self.name(), "path")?;
        let content = fs::read_to_string(&path).await?;
        Ok(toon_table(
            "read_result",
            &["path", "content"],
            &[path, content],
        ))
    }
}
