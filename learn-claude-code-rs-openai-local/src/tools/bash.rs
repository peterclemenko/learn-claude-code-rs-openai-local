use anyhow::Result;
use async_trait::async_trait;
use tokio::process::Command;
use tracing::instrument;

use crate::types::{parse_toon_field, toon_string};

use super::ToonTool;

pub struct BashTool;

#[async_trait]
impl ToonTool for BashTool {
    fn name(&self) -> &'static str {
        "bash"
    }

    fn description(&self) -> &'static str {
        "Run a shell command and return its exit code, stdout, and stderr."
    }

    fn parameters_signature(&self) -> &'static str {
        "bash[1]{command}:"
    }

    fn result_signature(&self) -> &'static str {
        "exec_result[1]{exit_code,stdout,stderr}:"
    }

    #[instrument(name = "tool.bash.execute", skip(self, payload))]
    async fn execute(&self, payload: &str) -> Result<String> {
        let command = parse_toon_field(payload, self.name(), "command")?;
        let output = Command::new("bash")
            .arg("-lc")
            .arg(command)
            .output()
            .await?;
        let exit_code = output.status.code().unwrap_or(-1).to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

        Ok(format!(
            "{}\n{exit_code},{},{}",
            self.result_signature(),
            toon_string(&stdout),
            toon_string(&stderr)
        ))
    }
}
