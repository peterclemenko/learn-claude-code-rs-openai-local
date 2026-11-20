use async_trait::async_trait;
use serde::Deserialize;
use tracing::instrument;

use super::ToonTool;
use crate::prompt::{PromptChangeSource, PromptManager};
use std::sync::Arc;

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct SelectSystemPromptRequest {
    #[serde(default)]
    action: String,
    #[serde(default)]
    name: Option<String>,
}

pub struct SelectSystemPromptTool {
    manager: Arc<PromptManager>,
}

impl SelectSystemPromptTool {
    pub fn new(manager: Arc<PromptManager>) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl ToonTool for SelectSystemPromptTool {
    fn name(&self) -> &'static str {
        "select_system_prompt"
    }

    fn description(&self) -> &'static str {
        "List available RON system prompt templates or switch the active prompt. JSON: {action:\"list|switch\",name:\"default\"}. TOON signature is select_system_prompt[1]{action,name}:"
    }

    fn parameters_signature(&self) -> &'static str {
        "select_system_prompt[1]{action,name}:"
    }

    fn result_signature(&self) -> &'static str {
        "prompt_switch{name,path,outcome}: or prompts[N]{name}:"
    }

    #[instrument(name = "tool.prompt.select", skip(self, payload))]
    async fn execute(&self, payload: &str) -> anyhow::Result<String> {
        let request: SelectSystemPromptRequest = if payload.trim_start().starts_with('{') {
            serde_json::from_str(payload)?
        } else {
            let mut fields = payload.lines().find_map(|line| {
                if line.contains(',') {
                    Some(line.split(',').map(str::trim).collect::<Vec<_>>())
                } else {
                    None
                }
            });
            match fields.as_mut().and_then(|fields| {
                (fields.len() >= 1).then(|| SelectSystemPromptRequest {
                    action: fields.first().unwrap_or(&"").to_string(),
                    name: fields
                        .get(1)
                        .filter(|value| !value.is_empty())
                        .map(|value| (*value).to_string()),
                })
            }) {
                Some(request) => request,
                None => SelectSystemPromptRequest::default(),
            }
        };

        match request.action.trim().to_ascii_lowercase().as_str() {
            "list" => {
                let names = self.manager.list_available()?;
                let mut output = format!("prompts[{}]{{name}}:", names.len());
                for name in names {
                    output.push('\n');
                    output.push_str(&name);
                }
                Ok(output)
            }
            "switch" => {
                let name = request
                    .name
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("Switch action requires prompt name"))?;
                let outcome = self
                    .manager
                    .select_prompt(name, PromptChangeSource::AgentTool)
                    .await?;
                Ok(format!(
                    "prompt_switch[1]{{name,path,outcome}}:\n{}, {}, switched",
                    outcome.name,
                    outcome.path.display()
                ))
            }
            _ => anyhow::bail!("Prompt tool action must be 'list' or 'switch'"),
        }
    }
}
