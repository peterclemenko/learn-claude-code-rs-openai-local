use std::{
    sync::atomic::{AtomicU64, Ordering},
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{Instrument, error, info, info_span};

use crate::{
    agent::{Agent, AgentRuntime},
    tools::ToonTool,
    types::{AgentRonConfig, SubagentConfig, TodoState, parse_toon_table, toon_table},
};

const DELEGATE_COLUMNS: &[&str] = &["agent", "task", "context"];
const RESULT_COLUMNS: &[&str] = &["id", "agent", "task", "status", "result", "turns"];
static NEXT_SUBAGENT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DelegateTaskPayload {
    pub agent: String,
    pub task: String,
    #[serde(default)]
    pub context: String,
}

impl DelegateTaskPayload {
    pub fn from_payload(payload: &str) -> Result<Self> {
        let payload = payload.trim();
        let request = if payload.starts_with('{') {
            serde_json::from_str(payload).context("Invalid delegate_task JSON payload")?
        } else {
            let rows = parse_toon_table(payload, "delegate_task", DELEGATE_COLUMNS)?;
            let [row] = rows.as_slice() else {
                bail!("delegate_task requires exactly one row");
            };
            Self {
                agent: row[0].clone(),
                task: row[1].clone(),
                context: row[2].clone(),
            }
        };
        if request.agent.trim().is_empty() {
            bail!("delegate_task agent must not be empty");
        }
        if request.task.trim().is_empty() {
            bail!("delegate_task task must not be empty");
        }
        Ok(request)
    }

    pub fn render_toon(&self) -> String {
        toon_table(
            "delegate_task",
            DELEGATE_COLUMNS,
            &[self.agent.clone(), self.task.clone(), self.context.clone()],
        )
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub enum SubagentStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

impl std::fmt::Display for SubagentStatus {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Pending => "Pending",
            Self::InProgress => "InProgress",
            Self::Completed => "Completed",
            Self::Failed => "Failed",
        })
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct SubagentResult {
    pub id: String,
    pub agent: String,
    pub task: String,
    pub status: SubagentStatus,
    pub result: String,
    pub turns: usize,
}

impl SubagentResult {
    pub fn render_toon(&self) -> String {
        toon_table(
            "subagent_result",
            RESULT_COLUMNS,
            &[
                self.id.clone(),
                self.agent.clone(),
                self.task.clone(),
                self.status.to_string(),
                self.result.clone(),
                self.turns.to_string(),
            ],
        )
    }

    pub fn from_toon(payload: &str) -> Result<Self> {
        let rows = parse_toon_table(payload, "subagent_result", RESULT_COLUMNS)?;
        let [row] = rows.as_slice() else {
            bail!("subagent_result requires exactly one row");
        };
        Ok(Self {
            id: row[0].clone(),
            agent: row[1].clone(),
            task: row[2].clone(),
            status: match row[3].as_str() {
                "Pending" => SubagentStatus::Pending,
                "InProgress" => SubagentStatus::InProgress,
                "Completed" => SubagentStatus::Completed,
                "Failed" => SubagentStatus::Failed,
                status => bail!("Unknown subagent status '{status}'"),
            },
            result: row[4].clone(),
            turns: row[5].parse().context("Invalid subagent turn count")?,
        })
    }
}

pub(crate) struct DelegateTaskTool {
    parent_config: AgentRonConfig,
    runtime: AgentRuntime,
    parent_depth: usize,
    depth_limit: Option<usize>,
}

impl DelegateTaskTool {
    pub(crate) fn new(
        parent_config: AgentRonConfig,
        runtime: AgentRuntime,
        parent_depth: usize,
        depth_limit: Option<usize>,
    ) -> Self {
        Self {
            parent_config,
            runtime,
            parent_depth,
            depth_limit,
        }
    }

    async fn execute_delegation(&self, request: DelegateTaskPayload) -> Result<String> {
        let subagent = self
            .runtime
            .subagents
            .iter()
            .find(|subagent| subagent.name == request.agent)
            .with_context(|| format!("Unknown subagent '{}'", request.agent))?
            .clone();
        let child_depth = self.parent_depth + 1;
        let depth_limit = self
            .depth_limit
            .map_or(subagent.max_depth, |limit| limit.min(subagent.max_depth));
        if child_depth > depth_limit {
            bail!(
                "Delegation to '{}' exceeds max_depth {} at depth {}",
                subagent.name,
                depth_limit,
                child_depth
            );
        }

        let id = format!(
            "subagent-{}",
            NEXT_SUBAGENT_ID.fetch_add(1, Ordering::Relaxed)
        );
        let model = subagent
            .model
            .as_deref()
            .unwrap_or(&self.parent_config.model)
            .to_owned();
        let delegation_payload_bytes = request.render_toon().len();
        let span = info_span!(
            "subagent_execution",
            "subagent.id" = %id,
            "subagent.task" = %request.task,
            "subagent.model" = %model,
            subagent_name = %subagent.name,
            depth = child_depth,
            status = %SubagentStatus::Pending,
            delegation_payload_bytes,
            turns = tracing::field::Empty,
            result_bytes = tracing::field::Empty,
            succeeded = tracing::field::Empty,
        );

        async {
            info!(
                subagent_id = %id,
                subagent = %subagent.name,
                status = %SubagentStatus::Pending,
                "Subagent task created"
            );
            info!(
                subagent_id = %id,
                status = %SubagentStatus::InProgress,
                "Subagent task started"
            );

            let prompt = render_subagent_prompt(&subagent, &request);
            let mut child_config = self.parent_config.clone();
            child_config.name.clone_from(&subagent.name);
            child_config.model.clone_from(&model);
            child_config.system_prompt = prompt;
            let child = Agent::from_runtime(
                child_config,
                Arc::new(Mutex::new(TodoState::default())),
                self.runtime.clone(),
                child_depth,
                Some(depth_limit),
                subagent.allowed_tools.clone(),
            );

            match child.run_prompt(&request.task).await {
                Ok(result) => {
                    let result = SubagentResult {
                        id: id.clone(),
                        agent: subagent.name.clone(),
                        task: request.task.clone(),
                        status: SubagentStatus::Completed,
                        result,
                        turns: child.turns_completed(),
                    };
                    let output = result.render_toon();
                    let validated_result = SubagentResult::from_toon(&output)?;
                    let span = tracing::Span::current();
                    span.record("status", SubagentStatus::Completed.to_string());
                    span.record("turns", result.turns);
                    span.record("result_bytes", output.len());
                    span.record("succeeded", true);
                    info!(
                        subagent_id = %id,
                        status = %SubagentStatus::Completed,
                        turns = result.turns,
                        result_bytes = output.len(),
                        "Subagent task completed"
                    );
                    Ok(validated_result.render_toon())
                }
                Err(run_error) => {
                    let span = tracing::Span::current();
                    span.record("status", SubagentStatus::Failed.to_string());
                    span.record("succeeded", false);
                    error!(
                        subagent_id = %id,
                        status = %SubagentStatus::Failed,
                        error = %run_error,
                        "Subagent task failed"
                    );
                    Err(run_error).context(format!("Subagent '{}' failed", subagent.name))
                }
            }
        }
        .instrument(span)
        .await
    }
}

fn render_subagent_prompt(subagent: &SubagentConfig, request: &DelegateTaskPayload) -> String {
    subagent
        .system_prompt_template
        .replace("{name}", &subagent.name)
        .replace("{description}", &subagent.description)
        .replace("{task}", &request.task)
        .replace("{context}", &request.context)
}

#[async_trait]
impl ToonTool for DelegateTaskTool {
    fn name(&self) -> &'static str {
        "delegate_task"
    }

    fn description(&self) -> &'static str {
        "Delegate an isolated task to a configured local subagent."
    }

    fn parameters_signature(&self) -> &'static str {
        "delegate_task[1]{agent,task,context}:"
    }

    fn result_signature(&self) -> &'static str {
        "subagent_result[1]{id,agent,task,status,result,turns}:"
    }

    async fn execute(&self, payload: &str) -> Result<String> {
        self.execute_delegation(DelegateTaskPayload::from_payload(payload)?)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::{DelegateTaskPayload, SubagentResult, SubagentStatus};

    #[test]
    fn delegate_payload_round_trips_as_toon() {
        let payload = DelegateTaskPayload {
            agent: "code_runner".to_owned(),
            task: "run tests".to_owned(),
            context: "The workspace is under /repo and may contain,commas.".to_owned(),
        };
        let serialized = payload.render_toon();
        assert_eq!(
            serialized,
            "delegate_task[1]{agent,task,context}:\n\"code_runner\",\"run tests\",\"The workspace is under /repo and may contain,commas.\""
        );
        assert_eq!(
            DelegateTaskPayload::from_payload(&serialized).unwrap(),
            payload
        );
    }

    #[test]
    fn rejects_invalid_or_incomplete_delegation_payloads() {
        assert!(
            DelegateTaskPayload::from_payload(
                "delegate_task[1]{agent,task,context}:\ncode_runner,run tests"
            )
            .is_err()
        );
        assert!(DelegateTaskPayload::from_payload(r#"{"agent":" ","task":"run"}"#).is_err());
        assert!(
            DelegateTaskPayload::from_payload(
                "delegate_task[2]{agent,task,context}:\ncode_runner,first,\nreviewer,second,"
            )
            .is_err()
        );
    }

    #[test]
    fn subagent_result_round_trips_with_toon_escaping() {
        let result = SubagentResult {
            id: "subagent-3".to_owned(),
            agent: "code_runner".to_owned(),
            task: "run \"fast\" tests".to_owned(),
            status: SubagentStatus::Completed,
            result: "Finished.\nAll tests passed.".to_owned(),
            turns: 2,
        };
        assert_eq!(
            SubagentResult::from_toon(&result.render_toon()).unwrap(),
            result
        );
        assert!(
            SubagentResult::from_toon(
                "subagent_result[2]{id,agent,task,status,result,turns}:\nsubagent-1,code_runner,first,Completed,ok,1\nsubagent-2,reviewer,second,Completed,ok,1"
            )
            .is_err()
        );
    }
}
