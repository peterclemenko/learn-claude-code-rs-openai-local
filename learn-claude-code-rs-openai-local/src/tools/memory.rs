use std::{collections::BTreeMap, str::FromStr, sync::Arc};

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde::Deserialize;
use tracing::instrument;

use crate::{
    memory::{
        MemoryBackend, MemoryCategory, MemoryEntry, MemoryQuery, MemoryRecallResult,
        format_recall_as_toon,
    },
    types::{parse_toon_table, toon_table},
};

const MEMORY_COLUMNS: &[&str] = &[
    "action", "category", "content", "tags", "scope", "metadata", "query", "id", "limit",
];

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct MemoryToolRequest {
    action: String,
    #[serde(default)]
    category: Option<MemoryCategory>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    metadata: BTreeMap<String, String>,
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
}

/// Permission- and hook-aware tool interface for durable memory operations.
pub struct MemoryTool {
    backend: Arc<dyn MemoryBackend>,
}

impl MemoryTool {
    /// Creates a memory tool backed by the selected shared provider.
    pub fn new(backend: Arc<dyn MemoryBackend>) -> Self {
        Self { backend }
    }

    fn parse_request(payload: &str) -> Result<MemoryToolRequest> {
        if payload.trim_start().starts_with('{') {
            return serde_json::from_str(payload).context("Invalid JSON memory tool arguments");
        }

        let rows = parse_toon_table(payload, "memory", MEMORY_COLUMNS)?;
        if rows.len() != 1 {
            bail!("Memory tool expects exactly one TOON row");
        }
        let row = rows.first().context("Memory tool requires one TOON row")?;
        let category = match row[1].trim() {
            "" => None,
            name => Some(MemoryCategory::from_str(name)?),
        };
        let tags = if row[3].trim().is_empty() {
            Vec::new()
        } else {
            row[3]
                .split(',')
                .map(str::trim)
                .filter(|tag| !tag.is_empty())
                .map(str::to_owned)
                .collect()
        };
        let metadata = if row[5].trim().is_empty() {
            BTreeMap::new()
        } else {
            serde_json::from_str(&row[5]).context("Memory metadata must be a JSON string map")?
        };
        let limit = if row[8].trim().is_empty() {
            None
        } else {
            Some(
                row[8]
                    .parse()
                    .context("Memory limit must be a non-negative integer")?,
            )
        };
        Ok(MemoryToolRequest {
            action: row[0].clone(),
            category,
            content: (!row[2].is_empty()).then(|| row[2].clone()),
            tags,
            scope: (!row[4].is_empty()).then(|| row[4].clone()),
            metadata,
            query: (!row[6].is_empty()).then(|| row[6].clone()),
            id: (!row[7].is_empty()).then(|| row[7].clone()),
            limit,
        })
    }
}

#[async_trait]
impl super::ToonTool for MemoryTool {
    fn name(&self) -> &'static str {
        "memory"
    }

    fn description(&self) -> &'static str {
        "Recall, remember, forget, list, or health-check durable memories. Send one TOON row or JSON with action=recall|remember|forget|list|health_check; the memory tool requires permission approval by default."
    }

    fn parameters_signature(&self) -> &'static str {
        "memory[1]{action,category,content,tags,scope,metadata,query,id,limit}:"
    }

    fn result_signature(&self) -> &'static str {
        "memories[N]{id,category,content,tags,scope,metadata,created_at}:"
    }

    #[instrument(name = "tool.memory", skip(self, payload), fields(memory.backend = %self.backend.backend_name()))]
    async fn execute(&self, payload: &str) -> Result<String> {
        let request = Self::parse_request(payload)?;
        match request.action.trim().to_ascii_lowercase().as_str() {
            "recall" => {
                let result = self
                    .backend
                    .recall(MemoryQuery {
                        text: request.query.unwrap_or_default(),
                        categories: request.category.into_iter().collect(),
                        scope: request.scope,
                        limit: request.limit,
                    })
                    .await?;
                Ok(self.backend.format_for_context(&result).await?)
            }
            "remember" => {
                let category = request
                    .category
                    .context("Memory remember action requires a category")?;
                let content = request
                    .content
                    .context("Memory remember action requires content")?;
                let id = self
                    .backend
                    .remember(MemoryEntry {
                        category,
                        content,
                        tags: request.tags,
                        scope: request.scope,
                        metadata: request.metadata,
                    })
                    .await?;
                Ok(toon_table("memory_saved", &["id"], &[id]))
            }
            "forget" => {
                let id = request.id.context("Memory forget action requires an id")?;
                let forgotten = self.backend.forget(&id).await?;
                Ok(toon_table(
                    "memory_forgotten",
                    &["id", "forgotten"],
                    &[id, forgotten.to_string()],
                ))
            }
            "list" => {
                let records = self.backend.list(request.limit.unwrap_or(100)).await?;
                Ok(format_recall_as_toon(&MemoryRecallResult {
                    query: MemoryQuery::default(),
                    records,
                }))
            }
            "health_check" => {
                self.backend.health_check().await?;
                Ok(toon_table(
                    "memory_health",
                    &["backend", "status"],
                    &[self.backend.backend_name().to_owned(), "healthy".to_owned()],
                ))
            }
            action => bail!("Unknown memory action '{action}'"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::memory::NoopMemoryBackend;

    use super::{super::ToonTool, MemoryTool};

    #[tokio::test]
    async fn memory_tool_accepts_toon_and_json_payloads() {
        let tool = MemoryTool::new(Arc::new(NoopMemoryBackend));
        let toon = tool
            .execute(
                "memory[1]{action,category,content,tags,scope,metadata,query,id,limit}:\nremember,Fact,\"A saved fact\",,,{},,,",
            )
            .await
            .unwrap();
        assert!(toon.contains("memory_saved[1]{id}"));
        let json = tool
            .execute(r#"{"action":"recall","query":"saved"}"#)
            .await
            .unwrap();
        assert_eq!(
            json,
            "memories[0]{id,category,content,tags,scope,metadata,created_at}:"
        );
    }
}
