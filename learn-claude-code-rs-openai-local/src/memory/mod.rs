mod naive_memory_system;

pub mod config;
pub mod registry;

use std::{collections::BTreeMap, str::FromStr, sync::Arc, time::Instant};

use anyhow::{Result, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{error, info, instrument};

pub use naive_memory_system::{NaiveMemorySystem, NoopMemoryBackend};
pub use registry::{MemoryRegistry, build_memory_from_config, register_memory_backend};

pub type MemoryResult<T> = Result<T>;

/// Durable categories supported by the built-in memory model.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryCategory {
    Preference,
    Fact,
    Feedback,
    Reference,
}

impl MemoryCategory {
    /// Returns the canonical spelling used in RON config and TOON output.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Preference => "Preference",
            Self::Fact => "Fact",
            Self::Feedback => "Feedback",
            Self::Reference => "Reference",
        }
    }
}

impl FromStr for MemoryCategory {
    type Err = anyhow::Error;

    fn from_str(category: &str) -> Result<Self> {
        match category.trim().to_ascii_lowercase().as_str() {
            "preference" => Ok(Self::Preference),
            "fact" => Ok(Self::Fact),
            "feedback" => Ok(Self::Feedback),
            "reference" => Ok(Self::Reference),
            _ => bail!("Unknown memory category '{category}'"),
        }
    }
}

/// User-authored memory content and its filtering/indexing metadata.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct MemoryEntry {
    pub category: MemoryCategory,
    pub content: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

/// A durable memory entry with its stable identifier and timestamps.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct MemoryRecord {
    pub id: String,
    pub entry: MemoryEntry,
    pub created_at: String,
    pub updated_at: String,
}

/// Search terms and optional filters for memory recall.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct MemoryQuery {
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub categories: Vec<MemoryCategory>,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Ranked memory records returned for a query.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct MemoryRecallResult {
    pub query: MemoryQuery,
    pub records: Vec<MemoryRecord>,
}

/// Pluggable async contract between the agent and a durable memory provider.
#[async_trait]
pub trait MemoryBackend: Send + Sync {
    /// Identifies this implementation for diagnostics and telemetry.
    fn backend_name(&self) -> &str;

    /// Returns records relevant to the query, ordered by backend relevance.
    async fn recall(&self, query: MemoryQuery) -> MemoryResult<MemoryRecallResult>;

    /// Stores an entry and returns its stable backend identifier.
    async fn remember(&self, entry: MemoryEntry) -> MemoryResult<String>;

    /// Removes the record with this identifier, returning whether it existed.
    async fn forget(&self, id: &str) -> MemoryResult<bool>;

    /// Lists the most recent records up to the supplied limit.
    async fn list(&self, limit: usize) -> MemoryResult<Vec<MemoryRecord>>;

    /// Formats recalled records for prompt context using TOON by default.
    async fn format_for_context(&self, recall: &MemoryRecallResult) -> MemoryResult<String> {
        Ok(format_recall_as_toon(recall))
    }

    /// Verifies that the backend can currently serve requests.
    async fn health_check(&self) -> MemoryResult<()>;
}

/// Renders recall results as a compact TOON table for prompts and tools.
pub fn format_recall_as_toon(recall: &MemoryRecallResult) -> String {
    let columns = "id,category,content,tags,scope,metadata,created_at";
    let header = format!("memories[{}]{{{columns}}}:", recall.records.len());
    if recall.records.is_empty() {
        return header;
    }

    let rows = recall
        .records
        .iter()
        .map(|record| {
            let metadata = record
                .entry
                .metadata
                .iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>()
                .join(";");
            [
                record.id.clone(),
                record.entry.category.as_str().to_owned(),
                record.entry.content.clone(),
                record.entry.tags.join(","),
                record.entry.scope.clone().unwrap_or_default(),
                metadata,
                record.created_at.clone(),
            ]
            .iter()
            .map(|value| crate::types::toon_string(value))
            .collect::<Vec<_>>()
            .join(",")
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!("{header}\n{rows}")
}

pub(crate) fn observed_backend(inner: Arc<dyn MemoryBackend>) -> Arc<dyn MemoryBackend> {
    Arc::new(ObservedMemoryBackend { inner })
}

struct ObservedMemoryBackend {
    inner: Arc<dyn MemoryBackend>,
}

fn record_operation(started_at: Instant, entry_count: usize, failure: Option<&anyhow::Error>) {
    let duration_ms = started_at
        .elapsed()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX);
    let outcome = if failure.is_some() { "error" } else { "ok" };
    let span = tracing::Span::current();
    span.record("memory.entry_count", entry_count);
    span.record("memory.duration_ms", duration_ms);
    span.record("memory.outcome", outcome);
    if let Some(failure) = failure {
        error!(
            memory.entry_count = entry_count,
            memory.duration_ms = duration_ms,
            memory.outcome = outcome,
            error = %failure,
            "Memory backend operation failed"
        );
    } else {
        info!(
            memory.entry_count = entry_count,
            memory.duration_ms = duration_ms,
            memory.outcome = outcome,
            "Memory backend operation completed"
        );
    }
}

#[async_trait]
impl MemoryBackend for ObservedMemoryBackend {
    fn backend_name(&self) -> &str {
        self.inner.backend_name()
    }

    #[instrument(
        name = "memory.recall",
        skip(self, query),
        fields(
            memory.backend = %self.backend_name(),
            memory.operation = "recall",
            memory.entry_count = tracing::field::Empty,
            memory.duration_ms = tracing::field::Empty,
            memory.outcome = tracing::field::Empty
        )
    )]
    async fn recall(&self, query: MemoryQuery) -> MemoryResult<MemoryRecallResult> {
        let started_at = Instant::now();
        let result = self.inner.recall(query).await;
        match &result {
            Ok(recall) => record_operation(started_at, recall.records.len(), None),
            Err(failure) => record_operation(started_at, 0, Some(failure)),
        }
        result
    }

    #[instrument(
        name = "memory.remember",
        skip(self, entry),
        fields(
            memory.backend = %self.backend_name(),
            memory.operation = "remember",
            memory.entry_count = tracing::field::Empty,
            memory.duration_ms = tracing::field::Empty,
            memory.outcome = tracing::field::Empty
        )
    )]
    async fn remember(&self, entry: MemoryEntry) -> MemoryResult<String> {
        let started_at = Instant::now();
        let result = self.inner.remember(entry).await;
        match &result {
            Ok(_) => record_operation(started_at, 1, None),
            Err(failure) => record_operation(started_at, 0, Some(failure)),
        }
        result
    }

    #[instrument(
        name = "memory.forget",
        skip(self, id),
        fields(
            memory.backend = %self.backend_name(),
            memory.operation = "forget",
            memory.entry_count = tracing::field::Empty,
            memory.duration_ms = tracing::field::Empty,
            memory.outcome = tracing::field::Empty
        )
    )]
    async fn forget(&self, id: &str) -> MemoryResult<bool> {
        let started_at = Instant::now();
        let result = self.inner.forget(id).await;
        match &result {
            Ok(forgotten) => record_operation(started_at, usize::from(*forgotten), None),
            Err(failure) => record_operation(started_at, 0, Some(failure)),
        }
        result
    }

    #[instrument(
        name = "memory.list",
        skip(self),
        fields(
            memory.backend = %self.backend_name(),
            memory.operation = "list",
            memory.entry_count = tracing::field::Empty,
            memory.duration_ms = tracing::field::Empty,
            memory.outcome = tracing::field::Empty
        )
    )]
    async fn list(&self, limit: usize) -> MemoryResult<Vec<MemoryRecord>> {
        let started_at = Instant::now();
        let result = self.inner.list(limit).await;
        match &result {
            Ok(records) => record_operation(started_at, records.len(), None),
            Err(failure) => record_operation(started_at, 0, Some(failure)),
        }
        result
    }

    #[instrument(
        name = "memory.format_context",
        skip(self, recall),
        fields(
            memory.backend = %self.backend_name(),
            memory.operation = "format_for_context",
            memory.entry_count = tracing::field::Empty,
            memory.duration_ms = tracing::field::Empty,
            memory.outcome = tracing::field::Empty
        )
    )]
    async fn format_for_context(&self, recall: &MemoryRecallResult) -> MemoryResult<String> {
        let started_at = Instant::now();
        let result = self.inner.format_for_context(recall).await;
        match &result {
            Ok(_) => record_operation(started_at, recall.records.len(), None),
            Err(failure) => record_operation(started_at, recall.records.len(), Some(failure)),
        }
        result
    }

    #[instrument(
        name = "memory.health_check",
        skip(self),
        fields(
            memory.backend = %self.backend_name(),
            memory.operation = "health_check",
            memory.entry_count = tracing::field::Empty,
            memory.duration_ms = tracing::field::Empty,
            memory.outcome = tracing::field::Empty
        )
    )]
    async fn health_check(&self) -> MemoryResult<()> {
        let started_at = Instant::now();
        let result = self.inner.health_check().await;
        match &result {
            Ok(()) => record_operation(started_at, 0, None),
            Err(failure) => record_operation(started_at, 0, Some(failure)),
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MemoryCategory, MemoryEntry, MemoryQuery, MemoryRecallResult, format_recall_as_toon,
    };

    #[test]
    fn recall_context_is_formatted_as_toon() {
        let recall = MemoryRecallResult {
            query: MemoryQuery::default(),
            records: vec![super::MemoryRecord {
                id: "memory-1".to_owned(),
                entry: MemoryEntry {
                    category: MemoryCategory::Preference,
                    content: "Use Rust".to_owned(),
                    tags: vec!["language".to_owned()],
                    scope: None,
                    metadata: Default::default(),
                },
                created_at: "2025-01-01T00:00:00Z".to_owned(),
                updated_at: "2025-01-01T00:00:00Z".to_owned(),
            }],
        };
        assert_eq!(
            format_recall_as_toon(&recall),
            "memories[1]{id,category,content,tags,scope,metadata,created_at}:\n\"memory-1\",\"Preference\",\"Use Rust\",\"language\",\"\",\"\",\"2025-01-01T00:00:00Z\""
        );
    }

    #[cfg(test)]
    mod otel_memory_tests {
        use opentelemetry::{Value, trace::TracerProvider as _};
        use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider, SpanData};
        use tracing_subscriber::prelude::*;

        use crate::memory::{
            MemoryCategory, MemoryEntry, MemoryQuery, MemoryRegistry, config::MemorySystemConfig,
        };

        fn attribute<'a>(span: &'a SpanData, name: &str) -> Option<&'a Value> {
            span.attributes
                .iter()
                .find(|attribute| attribute.key.as_str() == name)
                .map(|attribute| &attribute.value)
        }

        #[tokio::test]
        async fn otel_memory_operations_export_structured_spans_and_shutdown_cleanly() {
            let exporter = InMemorySpanExporter::default();
            let provider = SdkTracerProvider::builder()
                .with_simple_exporter(exporter.clone())
                .build();
            let subscriber = tracing_subscriber::registry().with(
                tracing_opentelemetry::layer().with_tracer(provider.tracer("memory-otel-test")),
            );
            let _guard = tracing::subscriber::set_default(subscriber);

            let mut config = MemorySystemConfig::default();
            config.backend = "noop".to_owned();
            let backend = MemoryRegistry::new().build(&config).unwrap();
            backend.health_check().await.unwrap();
            backend
                .remember(MemoryEntry {
                    category: MemoryCategory::Fact,
                    content: "OTel test memory".to_owned(),
                    tags: Vec::new(),
                    scope: None,
                    metadata: Default::default(),
                })
                .await
                .unwrap();
            let recall = backend
                .recall(MemoryQuery {
                    text: "OTel".to_owned(),
                    ..Default::default()
                })
                .await
                .unwrap();
            backend.format_for_context(&recall).await.unwrap();
            backend.list(10).await.unwrap();
            backend.forget("noop").await.unwrap();

            provider.force_flush().unwrap();
            let spans = exporter.get_finished_spans().unwrap();
            for operation in [
                "health_check",
                "remember",
                "recall",
                "format_for_context",
                "list",
                "forget",
            ] {
                let span = spans
                    .iter()
                    .find(|span| {
                        matches!(
                            attribute(span, "memory.operation"),
                            Some(Value::String(value)) if value.as_str() == operation
                        )
                    })
                    .unwrap_or_else(|| {
                        panic!("missing OTel span for memory operation {operation}")
                    });
                assert!(attribute(span, "memory.duration_ms").is_some());
                assert!(attribute(span, "memory.entry_count").is_some());
                assert!(matches!(
                    attribute(span, "memory.outcome"),
                    Some(Value::String(value)) if value.as_str() == "ok"
                ));
                assert!(matches!(
                    attribute(span, "memory.backend"),
                    Some(Value::String(value)) if value.as_str() == "noop"
                ));
            }
            assert!(spans.iter().any(|span| {
                span.name.as_ref() == "memory.backend.build"
                    && matches!(
                        attribute(span, "memory.outcome"),
                        Some(Value::String(value)) if value.as_str() == "ok"
                    )
            }));

            drop(_guard);
            provider.shutdown().unwrap();
            assert!(exporter.is_shutdown_called());
        }
    }
}
