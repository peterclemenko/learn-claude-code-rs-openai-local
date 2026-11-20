use std::{
    cmp::Ordering,
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering as AtomicOrdering},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use chrono::{SecondsFormat, Utc};
use tokio::{fs, sync::Mutex};
use tracing::instrument;

use super::{
    MemoryBackend, MemoryCategory, MemoryEntry, MemoryQuery, MemoryRecallResult, MemoryRecord,
    MemoryResult, config::MemorySystemConfig,
};

static NEXT_FILE_ID: AtomicU64 = AtomicU64::new(1);

/// RON-backed implementation of the naive durable memory system.
pub struct NaiveMemorySystem {
    config: MemorySystemConfig,
    path: PathBuf,
    records: Mutex<Vec<MemoryRecord>>,
}

impl NaiveMemorySystem {
    /// Opens or initializes the configured on-disk store.
    pub fn open(config: &MemorySystemConfig) -> Result<Self> {
        config.validate()?;
        let file_name = Path::new(config.memory_file_name());
        let mut components = file_name.components();
        if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
            bail!("Naive memory file_name must be a plain relative file name");
        }

        std::fs::create_dir_all(&config.storage_dir).with_context(|| {
            format!(
                "Failed to create memory storage directory {}",
                config.storage_dir.display()
            )
        })?;
        let path = config.storage_dir.join(file_name);
        let records = match std::fs::read_to_string(&path) {
            Ok(contents) => ron::from_str(&contents)
                .with_context(|| format!("Failed to parse memory store {}", path.display()))?,
            Err(failure) if failure.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(failure) => {
                return Err(failure)
                    .with_context(|| format!("Failed to read memory store {}", path.display()));
            }
        };

        Ok(Self {
            config: config.clone(),
            path,
            records: Mutex::new(records),
        })
    }

    async fn persist(&self, records: &[MemoryRecord]) -> Result<()> {
        let contents = ron::ser::to_string(records).context("Failed to serialize memory store")?;
        let temp_path = self.path.with_extension(format!(
            "ron.tmp-{}-{}",
            std::process::id(),
            NEXT_FILE_ID.fetch_add(1, AtomicOrdering::Relaxed)
        ));
        fs::write(&temp_path, contents).await.with_context(|| {
            format!(
                "Failed to write temporary memory store {}",
                temp_path.display()
            )
        })?;
        if let Err(failure) = fs::rename(&temp_path, &self.path).await {
            if let Err(cleanup_error) = fs::remove_file(&temp_path).await {
                tracing::warn!(
                    temporary_path = %temp_path.display(),
                    error = %cleanup_error,
                    "Failed to remove temporary memory store after persist failure"
                );
            }
            return Err(failure).with_context(|| {
                format!("Failed to replace memory store {}", self.path.display())
            });
        }
        Ok(())
    }

    fn category_enabled(&self, category: MemoryCategory) -> bool {
        self.config.enabled_categories.contains(&category)
    }

    fn category_weight(&self, category: MemoryCategory) -> f64 {
        self.config
            .category_weights
            .get(category.as_str())
            .copied()
            .unwrap_or(1.0)
    }
}

#[async_trait]
impl MemoryBackend for NaiveMemorySystem {
    fn backend_name(&self) -> &str {
        "naivememorysystem"
    }

    #[instrument(name = "memory.naive.recall", skip(self, query), fields(memory.query_bytes = query.text.len()))]
    async fn recall(&self, query: MemoryQuery) -> MemoryResult<MemoryRecallResult> {
        let records = self.records.lock().await;
        let mut ranked = records
            .iter()
            .filter(|record| self.category_enabled(record.entry.category))
            .filter(|record| {
                query.categories.is_empty() || query.categories.contains(&record.entry.category)
            })
            .filter(|record| {
                query
                    .scope
                    .as_ref()
                    .is_none_or(|scope| record.entry.scope.as_ref() == Some(scope))
            })
            .filter_map(|record| {
                let score = relevance_score(record, &query.text)
                    * self.category_weight(record.entry.category);
                (query.text.trim().is_empty() || score > 0.0).then(|| (score, record.clone()))
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|(left_score, left), (right_score, right)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| right.updated_at.cmp(&left.updated_at))
        });
        let limit = query.limit.unwrap_or(self.config.recall_limit);
        Ok(MemoryRecallResult {
            query,
            records: ranked
                .into_iter()
                .take(limit)
                .map(|(_, record)| record)
                .collect(),
        })
    }

    #[instrument(name = "memory.naive.remember", skip(self, entry), fields(memory.category = %entry.category.as_str()))]
    async fn remember(&self, entry: MemoryEntry) -> MemoryResult<String> {
        if entry.content.trim().is_empty() {
            bail!("Memory content must not be empty");
        }
        if !self.category_enabled(entry.category) {
            bail!(
                "Memory category '{}' is disabled by configuration",
                entry.category.as_str()
            );
        }
        let now = Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true);
        let record = MemoryRecord {
            id: new_memory_id(),
            entry,
            created_at: now.clone(),
            updated_at: now,
        };
        let mut current = self.records.lock().await;
        let mut updated = current.clone();
        updated.push(record.clone());
        self.persist(&updated).await?;
        *current = updated;
        Ok(record.id)
    }

    #[instrument(name = "memory.naive.forget", skip(self, id))]
    async fn forget(&self, id: &str) -> MemoryResult<bool> {
        let mut current = self.records.lock().await;
        if !current.iter().any(|record| record.id == id) {
            return Ok(false);
        }
        let updated = current
            .iter()
            .filter(|record| record.id != id)
            .cloned()
            .collect::<Vec<_>>();
        self.persist(&updated).await?;
        *current = updated;
        Ok(true)
    }

    #[instrument(name = "memory.naive.list", skip(self))]
    async fn list(&self, limit: usize) -> MemoryResult<Vec<MemoryRecord>> {
        let records = self.records.lock().await;
        let mut records = records.clone();
        records.sort_by(|left, right| match right.updated_at.cmp(&left.updated_at) {
            Ordering::Equal => right.id.cmp(&left.id),
            ordering => ordering,
        });
        records.truncate(limit);
        Ok(records)
    }

    #[instrument(name = "memory.naive.health_check", skip(self))]
    async fn health_check(&self) -> MemoryResult<()> {
        fs::metadata(&self.config.storage_dir)
            .await
            .with_context(|| {
                format!(
                    "Memory storage directory {} is unavailable",
                    self.config.storage_dir.display()
                )
            })?;
        Ok(())
    }
}

/// No-storage backend for runs where durable memory is disabled.
#[derive(Default)]
pub struct NoopMemoryBackend;

#[async_trait]
impl MemoryBackend for NoopMemoryBackend {
    fn backend_name(&self) -> &str {
        "noop"
    }

    async fn recall(&self, query: MemoryQuery) -> MemoryResult<MemoryRecallResult> {
        Ok(MemoryRecallResult {
            query,
            records: Vec::new(),
        })
    }

    async fn remember(&self, _entry: MemoryEntry) -> MemoryResult<String> {
        Ok("noop".to_owned())
    }

    async fn forget(&self, _id: &str) -> MemoryResult<bool> {
        Ok(false)
    }

    async fn list(&self, _limit: usize) -> MemoryResult<Vec<MemoryRecord>> {
        Ok(Vec::new())
    }

    async fn health_check(&self) -> MemoryResult<()> {
        Ok(())
    }
}

fn relevance_score(record: &MemoryRecord, query: &str) -> f64 {
    let tokens = query
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
        .collect::<Vec<_>>();
    if tokens.is_empty() {
        return 0.0;
    }
    let searchable = format!(
        "{} {} {} {}",
        record.entry.content,
        record.entry.tags.join(" "),
        record.entry.scope.as_deref().unwrap_or_default(),
        record
            .entry
            .metadata
            .iter()
            .map(|(key, value)| format!("{key} {value}"))
            .collect::<Vec<_>>()
            .join(" ")
    )
    .to_lowercase();
    tokens
        .iter()
        .map(|token| searchable.matches(token).count() as f64)
        .sum()
}

fn new_memory_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!(
        "mem-{}-{nanos}-{}",
        std::process::id(),
        NEXT_FILE_ID.fetch_add(1, AtomicOrdering::Relaxed)
    )
}

#[cfg(test)]
mod tests {
    use std::{fs, time::SystemTime};

    use super::{super::MemoryBackend, NaiveMemorySystem};
    use crate::memory::{MemoryCategory, MemoryEntry, MemoryQuery, config::MemorySystemConfig};

    fn config() -> MemorySystemConfig {
        let mut config = MemorySystemConfig::default();
        config.storage_dir = std::env::temp_dir().join(format!(
            "naive-memory-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        config
    }

    fn entry(content: &str) -> MemoryEntry {
        MemoryEntry {
            category: MemoryCategory::Fact,
            content: content.to_owned(),
            tags: vec!["rust".to_owned()],
            scope: Some("project".to_owned()),
            metadata: Default::default(),
        }
    }

    #[tokio::test]
    async fn naive_memory_remembers_recalls_lists_forgets_and_reopens() {
        let config = config();
        let backend = NaiveMemorySystem::open(&config).unwrap();
        let id = backend
            .remember(entry("This project uses pure Rust storage"))
            .await
            .unwrap();
        let recalled = backend
            .recall(MemoryQuery {
                text: "pure Rust".to_owned(),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(recalled.records.len(), 1);
        assert_eq!(recalled.records[0].id, id);
        assert_eq!(backend.list(10).await.unwrap().len(), 1);

        let reopened = NaiveMemorySystem::open(&config).unwrap();
        assert_eq!(reopened.list(10).await.unwrap().len(), 1);
        assert!(reopened.forget(&id).await.unwrap());
        assert!(reopened.list(10).await.unwrap().is_empty());
        fs::remove_dir_all(config.storage_dir).unwrap();
    }

    #[tokio::test]
    async fn naive_memory_respects_categories_and_scope_filters() {
        let config = config();
        let backend = NaiveMemorySystem::open(&config).unwrap();
        backend.remember(entry("Use Rust")).await.unwrap();
        let recalled = backend
            .recall(MemoryQuery {
                text: "Rust".to_owned(),
                categories: vec![MemoryCategory::Preference],
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(recalled.records.is_empty());
        fs::remove_dir_all(config.storage_dir).unwrap();
    }

    #[test]
    fn naive_memory_rejects_path_traversal_in_file_name() {
        let mut config = config();
        config
            .backend_options
            .insert("file_name".to_owned(), "../outside.ron".to_owned());
        assert!(NaiveMemorySystem::open(&config).is_err());
    }
}
