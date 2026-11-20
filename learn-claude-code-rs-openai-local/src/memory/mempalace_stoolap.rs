use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail, ensure};
use async_trait::async_trait;
use chrono::{SecondsFormat, Utc};
use stoolap::{ApiTransaction, Database, ResultRow};
use tracing::instrument;

use super::{
    MemoryBackend, MemoryCategory, MemoryEntry, MemoryQuery, MemoryRecallResult, MemoryRecord,
    MemoryResult,
    config::{MemPalaceStoolapConfig, MemorySystemConfig},
    format_recall_as_toon,
};

const DRAWER_COLUMNS: &str =
    "id, category, content, tags_json, scope, metadata_json, created_at, updated_at";
const MAX_CLOSET_HALLS: usize = 8;
const MAX_CLOSET_TAGS: usize = 24;
static NEXT_DRAWER_ID: AtomicU64 = AtomicU64::new(1);

/// Produces optional vectors for semantic indexing and recall.
///
/// Implementations may return `None` when a model is unavailable for a request;
/// the backend then uses lexical, metadata, scope, category, and recency ranking.
#[async_trait]
pub trait MemoryEmbeddingProvider: Send + Sync {
    /// Returns the dimension of vectors produced by this provider.
    fn dimensions(&self) -> usize;

    /// Embeds text, returning `None` when semantic search is unavailable.
    async fn embed(&self, text: &str) -> MemoryResult<Option<Vec<f32>>>;
}

/// Embedded Stoolap memory organized as wings, rooms, halls, closets, and drawers.
///
/// Drawer content and entry metadata are stored verbatim in relational tables.
/// Stoolap's native `VECTOR(N)` column and HNSW index are available when an
/// embedding provider is supplied; without one, recall safely uses lexical,
/// metadata, scope, category, and recency signals.
pub struct MemPalaceStoolapMemory {
    config: MemorySystemConfig,
    database: Mutex<Database>,
    embedding_provider: Option<Arc<dyn MemoryEmbeddingProvider>>,
}

impl MemPalaceStoolapMemory {
    /// Opens a persistent or isolated in-memory MemPalace Stoolap database.
    #[instrument(
        name = "memory.mempalace.open",
        skip(config),
        fields(
            memory.backend = "mempalace_stoolap",
            memory.storage_mode = %config.mempalace_stoolap.storage_mode,
            memory.vector_dimensions = config.mempalace_stoolap.vector_dimensions
        )
    )]
    pub fn open(config: &MemorySystemConfig) -> Result<Self> {
        Self::open_with_embedding_provider(config, None)
    }

    /// Opens a database with an optional pluggable vector embedding provider.
    ///
    /// The provider's dimension must match the configured native Stoolap vector
    /// column. It is not called for empty recall queries.
    #[instrument(
        name = "memory.mempalace.open_with_embedding_provider",
        skip(config, embedding_provider),
        fields(
            memory.backend = "mempalace_stoolap",
            memory.storage_mode = %config.mempalace_stoolap.storage_mode,
            memory.vector_dimensions = config.mempalace_stoolap.vector_dimensions,
            memory.embedding_provider = embedding_provider.is_some()
        )
    )]
    pub fn open_with_embedding_provider(
        config: &MemorySystemConfig,
        embedding_provider: Option<Arc<dyn MemoryEmbeddingProvider>>,
    ) -> Result<Self> {
        config.validate()?;
        ensure!(
            config.backend == "mempalace_stoolap",
            "MemPalace Stoolap requires backend = 'mempalace_stoolap'"
        );
        if let Some(provider) = &embedding_provider {
            ensure!(
                provider.dimensions() == config.mempalace_stoolap.vector_dimensions,
                "Embedding provider dimension does not match MemPalace Stoolap vector_dimensions"
            );
        }

        let database = open_database(config)?;
        initialize_schema(&database, &config.mempalace_stoolap)?;
        Ok(Self {
            config: config.clone(),
            database: Mutex::new(database),
            embedding_provider,
        })
    }

    fn database(&self) -> Result<MutexGuard<'_, Database>> {
        self.database.lock().map_err(|failure| {
            anyhow::anyhow!("MemPalace Stoolap database lock is poisoned: {failure}")
        })
    }

    fn settings(&self) -> &MemPalaceStoolapConfig {
        &self.config.mempalace_stoolap
    }

    fn placement(&self, entry: &MemoryEntry) -> (String, String, String) {
        let wing = entry
            .metadata
            .get("wing")
            .or(entry.scope.as_ref())
            .cloned()
            .unwrap_or_else(|| self.settings().default_wing.clone());
        let room = entry
            .metadata
            .get("room")
            .cloned()
            .unwrap_or_else(|| self.settings().default_room.clone());
        let hall = entry
            .metadata
            .get("hall")
            .filter(|hall| !hall.trim().is_empty())
            .cloned()
            .unwrap_or_else(|| category_hall(entry.category).to_owned());
        (wing, room, hall)
    }
}

#[async_trait]
impl MemoryBackend for MemPalaceStoolapMemory {
    fn backend_name(&self) -> &str {
        "mempalace_stoolap"
    }

    #[instrument(
        name = "memory.mempalace.recall",
        skip(self, query),
        fields(
            memory.query_bytes = query.text.len(),
            memory.search_mode = tracing::field::Empty,
            memory.entry_count = tracing::field::Empty
        )
    )]
    async fn recall(&self, query: MemoryQuery) -> MemoryResult<MemoryRecallResult> {
        let limit = query.limit.unwrap_or(self.config.recall_limit);
        let query_embedding = if query.text.trim().is_empty() {
            None
        } else if let Some(provider) = &self.embedding_provider {
            provider.embed(&query.text).await?
        } else {
            None
        };
        if let Some(vector) = &query_embedding {
            validate_embedding(vector, self.settings().vector_dimensions)?;
        }

        let database = self.database()?;
        let mut records = if let Some(vector) = &query_embedding {
            tracing::Span::current().record("memory.search_mode", "vector");
            vector_recall(&database, self, &query, vector, limit)?
        } else {
            Vec::new()
        };
        if records.is_empty() {
            tracing::Span::current().record("memory.search_mode", "lexical");
            records = lexical_recall(&database, self, &query, limit)?;
        }
        tracing::Span::current().record("memory.entry_count", records.len());
        Ok(MemoryRecallResult { query, records })
    }

    #[instrument(
        name = "memory.mempalace.remember",
        skip(self, entry),
        fields(
            memory.category = %entry.category.as_str(),
            memory.content_bytes = entry.content.len()
        )
    )]
    async fn remember(&self, entry: MemoryEntry) -> MemoryResult<String> {
        if entry.content.trim().is_empty() {
            bail!("Memory content must not be empty");
        }
        if !self.config.enabled_categories.contains(&entry.category) {
            bail!(
                "Memory category '{}' is disabled by configuration",
                entry.category.as_str()
            );
        }
        let embedding = if let Some(provider) = &self.embedding_provider {
            provider.embed(&entry.content).await?
        } else {
            None
        };
        if let Some(vector) = &embedding {
            validate_embedding(vector, self.settings().vector_dimensions)?;
        }

        let (wing, room, hall) = self.placement(&entry);
        let id = new_drawer_id();
        let timestamp = Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true);
        let tags_json = serde_json::to_string(&entry.tags)
            .context("Failed to serialize MemPalace drawer tags")?;
        let metadata_json = serde_json::to_string(&entry.metadata)
            .context("Failed to serialize MemPalace drawer metadata")?;
        let vector_sql = embedding
            .as_deref()
            .map(vector_literal)
            .transpose()?
            .map(|vector| format!("'{vector}'"))
            .unwrap_or_else(|| "NULL".to_owned());
        let insert_drawer = format!(
            "INSERT INTO drawers \
             (id, wing, room, hall, category, content, tags_json, scope, metadata_json, \
              created_at, updated_at, embedding) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, {vector_sql})"
        );

        let database = self.database()?;
        let mut transaction = database.begin()?;
        if transaction
            .query_opt::<String, _>("SELECT name FROM wings WHERE name = $1", (&wing,))?
            .is_none()
        {
            transaction.execute(
                "INSERT INTO wings (name, created_at) VALUES ($1, $2)",
                (&wing, &timestamp),
            )?;
        }
        if transaction
            .query_opt::<String, _>(
                "SELECT name FROM rooms WHERE wing = $1 AND name = $2",
                (&wing, &room),
            )?
            .is_none()
        {
            transaction.execute(
                "INSERT INTO rooms (wing, name, created_at) VALUES ($1, $2, $3)",
                (&wing, &room, &timestamp),
            )?;
        }
        if transaction
            .query_opt::<String, _>("SELECT name FROM halls WHERE name = $1", (&hall,))?
            .is_none()
        {
            transaction.execute(
                "INSERT INTO halls (name, category, created_at) VALUES ($1, $2, $3)",
                (&hall, entry.category.as_str(), &timestamp),
            )?;
        }
        transaction.execute(
            &insert_drawer,
            (
                &id,
                &wing,
                &room,
                &hall,
                entry.category.as_str(),
                &entry.content,
                &tags_json,
                entry.scope.as_deref(),
                &metadata_json,
                &timestamp,
                &timestamp,
            ),
        )?;
        refresh_closet(&mut transaction, &wing, &room, &timestamp)?;
        transaction.commit()?;
        Ok(id)
    }

    #[instrument(name = "memory.mempalace.forget", skip(self, id))]
    async fn forget(&self, id: &str) -> MemoryResult<bool> {
        let database = self.database()?;
        let mut transaction = database.begin()?;
        let mut rows = transaction.query("SELECT wing, room FROM drawers WHERE id = $1", (id,))?;
        let Some(row) = rows.next() else {
            transaction.rollback()?;
            return Ok(false);
        };
        let row = row?;
        let wing: String = row.get_by_name("wing")?;
        let room: String = row.get_by_name("room")?;
        transaction.execute("DELETE FROM drawers WHERE id = $1", (id,))?;
        let timestamp = Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true);
        refresh_closet(&mut transaction, &wing, &room, &timestamp)?;
        transaction.commit()?;
        Ok(true)
    }

    #[instrument(name = "memory.mempalace.list", skip(self), fields(memory.limit = limit))]
    async fn list(&self, limit: usize) -> MemoryResult<Vec<MemoryRecord>> {
        let limit = sql_limit(limit)?;
        let database = self.database()?;
        query_records(
            &database,
            &format!(
                "SELECT {DRAWER_COLUMNS} FROM drawers \
                 ORDER BY updated_at DESC, id DESC LIMIT $1"
            ),
            (limit,),
        )
    }

    #[instrument(
        name = "memory.mempalace.format_context",
        skip(self, recall),
        fields(memory.entry_count = recall.records.len())
    )]
    async fn format_for_context(&self, recall: &MemoryRecallResult) -> MemoryResult<String> {
        if recall.records.is_empty() {
            return Ok(format_recall_as_toon(recall));
        }

        let database = self.database()?;
        let mut locations = BTreeSet::new();
        let mut wake_up_rows = Vec::new();
        for record in &recall.records {
            if wake_up_rows.len() >= self.settings().wake_up_limit {
                break;
            }
            let (wing, room, _) = self.placement(&record.entry);
            if !locations.insert((wing.clone(), room.clone())) {
                continue;
            }
            if let Some(index_metadata) = database.query_opt::<String, _>(
                "SELECT index_metadata FROM closets WHERE wing = $1 AND room = $2",
                (&wing, &room),
            )? {
                wake_up_rows.push(
                    [wing, room, index_metadata]
                        .iter()
                        .map(|value| crate::types::toon_string(value))
                        .collect::<Vec<_>>()
                        .join(","),
                );
            }
        }
        if wake_up_rows.is_empty() {
            return Ok(format_recall_as_toon(recall));
        }
        let wake_up = format!(
            "wake_up[{}]{{wing,room,closet}}:\n{}",
            wake_up_rows.len(),
            wake_up_rows.join("\n")
        );
        Ok(format!("{wake_up}\n\n{}", format_recall_as_toon(recall)))
    }

    #[instrument(name = "memory.mempalace.health_check", skip(self))]
    async fn health_check(&self) -> MemoryResult<()> {
        let database = self.database()?;
        let _: i64 = database.query_one("SELECT COUNT(*) FROM drawers", ())?;
        Ok(())
    }
}

fn open_database(config: &MemorySystemConfig) -> Result<Database> {
    let settings = &config.mempalace_stoolap;
    if settings.storage_mode == "memory" {
        return Database::open_in_memory()
            .context("Failed to open in-memory MemPalace Stoolap database");
    }

    if settings.database_path.as_os_str().is_empty() {
        bail!("MemPalace Stoolap database_path must not be empty");
    }
    let path_value = settings.database_path.to_string_lossy();
    if let Some(file_path) = path_value.strip_prefix("file://") {
        let (database_path, query) = file_path
            .split_once('?')
            .map_or((file_path, None), |(path, query)| (path, Some(query)));
        let configured_path = PathBuf::from(database_path);
        let database_path = if configured_path.is_absolute() {
            configured_path
        } else {
            config.storage_dir.join(configured_path)
        };
        fs::create_dir_all(&database_path).with_context(|| {
            format!(
                "Failed to create MemPalace Stoolap directory {}",
                database_path.display()
            )
        })?;
        let database_path = database_path.canonicalize().with_context(|| {
            format!(
                "Failed to resolve MemPalace Stoolap directory {}",
                database_path.display()
            )
        })?;
        let mut dsn = format!("file://{}", database_path.display());
        if let Some(query) = query {
            dsn.push('?');
            dsn.push_str(query);
        }
        let dsn = append_full_sync(&dsn, query);
        return Database::open(&dsn)
            .context("Failed to open persistent MemPalace Stoolap database");
    }

    fs::create_dir_all(&config.storage_dir).with_context(|| {
        format!(
            "Failed to create memory storage directory {}",
            config.storage_dir.display()
        )
    })?;
    let configured_path = Path::new(path_value.as_ref());
    let database_path = if configured_path.is_absolute() {
        configured_path.to_path_buf()
    } else {
        config.storage_dir.join(configured_path)
    };
    fs::create_dir_all(&database_path).with_context(|| {
        format!(
            "Failed to create MemPalace Stoolap directory {}",
            database_path.display()
        )
    })?;
    let database_path = database_path.canonicalize().with_context(|| {
        format!(
            "Failed to resolve MemPalace Stoolap directory {}",
            database_path.display()
        )
    })?;
    let dsn = format!("file://{}?sync_mode=full", database_path.display());
    Database::open(&dsn).context("Failed to open persistent MemPalace Stoolap database")
}

fn append_full_sync(dsn: &str, query: Option<&str>) -> String {
    if query.is_some_and(|query| {
        query
            .split('&')
            .any(|parameter| parameter.starts_with("sync=") || parameter.starts_with("sync_mode="))
    }) {
        dsn.to_owned()
    } else if dsn.contains('?') {
        format!("{dsn}&sync_mode=full")
    } else {
        format!("{dsn}?sync_mode=full")
    }
}

fn initialize_schema(database: &Database, settings: &MemPalaceStoolapConfig) -> Result<()> {
    database.execute(
        "CREATE TABLE IF NOT EXISTS palace_settings \
         (name TEXT NOT NULL, value TEXT NOT NULL)",
        (),
    )?;
    database.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_mempalace_settings_name \
         ON palace_settings (name)",
        (),
    )?;
    for (name, value) in [
        ("schema_version", "1".to_owned()),
        ("vector_dimensions", settings.vector_dimensions.to_string()),
        ("vector_metric", settings.vector_metric.clone()),
    ] {
        if let Some(existing) = database
            .query_opt::<String, _>("SELECT value FROM palace_settings WHERE name = $1", (name,))?
        {
            ensure!(
                existing == value,
                "MemPalace Stoolap setting '{name}' does not match the existing database"
            );
        } else {
            database.execute(
                "INSERT INTO palace_settings (name, value) VALUES ($1, $2)",
                (name, &value),
            )?;
        }
    }

    database.execute(
        "CREATE TABLE IF NOT EXISTS wings (name TEXT NOT NULL, created_at TEXT NOT NULL)",
        (),
    )?;
    database.execute(
        "CREATE TABLE IF NOT EXISTS rooms \
         (wing TEXT NOT NULL, name TEXT NOT NULL, created_at TEXT NOT NULL)",
        (),
    )?;
    database.execute(
        "CREATE TABLE IF NOT EXISTS halls \
         (name TEXT NOT NULL, category TEXT NOT NULL, created_at TEXT NOT NULL)",
        (),
    )?;
    database.execute(
        "CREATE TABLE IF NOT EXISTS closets \
         (wing TEXT NOT NULL, room TEXT NOT NULL, index_metadata TEXT NOT NULL, \
          updated_at TEXT NOT NULL)",
        (),
    )?;
    database.execute(
        &format!(
            "CREATE TABLE IF NOT EXISTS drawers \
             (id TEXT NOT NULL, wing TEXT NOT NULL, room TEXT NOT NULL, \
              hall TEXT NOT NULL, category TEXT NOT NULL, content TEXT NOT NULL, \
              tags_json TEXT NOT NULL, scope TEXT, metadata_json TEXT NOT NULL, \
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL, \
              embedding VECTOR({}))",
            settings.vector_dimensions
        ),
        (),
    )?;
    for (name, table, columns) in [
        ("idx_mempalace_wings_name", "wings", "name"),
        ("idx_mempalace_rooms_location", "rooms", "wing, name"),
        ("idx_mempalace_halls_name", "halls", "name"),
        ("idx_mempalace_closets_location", "closets", "wing, room"),
        ("idx_mempalace_drawers_id", "drawers", "id"),
    ] {
        database.execute(
            &format!("CREATE UNIQUE INDEX IF NOT EXISTS {name} ON {table} ({columns})"),
            (),
        )?;
    }
    database.execute(
        "CREATE INDEX IF NOT EXISTS idx_mempalace_drawers_room \
         ON drawers (wing, room, hall)",
        (),
    )?;
    database.execute(
        "CREATE INDEX IF NOT EXISTS idx_mempalace_drawers_scope \
         ON drawers (scope)",
        (),
    )?;
    database
        .execute(
            &format!(
                "CREATE INDEX IF NOT EXISTS idx_mempalace_drawers_embedding \
                 ON drawers (embedding) USING HNSW WITH (metric = '{}')",
                settings.vector_metric
            ),
            (),
        )
        .context("Failed to initialize MemPalace Stoolap HNSW index")?;
    Ok(())
}

fn refresh_closet(
    transaction: &mut ApiTransaction,
    wing: &str,
    room: &str,
    updated_at: &str,
) -> Result<()> {
    let mut halls = BTreeSet::new();
    let mut tags = BTreeSet::new();
    for row in transaction.query(
        "SELECT hall, tags_json FROM drawers WHERE wing = $1 AND room = $2",
        (wing, room),
    )? {
        let row = row?;
        halls.insert(row.get_by_name::<String>("hall")?);
        let tags_json: String = row.get_by_name("tags_json")?;
        tags.extend(serde_json::from_str::<Vec<String>>(&tags_json)?);
    }
    if halls.is_empty() {
        transaction.execute(
            "DELETE FROM closets WHERE wing = $1 AND room = $2",
            (wing, room),
        )?;
        return Ok(());
    }
    let halls = halls
        .into_iter()
        .take(MAX_CLOSET_HALLS)
        .collect::<Vec<_>>()
        .join("|");
    let tags = tags
        .into_iter()
        .take(MAX_CLOSET_TAGS)
        .collect::<Vec<_>>()
        .join("|");
    let index_metadata = format!("halls={halls}; tags={tags}");
    transaction.execute(
        "DELETE FROM closets WHERE wing = $1 AND room = $2",
        (wing, room),
    )?;
    transaction.execute(
        "INSERT INTO closets (wing, room, index_metadata, updated_at) \
         VALUES ($1, $2, $3, $4)",
        (wing, room, &index_metadata, updated_at),
    )?;
    Ok(())
}

fn vector_recall(
    database: &Database,
    backend: &MemPalaceStoolapMemory,
    query: &MemoryQuery,
    vector: &[f32],
    limit: usize,
) -> Result<Vec<MemoryRecord>> {
    let categories = if query.categories.is_empty() {
        backend.config.enabled_categories.clone()
    } else {
        query
            .categories
            .iter()
            .copied()
            .filter(|category| backend.config.enabled_categories.contains(category))
            .collect()
    };
    if categories.is_empty() {
        return Ok(Vec::new());
    }

    let category_filter = categories
        .iter()
        .map(|category| format!("'{}'", category.as_str()))
        .collect::<Vec<_>>()
        .join(", ");
    let mut filters = vec![
        "embedding IS NOT NULL".to_owned(),
        format!("category IN ({category_filter})"),
    ];
    if let Some(scope) = &query.scope {
        filters.push(format!("scope = '{}'", scope.replace('\'', "''")));
    }
    let metric_function = match backend.settings().vector_metric.as_str() {
        "cosine" => "VEC_DISTANCE_COSINE",
        "l2" => "VEC_DISTANCE_L2",
        _ => bail!("Unsupported MemPalace Stoolap vector metric"),
    };
    let sql = format!(
        "SELECT {DRAWER_COLUMNS} FROM drawers WHERE {} \
         ORDER BY {metric_function}(embedding, '{}') ASC LIMIT {}",
        filters.join(" AND "),
        vector_literal(vector)?,
        sql_limit(limit)?
    );
    query_records(database, &sql, ())
}

fn lexical_recall(
    database: &Database,
    backend: &MemPalaceStoolapMemory,
    query: &MemoryQuery,
    limit: usize,
) -> Result<Vec<MemoryRecord>> {
    let mut ranked = query_records(
        database,
        &format!("SELECT {DRAWER_COLUMNS} FROM drawers"),
        (),
    )?
    .into_iter()
    .filter(|record| {
        backend
            .config
            .enabled_categories
            .contains(&record.entry.category)
    })
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
        let score = lexical_score(&record, &query.text)
            * backend
                .config
                .category_weights
                .get(record.entry.category.as_str())
                .copied()
                .unwrap_or(1.0);
        (query.text.trim().is_empty() || score > 0.0).then_some((score, record))
    })
    .collect::<Vec<_>>();
    ranked.sort_by(|(left_score, left), (right_score, right)| {
        right_score
            .total_cmp(left_score)
            .then_with(|| right.updated_at.cmp(&left.updated_at))
            .then_with(|| right.id.cmp(&left.id))
    });
    Ok(ranked
        .into_iter()
        .take(limit)
        .map(|(_, record)| record)
        .collect())
}

fn query_records<P: stoolap::Params>(
    database: &Database,
    sql: &str,
    params: P,
) -> Result<Vec<MemoryRecord>> {
    database
        .query(sql, params)?
        .map(|row| row.map_err(anyhow::Error::from).and_then(record_from_row))
        .collect()
}

fn record_from_row(row: ResultRow) -> Result<MemoryRecord> {
    let category_name: String = row.get_by_name("category")?;
    Ok(MemoryRecord {
        id: row.get_by_name("id")?,
        entry: MemoryEntry {
            category: category_name.parse::<MemoryCategory>()?,
            content: row.get_by_name("content")?,
            tags: serde_json::from_str(&row.get_by_name::<String>("tags_json")?)
                .context("Failed to parse MemPalace drawer tags")?,
            scope: row.get_by_name("scope")?,
            metadata: serde_json::from_str(&row.get_by_name::<String>("metadata_json")?)
                .context("Failed to parse MemPalace drawer metadata")?,
        },
        created_at: row.get_by_name("created_at")?,
        updated_at: row.get_by_name("updated_at")?,
    })
}

fn lexical_score(record: &MemoryRecord, query: &str) -> f64 {
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

fn validate_embedding(vector: &[f32], dimensions: usize) -> Result<()> {
    ensure!(
        vector.len() == dimensions,
        "Embedding has {} values; configured vector_dimensions is {dimensions}",
        vector.len()
    );
    ensure!(
        vector.iter().all(|value| value.is_finite()),
        "Embedding values must all be finite"
    );
    Ok(())
}

fn vector_literal(vector: &[f32]) -> Result<String> {
    ensure!(
        vector.iter().all(|value| value.is_finite()),
        "Embedding values must all be finite"
    );
    Ok(format!(
        "[{}]",
        vector
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",")
    ))
}

fn sql_limit(limit: usize) -> Result<i64> {
    i64::try_from(limit).context("Memory result limit exceeds Stoolap's supported range")
}

fn category_hall(category: MemoryCategory) -> &'static str {
    match category {
        MemoryCategory::Preference => "preferences",
        MemoryCategory::Fact => "facts",
        MemoryCategory::Feedback => "advice",
        MemoryCategory::Reference => "discoveries",
    }
}

fn new_drawer_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!(
        "drawer-{}-{nanos}-{}",
        std::process::id(),
        NEXT_DRAWER_ID.fetch_add(1, Ordering::Relaxed)
    )
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        fs,
        path::PathBuf,
        sync::Arc,
        time::{SystemTime, UNIX_EPOCH},
    };

    use anyhow::Result;
    use async_trait::async_trait;
    use opentelemetry::{Value, trace::TracerProvider as _};
    use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider, SpanData};
    use tracing_subscriber::prelude::*;

    use super::{MemPalaceStoolapMemory, MemoryEmbeddingProvider};
    use crate::memory::{
        MemoryBackend, MemoryCategory, MemoryEntry, MemoryQuery, MemoryRegistry,
        config::MemorySystemConfig,
    };

    fn config() -> MemorySystemConfig {
        let mut config = MemorySystemConfig::default();
        config.backend = "mempalace_stoolap".to_owned();
        config.mempalace_stoolap.storage_mode = "memory".to_owned();
        config.mempalace_stoolap.vector_dimensions = 2;
        config
    }

    fn entry(content: &str) -> MemoryEntry {
        MemoryEntry {
            category: MemoryCategory::Fact,
            content: content.to_owned(),
            tags: vec!["verbatim".to_owned(), "stoolap".to_owned()],
            scope: Some("project-alpha".to_owned()),
            metadata: BTreeMap::from([
                ("room".to_owned(), "design".to_owned()),
                ("hall".to_owned(), "discoveries".to_owned()),
            ]),
        }
    }

    fn temp_directory(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("mempalace-{label}-{}-{nonce}", std::process::id()))
    }

    fn attribute<'a>(span: &'a SpanData, name: &str) -> Option<&'a Value> {
        span.attributes
            .iter()
            .find(|attribute| attribute.key.as_str() == name)
            .map(|attribute| &attribute.value)
    }

    #[tokio::test]
    async fn mempalace_stoolap_keeps_verbatim_drawers_and_wake_up_context() {
        let backend = MemPalaceStoolapMemory::open(&config()).unwrap();
        let original = entry("verbatim marker:  exact wording stays untouched.\nsecond line");
        let id = backend.remember(original.clone()).await.unwrap();

        let listed = backend.list(10).await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, id);
        assert_eq!(listed[0].entry, original);

        let recall = backend
            .recall(MemoryQuery {
                text: "verbatim marker".to_owned(),
                scope: Some("project-alpha".to_owned()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(recall.records.len(), 1);
        let context = backend.format_for_context(&recall).await.unwrap();
        assert!(context.contains("wake_up[1]{wing,room,closet}:"));
        assert!(context.contains("project-alpha"));
        assert!(context.contains("design"));
        assert!(context.contains("halls=discoveries"));
        assert!(context.contains("verbatim marker:"));
        assert!(context.contains("second line"));
        assert!(backend.forget(&id).await.unwrap());
        assert!(!backend.forget(&id).await.unwrap());
    }

    #[tokio::test]
    async fn mempalace_stoolap_persists_drawers_across_reopen() -> Result<()> {
        let root = temp_directory("restart");
        let mut config = config();
        config.mempalace_stoolap.storage_mode = "persistent".to_owned();
        config.storage_dir = root.clone();
        config.mempalace_stoolap.database_path = PathBuf::from("palace");
        let backend = MemPalaceStoolapMemory::open(&config)?;
        let id = backend
            .remember(entry("persistent verbatim marker"))
            .await?;
        drop(backend);

        let reopened = MemPalaceStoolapMemory::open(&config)?;
        let listed = reopened.list(10).await?;
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, id);
        assert_eq!(listed[0].entry.content, "persistent verbatim marker");
        drop(reopened);

        config.mempalace_stoolap.database_path = PathBuf::from(format!(
            "file://{}?sync=full",
            root.join("url-palace").display()
        ));
        let url_backend = MemPalaceStoolapMemory::open(&config)?;
        let url_id = url_backend
            .remember(entry("file URL persistence marker"))
            .await?;
        drop(url_backend);
        let reopened_url_backend = MemPalaceStoolapMemory::open(&config)?;
        assert_eq!(reopened_url_backend.list(10).await?[0].id, url_id);
        drop(reopened_url_backend);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    struct FixedEmbeddingProvider;

    #[async_trait]
    impl MemoryEmbeddingProvider for FixedEmbeddingProvider {
        fn dimensions(&self) -> usize {
            2
        }

        async fn embed(&self, text: &str) -> crate::memory::MemoryResult<Option<Vec<f32>>> {
            let vector = if text == "vector query" || text.contains("near") {
                vec![0.0, 1.0]
            } else {
                vec![1.0, 0.0]
            };
            Ok(Some(vector))
        }
    }

    #[tokio::test]
    async fn mempalace_stoolap_uses_native_vector_search_when_provider_is_available() {
        let backend = MemPalaceStoolapMemory::open_with_embedding_provider(
            &config(),
            Some(Arc::new(FixedEmbeddingProvider)),
        )
        .unwrap();
        backend.remember(entry("far drawer")).await.unwrap();
        backend.remember(entry("near drawer")).await.unwrap();
        let recall = backend
            .recall(MemoryQuery {
                text: "vector query".to_owned(),
                limit: Some(1),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(recall.records.len(), 1);
        assert_eq!(recall.records[0].entry.content, "near drawer");
    }

    #[tokio::test]
    async fn otel_mempalace_operations_export_backend_and_operation_attributes() {
        let _subscriber_lock = crate::test_telemetry::lock_test_subscriber();
        let exporter = InMemorySpanExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        let subscriber = tracing_subscriber::registry()
            .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("mempalace-otel")));
        let _guard = tracing::subscriber::set_default(subscriber);
        let mut memory_config = config();
        memory_config.recall_limit = 3;
        let backend = MemoryRegistry::new().build(&memory_config).unwrap();
        backend.health_check().await.unwrap();
        let id = backend.remember(entry("otel drawer marker")).await.unwrap();
        let recall = backend
            .recall(MemoryQuery {
                text: "otel drawer".to_owned(),
                ..Default::default()
            })
            .await
            .unwrap();
        backend.format_for_context(&recall).await.unwrap();
        backend.list(10).await.unwrap();
        backend.forget(&id).await.unwrap();

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
                .unwrap_or_else(|| panic!("missing OTel span for memory operation {operation}"));
            assert!(matches!(
                attribute(span, "memory.backend"),
                Some(Value::String(value)) if value.as_str() == "mempalace_stoolap"
            ));
            assert!(attribute(span, "memory.duration_ms").is_some());
            assert!(attribute(span, "memory.entry_count").is_some());
            assert!(matches!(
                attribute(span, "memory.outcome"),
                Some(Value::String(value)) if value.as_str() == "ok"
            ));
        }
        drop(_guard);
        provider.shutdown().unwrap();
        assert!(exporter.is_shutdown_called());
    }
}
