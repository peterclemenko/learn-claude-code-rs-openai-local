use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use tracing::{error, info, instrument};

use super::MemoryCategory;

/// Selects a memory backend and defines its shared recall and storage policy.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MemorySystemConfig {
    pub backend: String,
    pub storage_dir: PathBuf,
    pub recall_limit: usize,
    #[serde(default = "default_enabled_categories")]
    pub enabled_categories: Vec<MemoryCategory>,
    #[serde(default = "default_category_weights")]
    pub category_weights: BTreeMap<String, f64>,
    #[serde(default)]
    pub backend_options: BTreeMap<String, String>,
    #[serde(default)]
    pub mempalace_stoolap: MemPalaceStoolapConfig,
}

/// Stoolap storage, vector, hierarchy, and wake-up layer settings.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MemPalaceStoolapConfig {
    /// Selects durable on-disk storage or an isolated in-memory database.
    pub storage_mode: String,
    /// Stoolap database directory, absolute or relative to `storage_dir`.
    pub database_path: PathBuf,
    /// Dimension of the native Stoolap vector column and HNSW index.
    pub vector_dimensions: usize,
    /// Vector distance metric: `cosine` or `l2`.
    pub vector_metric: String,
    /// Default top-level palace scope for entries without an explicit scope.
    pub default_wing: String,
    /// Default topic room for entries without `metadata["room"]`.
    pub default_room: String,
    /// Maximum number of room wake-up index rows included in prompt context.
    pub wake_up_limit: usize,
}

impl Default for MemPalaceStoolapConfig {
    fn default() -> Self {
        Self {
            storage_mode: "persistent".to_owned(),
            database_path: PathBuf::from("mempalace-db"),
            vector_dimensions: 384,
            vector_metric: "cosine".to_owned(),
            default_wing: "project".to_owned(),
            default_room: "general".to_owned(),
            wake_up_limit: 8,
        }
    }
}

impl MemorySystemConfig {
    /// Loads and validates a RON memory configuration from `path`.
    #[instrument(
        name = "memory.config.load",
        skip(path),
        fields(
            memory.config_path = %path.as_ref().display(),
            memory.backend = tracing::field::Empty,
            memory.outcome = tracing::field::Empty
        )
    )]
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let contents = match fs::read_to_string(path) {
            Ok(contents) => contents,
            Err(failure) => {
                tracing::Span::current().record("memory.outcome", "error");
                error!(error = %failure, "Failed to read memory configuration");
                return Err(failure).with_context(|| {
                    format!("Failed to read memory configuration {}", path.display())
                });
            }
        };
        let config: Self = match ron::from_str(&contents) {
            Ok(config) => config,
            Err(failure) => {
                tracing::Span::current().record("memory.outcome", "error");
                error!(error = %failure, "Failed to parse memory configuration");
                return Err(failure).with_context(|| {
                    format!("Failed to parse memory configuration {}", path.display())
                });
            }
        };
        let config = config.resolve_storage_dir(path);
        if let Err(failure) = config.validate() {
            tracing::Span::current().record("memory.outcome", "error");
            error!(error = %failure, "Memory configuration validation failed");
            return Err(failure);
        }
        tracing::Span::current().record("memory.backend", config.backend.as_str());
        tracing::Span::current().record("memory.outcome", "ok");
        info!(memory.backend = %config.backend, "Loaded memory configuration");
        Ok(config)
    }

    /// Loads a RON memory configuration or uses validated defaults if it is absent.
    #[instrument(
        name = "memory.config.load_or_default",
        skip(path),
        fields(
            memory.config_path = %path.as_ref().display(),
            memory.backend = tracing::field::Empty,
            memory.outcome = tracing::field::Empty
        )
    )]
    pub fn load_or_default(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        match fs::metadata(path) {
            Ok(_) => match Self::load(path) {
                Ok(config) => {
                    tracing::Span::current().record("memory.backend", config.backend.as_str());
                    tracing::Span::current().record("memory.outcome", "ok");
                    Ok(config)
                }
                Err(failure) => {
                    tracing::Span::current().record("memory.outcome", "error");
                    Err(failure)
                }
            },
            Err(failure) if failure.kind() == ErrorKind::NotFound => {
                let config = Self::default();
                config.validate()?;
                tracing::Span::current().record("memory.backend", config.backend.as_str());
                tracing::Span::current().record("memory.outcome", "default");
                info!(
                    config_path = %path.display(),
                    memory.backend = %config.backend,
                    "Memory configuration is absent; using defaults"
                );
                Ok(config)
            }
            Err(failure) => {
                tracing::Span::current().record("memory.outcome", "error");
                error!(error = %failure, "Failed to read memory configuration");
                Err(failure).with_context(|| {
                    format!("Failed to read memory configuration {}", path.display())
                })
            }
        }
    }

    /// Validates shared configuration values and category weights.
    #[instrument(
        name = "memory.config.validate",
        skip(self),
        fields(
            memory.backend = %self.backend,
            memory.outcome = tracing::field::Empty
        )
    )]
    pub fn validate(&self) -> Result<()> {
        match self.validate_values() {
            Ok(()) => {
                tracing::Span::current().record("memory.outcome", "ok");
                Ok(())
            }
            Err(failure) => {
                tracing::Span::current().record("memory.outcome", "error");
                error!(error = %failure, "Memory configuration validation failed");
                Err(failure)
            }
        }
    }

    fn validate_values(&self) -> Result<()> {
        if self.backend.trim().is_empty() {
            bail!("Memory backend name must not be empty");
        }
        if self.storage_dir.as_os_str().is_empty() {
            bail!("Memory storage_dir must not be empty");
        }
        if self.recall_limit == 0 {
            bail!("Memory recall_limit must be in the range 1..=i64::MAX");
        }
        if i64::try_from(self.recall_limit).is_err() {
            bail!("Memory recall_limit must be in the range 1..=i64::MAX");
        }
        let mut categories = BTreeSet::new();
        for category in &self.enabled_categories {
            if !categories.insert(*category) {
                bail!(
                    "Memory category '{:?}' is configured more than once",
                    category
                );
            }
        }
        for (category, weight) in &self.category_weights {
            if category.trim().is_empty() {
                bail!("Memory category weight names must not be empty");
            }
            let parsed_category = category
                .parse::<MemoryCategory>()
                .with_context(|| format!("Unknown memory category weight '{category}'"))?;
            if parsed_category.as_str() != category {
                bail!(
                    "Memory category weight '{category}' must use canonical spelling '{}'",
                    parsed_category.as_str()
                );
            }
            if !weight.is_finite() || *weight < 0.0 {
                bail!("Memory category weight for '{category}' must be finite and non-negative");
            }
        }
        if self.backend == "mempalace_stoolap" {
            let palace = &self.mempalace_stoolap;
            if !matches!(palace.storage_mode.as_str(), "persistent" | "memory") {
                bail!("MemPalace Stoolap storage_mode must be 'persistent' or 'memory'");
            }
            if palace.storage_mode == "persistent" && palace.database_path.as_os_str().is_empty() {
                bail!("MemPalace Stoolap database_path must not be empty");
            }
            if palace.vector_dimensions == 0 {
                bail!("MemPalace Stoolap vector_dimensions must be greater than zero");
            }
            if !matches!(palace.vector_metric.as_str(), "cosine" | "l2") {
                bail!("MemPalace Stoolap vector_metric must be 'cosine' or 'l2'");
            }
            if palace.default_wing.trim().is_empty() || palace.default_room.trim().is_empty() {
                bail!("MemPalace Stoolap default wing and room must not be empty");
            }
            if palace.wake_up_limit == 0 {
                bail!("MemPalace Stoolap wake_up_limit must be greater than zero");
            }
        }
        Ok(())
    }

    fn resolve_storage_dir(mut self, config_path: &Path) -> Self {
        if self.storage_dir.is_relative()
            && let Some(parent) = config_path.parent()
        {
            self.storage_dir = parent.join(&self.storage_dir);
        }
        self
    }

    pub(crate) fn memory_file_name(&self) -> &str {
        self.backend_options
            .get("file_name")
            .map(String::as_str)
            .unwrap_or("memories.ron")
    }
}

impl Default for MemorySystemConfig {
    fn default() -> Self {
        Self {
            backend: "naive_memory_system".to_owned(),
            storage_dir: PathBuf::from(".local/memory"),
            recall_limit: 5,
            enabled_categories: default_enabled_categories(),
            category_weights: default_category_weights(),
            backend_options: BTreeMap::new(),
            mempalace_stoolap: MemPalaceStoolapConfig::default(),
        }
    }
}

fn default_enabled_categories() -> Vec<MemoryCategory> {
    vec![
        MemoryCategory::Preference,
        MemoryCategory::Fact,
        MemoryCategory::Feedback,
        MemoryCategory::Reference,
    ]
}

fn default_category_weights() -> BTreeMap<String, f64> {
    BTreeMap::from([
        ("Preference".to_owned(), 2.0),
        ("Fact".to_owned(), 1.0),
        ("Feedback".to_owned(), 1.5),
        ("Reference".to_owned(), 0.75),
    ])
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf, time::SystemTime};

    use crate::memory::MemoryRegistry;

    use super::MemorySystemConfig;

    struct ConfigBuildCleanup {
        paths: Vec<PathBuf>,
    }

    impl ConfigBuildCleanup {
        fn new(config: &MemorySystemConfig) -> Self {
            let mut paths = Vec::new();
            match config.backend.as_str() {
                "naive_memory_system" if !config.storage_dir.exists() => {
                    paths.push(config.storage_dir.clone());
                }
                "mempalace_stoolap" if config.mempalace_stoolap.storage_mode == "persistent" => {
                    let configured_path = config.mempalace_stoolap.database_path.to_string_lossy();
                    let configured_path = configured_path
                        .strip_prefix("file://")
                        .unwrap_or(&configured_path);
                    let configured_path = configured_path
                        .split_once('?')
                        .map_or(configured_path, |(path, _)| path);
                    let configured_path = PathBuf::from(configured_path);
                    let database_path = if configured_path.is_absolute() {
                        configured_path
                    } else {
                        config.storage_dir.join(configured_path)
                    };
                    if !database_path.exists() {
                        if !config.storage_dir.exists()
                            && database_path.starts_with(&config.storage_dir)
                            && database_path != config.storage_dir
                        {
                            paths.push(config.storage_dir.clone());
                        }
                        paths.push(database_path);
                    }
                }
                _ => {}
            }
            Self { paths }
        }
    }

    impl Drop for ConfigBuildCleanup {
        fn drop(&mut self) {
            for path in self.paths.iter().rev() {
                if let Err(failure) = fs::remove_dir_all(path)
                    && failure.kind() != std::io::ErrorKind::NotFound
                {
                    tracing::warn!(
                        path = %path.display(),
                        error = %failure,
                        "Failed to clean up isolated checked-in memory config test storage"
                    );
                }
            }
        }
    }

    fn temp_dir() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("memory-config-test-{}-{nonce}", std::process::id()))
    }

    #[test]
    fn memory_config_loads_from_ron_and_resolves_relative_storage() {
        let root = temp_dir();
        fs::create_dir_all(&root).unwrap();
        let path = root.join("memory.ron");
        fs::write(
            &path,
            r#"(
                backend: "custom",
                storage_dir: "store",
                recall_limit: 4,
                enabled_categories: [Preference, Fact],
                category_weights: {"Preference": 2.5},
                backend_options: {"provider": "local"},
            )"#,
        )
        .unwrap();

        let config = MemorySystemConfig::load(&path).unwrap();
        assert_eq!(config.backend, "custom");
        assert_eq!(config.storage_dir, root.join("store"));
        assert_eq!(config.backend_options["provider"], "local");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn absent_memory_config_uses_validated_defaults() {
        let path = temp_dir().join("missing.ron");
        let config = MemorySystemConfig::load_or_default(&path).unwrap();
        assert_eq!(config.backend, "naive_memory_system");
        assert_eq!(config.recall_limit, 5);
        assert!(config.backend_options.is_empty());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn invalid_memory_config_is_rejected() {
        let mut config = MemorySystemConfig::default();
        config.recall_limit = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn mempalace_config_rejects_invalid_storage_and_vector_settings() {
        let mut config = MemorySystemConfig::default();
        config.backend = "mempalace_stoolap".to_owned();
        assert!(config.validate().is_ok());

        config.mempalace_stoolap.vector_dimensions = 0;
        assert!(config.validate().is_err());
        config.mempalace_stoolap.vector_dimensions = 384;
        config.mempalace_stoolap.vector_metric = "dot".to_owned();
        assert!(config.validate().is_err());
        config.mempalace_stoolap.vector_metric = "cosine".to_owned();
        config.mempalace_stoolap.storage_mode = "remote".to_owned();
        assert!(config.validate().is_err());
    }

    #[test]
    fn memory_ron_rejects_unknown_fields_and_typed_value_mismatches() {
        let root = temp_dir();
        fs::create_dir_all(&root).unwrap();
        let invalid_configs = [
            r#"(
                backend: "mempalace_stoolap",
                storage_dir: "store",
                recall_limit: 5,
                mempalace_stoolap: (
                    storage_mode: "memory",
                    database_path: "unused",
                    vector_dimensions: "384",
                    vector_metric: "cosine",
                    default_wing: "project",
                    default_room: "general",
                    wake_up_limit: 8,
                ),
            )"#,
            r#"(
                backend: "mempalace_stoolap",
                storage_dir: "store",
                recall_limit: 5,
                mempalace_stoolap: (
                    storage_mode: "memory",
                    database_path: "unused",
                    vector_dimensions: 384,
                    vector_metric: "cosine",
                    default_wing: "project",
                    default_room: "general",
                    wake_up_limit: 8,
                    unknown_setting: true,
                ),
            )"#,
        ];
        for (index, contents) in invalid_configs.into_iter().enumerate() {
            let path = root.join(format!("invalid-{index}.ron"));
            fs::write(&path, contents).unwrap();
            assert!(MemorySystemConfig::load(path).is_err());
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn omitted_category_settings_load_memory_defaults() {
        let root = temp_dir();
        fs::create_dir_all(&root).unwrap();
        let path = root.join("memory.ron");
        fs::write(
            &path,
            r#"(backend: "noop", storage_dir: "store", recall_limit: 2)"#,
        )
        .unwrap();

        let config = MemorySystemConfig::load(&path).unwrap();
        assert_eq!(config.enabled_categories.len(), 4);
        assert_eq!(config.category_weights["Preference"], 2.0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn checked_in_memory_config_parses() {
        let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let config = MemorySystemConfig::load(workspace_root.join("config/memory.ron")).unwrap();
        let registry = MemoryRegistry::new();
        let backends = registry.available_backends();
        assert!(
            backends.contains(&config.backend),
            "checked-in backend '{}' is not registered; available backends: {backends:?}",
            config.backend
        );
        let _cleanup = ConfigBuildCleanup::new(&config);
        let backend = registry
            .build(&config)
            .expect("checked-in memory backend should build");
        assert_eq!(backend.backend_name(), config.backend);
        assert_eq!(config.recall_limit, 5);
    }

    #[test]
    fn builtin_backends_parse_and_build_from_ron() {
        let root = temp_dir();
        fs::create_dir_all(&root).unwrap();
        let registry = MemoryRegistry::new();
        let configurations = [
            (
                "naive_memory_system",
                r#"(
                    backend: "naive_memory_system",
                    storage_dir: "naive-store",
                    recall_limit: 5,
                    backend_options: {"file_name": "memories.ron"},
                )"#,
            ),
            (
                "mempalace_stoolap",
                r#"(
                    backend: "mempalace_stoolap",
                    storage_dir: "palace-store",
                    recall_limit: 5,
                    mempalace_stoolap: (
                        storage_mode: "memory",
                        database_path: "unused",
                        vector_dimensions: 16,
                        vector_metric: "cosine",
                        default_wing: "project",
                        default_room: "general",
                        wake_up_limit: 8,
                    ),
                )"#,
            ),
            (
                "noop",
                r#"(
                    backend: "noop",
                    storage_dir: "noop-store",
                    recall_limit: 5,
                )"#,
            ),
        ];

        for (backend_name, contents) in configurations {
            let path = root.join(format!("memory-{backend_name}.ron"));
            fs::write(&path, contents).unwrap();
            let config = MemorySystemConfig::load(&path).unwrap();
            let backend = registry.build(&config).unwrap();
            assert_eq!(backend.backend_name(), backend_name);
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unknown_backend_in_ron_reports_registered_backends() {
        let root = temp_dir();
        fs::create_dir_all(&root).unwrap();
        let path = root.join("memory.ron");
        fs::write(
            &path,
            r#"(
                backend: "not_registered",
                storage_dir: "store",
                recall_limit: 5,
            )"#,
        )
        .unwrap();
        let config = MemorySystemConfig::load(&path).unwrap();
        let failure = MemoryRegistry::new()
            .build(&config)
            .err()
            .unwrap()
            .to_string();
        assert!(failure.contains("mempalace_stoolap"));
        assert!(failure.contains("naive_memory_system"));
        assert!(failure.contains("noop"));
        fs::remove_dir_all(root).unwrap();
    }
}
