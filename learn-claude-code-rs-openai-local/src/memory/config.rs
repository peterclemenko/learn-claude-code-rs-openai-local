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
    pub fn validate(&self) -> Result<()> {
        if self.backend.trim().is_empty() {
            bail!("Memory backend name must not be empty");
        }
        if self.storage_dir.as_os_str().is_empty() {
            bail!("Memory storage_dir must not be empty");
        }
        if self.recall_limit == 0 {
            bail!("Memory recall_limit must be greater than zero");
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
            backend_options: BTreeMap::from([("file_name".to_owned(), "memories.ron".to_owned())]),
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

    use super::MemorySystemConfig;

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
                backend: "noop",
                storage_dir: "store",
                recall_limit: 4,
                enabled_categories: [Preference, Fact],
                category_weights: {"Preference": 2.5},
                backend_options: {"provider": "local"},
            )"#,
        )
        .unwrap();

        let config = MemorySystemConfig::load(&path).unwrap();
        assert_eq!(config.backend, "noop");
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
    }

    #[test]
    fn invalid_memory_config_is_rejected() {
        let mut config = MemorySystemConfig::default();
        config.recall_limit = 0;
        assert!(config.validate().is_err());
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
        assert_eq!(config.backend, "naive_memory_system");
        assert_eq!(config.recall_limit, 5);
    }
}
