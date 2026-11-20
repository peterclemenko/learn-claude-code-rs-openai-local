use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock, RwLock},
};

use anyhow::{Context, Result, bail};
use tracing::{error, info, instrument};

use super::{
    MemPalaceStoolapMemory, MemoryBackend, MemoryResult, NaiveMemorySystem, NoopMemoryBackend,
    config::MemorySystemConfig, observed_backend,
};

type BackendBuilder =
    Arc<dyn Fn(&MemorySystemConfig) -> MemoryResult<Arc<dyn MemoryBackend>> + Send + Sync>;
type BackendValidator = Arc<dyn Fn(&MemorySystemConfig) -> MemoryResult<()> + Send + Sync>;

#[derive(Clone)]
struct BackendRegistration {
    builder: BackendBuilder,
    validate: BackendValidator,
}

/// Maps backend identifiers to constructors for pluggable implementations.
#[derive(Clone, Default)]
pub struct MemoryRegistry {
    builders: BTreeMap<String, BackendRegistration>,
}

static DEFAULT_REGISTRY: OnceLock<RwLock<MemoryRegistry>> = OnceLock::new();

fn default_registry() -> &'static RwLock<MemoryRegistry> {
    DEFAULT_REGISTRY.get_or_init(|| RwLock::new(MemoryRegistry::new()))
}

impl MemoryRegistry {
    /// Creates a registry containing all built-in memory backends.
    pub fn new() -> Self {
        let mut registry = Self::default();
        registry.builders.insert(
            "naive_memory_system".to_owned(),
            BackendRegistration {
                builder: Arc::new(|config| Ok(Arc::new(NaiveMemorySystem::open(config)?))),
                validate: Arc::new(|config| validate_backend_options(config, &["file_name"])),
            },
        );
        registry.builders.insert(
            "mempalace_stoolap".to_owned(),
            BackendRegistration {
                builder: Arc::new(|config| Ok(Arc::new(MemPalaceStoolapMemory::open(config)?))),
                validate: Arc::new(|config| validate_backend_options(config, &[])),
            },
        );
        registry.builders.insert(
            "noop".to_owned(),
            BackendRegistration {
                builder: Arc::new(|_| Ok(Arc::new(NoopMemoryBackend))),
                validate: Arc::new(|config| validate_backend_options(config, &[])),
            },
        );
        registry
    }

    /// Returns every registered backend identifier in deterministic sorted order.
    pub fn available_backends(&self) -> Vec<String> {
        self.builders.keys().cloned().collect()
    }

    /// Registers a custom backend constructor under a unique identifier.
    ///
    /// Custom builders accept any `backend_options` keys by default. Use
    /// [`Self::register_with_validator`] to validate a backend-specific schema.
    #[instrument(
        name = "memory.backend.register_local",
        skip(self, name, builder),
        fields(memory.backend = tracing::field::Empty, memory.outcome = tracing::field::Empty)
    )]
    pub fn register(
        &mut self,
        name: impl Into<String>,
        builder: impl Fn(&MemorySystemConfig) -> MemoryResult<Arc<dyn MemoryBackend>>
        + Send
        + Sync
        + 'static,
    ) -> Result<()> {
        self.register_inner(name.into(), builder, |_| Ok(()))
    }

    /// Registers a custom backend with explicit config validation before construction.
    #[instrument(
        name = "memory.backend.register_validated",
        skip(self, name, builder, validator),
        fields(memory.backend = tracing::field::Empty, memory.outcome = tracing::field::Empty)
    )]
    pub fn register_with_validator(
        &mut self,
        name: impl Into<String>,
        builder: impl Fn(&MemorySystemConfig) -> MemoryResult<Arc<dyn MemoryBackend>>
        + Send
        + Sync
        + 'static,
        validator: impl Fn(&MemorySystemConfig) -> MemoryResult<()> + Send + Sync + 'static,
    ) -> Result<()> {
        self.register_inner(name.into(), builder, validator)
    }

    fn register_inner(
        &mut self,
        name: String,
        builder: impl Fn(&MemorySystemConfig) -> MemoryResult<Arc<dyn MemoryBackend>>
        + Send
        + Sync
        + 'static,
        validator: impl Fn(&MemorySystemConfig) -> MemoryResult<()> + Send + Sync + 'static,
    ) -> Result<()> {
        tracing::Span::current().record("memory.backend", name.as_str());
        if name.trim().is_empty() {
            let failure = anyhow::anyhow!("Memory backend name must not be empty");
            tracing::Span::current().record("memory.outcome", "error");
            error!(error = %failure, "Memory backend registration failed");
            return Err(failure);
        }
        if self.builders.contains_key(&name) {
            let failure = anyhow::anyhow!("Memory backend '{name}' is already registered");
            tracing::Span::current().record("memory.outcome", "error");
            error!(memory.backend = %name, error = %failure, "Memory backend registration failed");
            return Err(failure);
        }
        self.builders.insert(
            name,
            BackendRegistration {
                builder: Arc::new(builder),
                validate: Arc::new(validator),
            },
        );
        tracing::Span::current().record("memory.outcome", "ok");
        Ok(())
    }

    #[instrument(
        name = "memory.backend.build",
        skip(self, config),
        fields(
            memory.backend = %config.backend,
            memory.outcome = tracing::field::Empty
        )
    )]
    pub fn build(&self, config: &MemorySystemConfig) -> Result<Arc<dyn MemoryBackend>> {
        let Some(registration) = self.builders.get(&config.backend) else {
            let failure = anyhow::anyhow!(
                "Unknown memory backend '{}'. Registered backends: {:?}",
                config.backend,
                self.available_backends()
            );
            tracing::Span::current().record("memory.outcome", "error");
            error!(memory.backend = %config.backend, error = %failure, "Memory backend is not registered");
            return Err(failure);
        };
        if let Err(failure) = config.validate() {
            tracing::Span::current().record("memory.outcome", "error");
            error!(error = %failure, "Memory backend configuration is invalid");
            return Err(failure);
        }
        if let Err(failure) = (registration.validate)(config) {
            let failure = anyhow::anyhow!(
                "Invalid configuration for memory backend '{}': {failure:#}",
                config.backend
            );
            tracing::Span::current().record("memory.outcome", "error");
            error!(memory.backend = %config.backend, error = %failure, "Memory backend configuration validation failed");
            return Err(failure);
        }
        let backend = match (registration.builder)(config)
            .with_context(|| format!("Failed to construct memory backend '{}'", config.backend))
        {
            Ok(backend) => backend,
            Err(failure) => {
                tracing::Span::current().record("memory.outcome", "error");
                error!(memory.backend = %config.backend, error = %failure, "Memory backend construction failed");
                return Err(failure);
            }
        };
        let name = backend.backend_name().to_owned();
        tracing::Span::current().record("memory.outcome", "ok");
        info!(memory.backend = %name, "Selected memory backend");
        Ok(observed_backend(backend))
    }
}

fn validate_backend_options(
    config: &MemorySystemConfig,
    valid_options: &[&str],
) -> MemoryResult<()> {
    let invalid_options = config
        .backend_options
        .keys()
        .filter(|option| !valid_options.contains(&option.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if !invalid_options.is_empty() {
        bail!(
            "Backend '{}' does not accept backend_options keys {:?}; valid options are {:?}",
            config.backend,
            invalid_options,
            valid_options
        );
    }
    Ok(())
}

/// Registers a backend in the process-wide registry used by agent startup.
#[instrument(name = "memory.backend.register", skip(name, builder), fields(memory.backend = tracing::field::Empty))]
pub fn register_memory_backend(
    name: impl AsRef<str>,
    builder: impl Fn(&MemorySystemConfig) -> MemoryResult<Arc<dyn MemoryBackend>>
    + Send
    + Sync
    + 'static,
) -> Result<()> {
    let name = name.as_ref().to_owned();
    tracing::Span::current().record("memory.backend", name.as_str());
    register_memory_backend_with_validator(name, builder, |_| Ok(()))
}

/// Registers a process-wide backend and validates its configuration before building it.
#[instrument(
    name = "memory.backend.register_validated",
    skip(name, builder, validator),
    fields(memory.backend = tracing::field::Empty)
)]
pub fn register_memory_backend_with_validator(
    name: impl AsRef<str>,
    builder: impl Fn(&MemorySystemConfig) -> MemoryResult<Arc<dyn MemoryBackend>>
    + Send
    + Sync
    + 'static,
    validator: impl Fn(&MemorySystemConfig) -> MemoryResult<()> + Send + Sync + 'static,
) -> Result<()> {
    let name = name.as_ref().to_owned();
    tracing::Span::current().record("memory.backend", name.as_str());
    let mut registry = match default_registry().write() {
        Ok(registry) => registry,
        Err(failure) => {
            let failure = anyhow::anyhow!("Memory registry lock is poisoned: {failure}");
            tracing::Span::current().record("memory.outcome", "error");
            error!(error = %failure, "Memory backend registration failed");
            return Err(failure);
        }
    };
    registry.register_with_validator(name, builder, validator)
}

/// Builds the configured backend and wraps its operations in OTel instrumentation.
#[instrument(name = "memory.backend.select", skip(config), fields(memory.backend = %config.backend))]
pub fn build_memory_from_config(config: &MemorySystemConfig) -> Result<Arc<dyn MemoryBackend>> {
    let registry = match default_registry().read() {
        Ok(registry) => registry.clone(),
        Err(failure) => {
            let failure = anyhow::anyhow!("Memory registry lock is poisoned: {failure}");
            error!(memory.backend = %config.backend, error = %failure, "Memory registry lookup failed");
            return Err(failure);
        }
    };
    registry
        .build(config)
        .context("Failed to build configured memory backend")
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::SystemTime};

    use super::{MemoryBackend, MemoryRegistry};
    use crate::memory::{
        MemoryCategory, MemoryEntry, MemoryQuery, NoopMemoryBackend, config::MemorySystemConfig,
    };

    #[tokio::test]
    async fn registry_selects_noop_or_naive_memory_from_configuration() {
        let mut config = MemorySystemConfig::default();
        config.backend = "noop".to_owned();
        let noop = MemoryRegistry::new().build(&config).unwrap();
        assert_eq!(noop.backend_name(), "noop");

        config.backend = "naive_memory_system".to_owned();
        config.storage_dir = std::env::temp_dir().join(format!(
            "memory-registry-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let naive = MemoryRegistry::new().build(&config).unwrap();
        assert_eq!(naive.backend_name(), "naive_memory_system");
        let id = naive
            .remember(MemoryEntry {
                category: MemoryCategory::Fact,
                content: "registry selection persists data".to_owned(),
                tags: Vec::new(),
                scope: None,
                metadata: Default::default(),
            })
            .await
            .unwrap();
        assert_eq!(
            naive
                .recall(MemoryQuery {
                    text: "persists".to_owned(),
                    ..Default::default()
                })
                .await
                .unwrap()
                .records[0]
                .id,
            id
        );
        tokio::fs::remove_dir_all(&config.storage_dir)
            .await
            .unwrap();
    }

    #[test]
    fn registry_accepts_custom_backend_builders() {
        let mut registry = MemoryRegistry::new();
        registry
            .register("custom", |_| Ok(Arc::new(NoopMemoryBackend)))
            .unwrap();
        let mut config = MemorySystemConfig::default();
        config.backend = "custom".to_owned();
        let backend = registry.build(&config).unwrap();
        assert_eq!(backend.backend_name(), "noop");
    }

    #[test]
    fn registry_rejects_duplicate_backend_names() {
        let mut registry = MemoryRegistry::new();
        assert!(
            registry
                .register("noop", |_| Ok(Arc::new(NoopMemoryBackend)))
                .is_err()
        );
    }

    #[test]
    fn registry_lists_backend_ids_and_reports_unknown_backend_options() {
        let registry = MemoryRegistry::new();
        assert_eq!(
            registry.available_backends(),
            ["mempalace_stoolap", "naive_memory_system", "noop"]
        );

        let mut config = MemorySystemConfig::default();
        config.backend = "noop".to_owned();
        config
            .backend_options
            .insert("typo".to_owned(), "value".to_owned());
        let failure = registry.build(&config).err().unwrap().to_string();
        assert!(failure.contains("typo"));
        assert!(failure.contains("valid options are []"));
    }

    #[test]
    fn registry_unknown_backend_error_lists_registered_ids() {
        let mut config = MemorySystemConfig::default();
        config.backend = "not_registered".to_owned();
        let failure = MemoryRegistry::new()
            .build(&config)
            .err()
            .unwrap()
            .to_string();
        assert!(failure.contains("mempalace_stoolap"));
        assert!(failure.contains("naive_memory_system"));
        assert!(failure.contains("noop"));
    }

    #[test]
    fn registry_custom_backend_validator_rejects_invalid_options() {
        let mut registry = MemoryRegistry::new();
        registry
            .register_with_validator(
                "strict_custom",
                |_| Ok(Arc::new(NoopMemoryBackend)),
                |config| super::validate_backend_options(config, &["endpoint"]),
            )
            .unwrap();
        let mut config = MemorySystemConfig::default();
        config.backend = "strict_custom".to_owned();
        config
            .backend_options
            .insert("endpont".to_owned(), "local".to_owned());
        let failure = registry.build(&config).err().unwrap().to_string();
        assert!(failure.contains("endpont"));
        assert!(failure.contains("endpoint"));
    }

    #[test]
    fn config_selects_a_globally_registered_custom_backend() {
        super::register_memory_backend("test_custom_backend", |_| Ok(Arc::new(NoopMemoryBackend)))
            .unwrap();
        let mut config = MemorySystemConfig::default();
        config.backend = "test_custom_backend".to_owned();
        assert_eq!(
            super::build_memory_from_config(&config)
                .unwrap()
                .backend_name(),
            "noop"
        );
    }

    #[tokio::test]
    async fn noop_backend_is_available_without_storage() {
        let backend = NoopMemoryBackend;
        assert!(
            backend
                .recall(MemoryQuery::default())
                .await
                .unwrap()
                .records
                .is_empty()
        );
        assert!(backend.health_check().await.is_ok());
    }
}
