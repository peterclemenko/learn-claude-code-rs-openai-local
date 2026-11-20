use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock, RwLock},
};

use anyhow::{Context, Result};
use tracing::{error, info, instrument};

use super::{
    MemoryBackend, MemoryResult, NaiveMemorySystem, NoopMemoryBackend, config::MemorySystemConfig,
    observed_backend,
};

type BackendBuilder =
    Arc<dyn Fn(&MemorySystemConfig) -> MemoryResult<Arc<dyn MemoryBackend>> + Send + Sync>;

/// Maps backend identifiers to constructors for pluggable implementations.
#[derive(Clone, Default)]
pub struct MemoryRegistry {
    builders: BTreeMap<String, BackendBuilder>,
}

static DEFAULT_REGISTRY: OnceLock<RwLock<MemoryRegistry>> = OnceLock::new();

fn default_registry() -> &'static RwLock<MemoryRegistry> {
    DEFAULT_REGISTRY.get_or_init(|| RwLock::new(MemoryRegistry::new()))
}

impl MemoryRegistry {
    /// Creates a registry containing the naive memory system and noop backend.
    pub fn new() -> Self {
        let mut registry = Self::default();
        registry.builders.insert(
            "naivememorysystem".to_owned(),
            Arc::new(|config| Ok(Arc::new(NaiveMemorySystem::open(config)?))),
        );
        registry.builders.insert(
            "noop".to_owned(),
            Arc::new(|_| Ok(Arc::new(NoopMemoryBackend))),
        );
        registry
    }

    /// Registers a custom backend constructor under a unique identifier.
    /// Registers a custom backend constructor under a unique identifier.
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
        let name = name.into();
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
        self.builders.insert(name, Arc::new(builder));
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
        if let Err(failure) = config.validate() {
            tracing::Span::current().record("memory.outcome", "error");
            error!(error = %failure, "Memory backend configuration is invalid");
            return Err(failure);
        }
        let Some(builder) = self.builders.get(&config.backend) else {
            let failure = anyhow::anyhow!("Unknown memory backend '{}'", config.backend);
            tracing::Span::current().record("memory.outcome", "error");
            error!(memory.backend = %config.backend, error = %failure, "Memory backend is not registered");
            return Err(failure);
        };
        let backend = match builder(config)
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
    let mut registry = match default_registry().write() {
        Ok(registry) => registry,
        Err(failure) => {
            let failure = anyhow::anyhow!("Memory registry lock is poisoned: {failure}");
            tracing::Span::current().record("memory.outcome", "error");
            error!(error = %failure, "Memory backend registration failed");
            return Err(failure);
        }
    };
    registry.register(name, builder)
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

        config.backend = "naivememorysystem".to_owned();
        config.storage_dir = std::env::temp_dir().join(format!(
            "memory-registry-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let naive = MemoryRegistry::new().build(&config).unwrap();
        assert_eq!(naive.backend_name(), "naivememorysystem");
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
