use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, ensure};
use async_trait::async_trait;
use opentelemetry::{Value, trace::TracerProvider as _};
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider, SpanData};
use tokio::task::JoinHandle;
use tracing::{info, info_span};
use tracing_subscriber::prelude::*;

use super::{
    MemPalaceStoolapMemory, MemoryBackend, MemoryCategory, MemoryEntry, MemoryQuery, MemoryRecord,
    MemoryRegistry, NaiveMemorySystem, config::MemorySystemConfig,
};

static NEXT_TEMP_DIRECTORY: AtomicU64 = AtomicU64::new(1);

struct TestDirectory {
    path: PathBuf,
}

impl TestDirectory {
    fn new(label: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "memory-{label}-{}-{nonce}-{}",
            std::process::id(),
            NEXT_TEMP_DIRECTORY.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).expect("create isolated memory test directory");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn write_config(&self, backend: &str, storage_dir: PathBuf) -> PathBuf {
        let config = MemorySystemConfig {
            backend: backend.to_owned(),
            storage_dir,
            ..MemorySystemConfig::default()
        };
        let path = self.path.join(format!("memory-{backend}.ron"));
        let contents = ron::ser::to_string(&config).expect("serialize test memory config");
        fs::write(&path, contents).expect("write isolated memory config");
        path
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        if let Err(failure) = fs::remove_dir_all(&self.path) {
            tracing::warn!(
                path = %self.path.display(),
                error = %failure,
                "Failed to clean up isolated memory test directory"
            );
        }
    }
}

struct OtelCapture {
    exporter: InMemorySpanExporter,
    provider: SdkTracerProvider,
    subscriber_guard: tracing::subscriber::DefaultGuard,
}

impl OtelCapture {
    fn new() -> Self {
        let exporter = InMemorySpanExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        let subscriber = tracing_subscriber::registry().with(
            tracing_opentelemetry::layer().with_tracer(provider.tracer("memory-conformance")),
        );
        let subscriber_guard = tracing::subscriber::set_default(subscriber);
        Self {
            exporter,
            provider,
            subscriber_guard,
        }
    }

    fn finish(self) -> Vec<SpanData> {
        self.provider.force_flush().expect("flush in-memory spans");
        let spans = self
            .exporter
            .get_finished_spans()
            .expect("read in-memory spans");
        drop(self.subscriber_guard);
        self.provider.shutdown().expect("shutdown test tracer");
        assert!(self.exporter.is_shutdown_called());
        spans
    }
}

struct RegisteredMemoryBackend {
    inner: NaiveMemorySystem,
}

impl RegisteredMemoryBackend {
    fn open(config: &MemorySystemConfig) -> Result<Self> {
        Ok(Self {
            inner: NaiveMemorySystem::open(config)?,
        })
    }
}

#[async_trait]
impl MemoryBackend for RegisteredMemoryBackend {
    fn backend_name(&self) -> &str {
        "test_custom"
    }

    async fn recall(&self, query: MemoryQuery) -> super::MemoryResult<super::MemoryRecallResult> {
        self.inner.recall(query).await
    }

    async fn remember(&self, entry: MemoryEntry) -> super::MemoryResult<String> {
        self.inner.remember(entry).await
    }

    async fn forget(&self, id: &str) -> super::MemoryResult<bool> {
        self.inner.forget(id).await
    }

    async fn list(&self, limit: usize) -> super::MemoryResult<Vec<MemoryRecord>> {
        self.inner.list(limit).await
    }

    async fn health_check(&self) -> super::MemoryResult<()> {
        self.inner.health_check().await
    }
}

fn custom_registry() -> MemoryRegistry {
    let mut registry = MemoryRegistry::new();
    registry
        .register("test_custom", |config| {
            Ok(Arc::new(RegisteredMemoryBackend::open(config)?))
        })
        .expect("register test backend");
    registry
}

fn category_cases() -> [(MemoryCategory, &'static str); 4] {
    [
        (MemoryCategory::Preference, "markerpreferencetag"),
        (MemoryCategory::Fact, "markerfacttag"),
        (MemoryCategory::Feedback, "markerfeedbacktag"),
        (MemoryCategory::Reference, "markerreferencetag"),
    ]
}

fn entry(category: MemoryCategory, marker: &str) -> MemoryEntry {
    MemoryEntry {
        category,
        content: format!("contract-shared memory {}", category.as_str()),
        tags: vec!["contract-shared".to_owned(), marker.to_owned()],
        scope: Some("contract-scope".to_owned()),
        metadata: [
            ("source".to_owned(), "conformance".to_owned()),
            (
                "category".to_owned(),
                category.as_str().to_ascii_lowercase(),
            ),
        ]
        .into(),
    }
}

pub(crate) async fn run_backend_conformance_suite(
    backend: Arc<dyn MemoryBackend>,
) -> Result<(String, String)> {
    backend.health_check().await?;
    let empty = backend
        .recall(MemoryQuery {
            text: "contract-shared".to_owned(),
            ..Default::default()
        })
        .await?;
    ensure!(
        empty.records.is_empty(),
        "new backend unexpectedly contains data"
    );
    ensure!(
        backend
            .format_for_context(&empty)
            .await?
            .contains("memories[0]"),
        "empty recall should format as an empty TOON table"
    );

    let cases = category_cases();
    let mut stored = Vec::with_capacity(cases.len());
    for (category, marker) in cases {
        let record = entry(category, marker);
        let id = backend.remember(record.clone()).await?;
        stored.push((id, record));
    }

    let listed = backend.list(20).await?;
    ensure!(listed.len() == 4, "expected all durable categories in list");
    for (id, expected) in &stored {
        let record = listed
            .iter()
            .find(|record| &record.id == id)
            .context("remembered record is missing from list")?;
        ensure!(
            record.entry == *expected,
            "entry metadata was not preserved"
        );
    }

    for (category, marker) in cases {
        let recalled = backend
            .recall(MemoryQuery {
                text: marker.to_owned(),
                categories: vec![category],
                scope: Some("contract-scope".to_owned()),
                limit: Some(1),
            })
            .await?;
        ensure!(
            recalled.records.len() == 1,
            "tag/category filter did not match one record"
        );
        ensure!(
            recalled.records[0].entry.category == category,
            "category filter returned the wrong category"
        );
        ensure!(
            recalled.records[0].entry.tags.contains(&marker.to_owned()),
            "tag marker was not searchable"
        );
    }

    let limited = backend
        .recall(MemoryQuery {
            text: "contract-shared".to_owned(),
            limit: Some(2),
            ..Default::default()
        })
        .await?;
    ensure!(limited.records.len() == 2, "query limit was not enforced");
    let wrong_scope = backend
        .recall(MemoryQuery {
            text: "contract-shared".to_owned(),
            scope: Some("different-scope".to_owned()),
            ..Default::default()
        })
        .await?;
    ensure!(wrong_scope.records.is_empty(), "scope filter was ignored");

    let populated_toon = backend.format_for_context(&limited).await?;
    ensure!(
        populated_toon.contains("memories[2]"),
        "populated recall should format as a TOON table"
    );
    ensure!(
        populated_toon.contains("contract-shared"),
        "TOON output should retain entry content"
    );

    let large_content = format!("largepayloadmarker:{}", "x".repeat(8 * 1024));
    let large_id = backend
        .remember(MemoryEntry {
            category: MemoryCategory::Reference,
            content: large_content.clone(),
            tags: vec!["large-payload".to_owned()],
            scope: Some("contract-scope".to_owned()),
            metadata: [("size".to_owned(), "8192".to_owned())].into(),
        })
        .await?;
    let large_recall = backend
        .recall(MemoryQuery {
            text: "largepayloadmarker".to_owned(),
            limit: Some(1),
            ..Default::default()
        })
        .await?;
    ensure!(
        large_recall.records[0].entry.content == large_content,
        "multi-kilobyte entry did not round-trip"
    );

    let removed_id = &stored[0].0;
    ensure!(
        backend.forget(removed_id).await?,
        "forget should remove a hit"
    );
    ensure!(
        !backend.forget("missing-memory-id").await?,
        "forget miss should return false"
    );
    ensure!(
        !backend
            .list(20)
            .await?
            .iter()
            .any(|record| &record.id == removed_id)
    );
    ensure!(
        backend
            .list(20)
            .await?
            .iter()
            .any(|record| record.id == large_id)
    );
    backend.health_check().await?;

    let (rehydrated_id, rehydrated_entry) = stored[1].clone();
    let rehydrated_content = rehydrated_entry.content;
    Ok((rehydrated_id, rehydrated_content))
}

async fn assert_rehydrated(backend: &dyn MemoryBackend, id: &str, content: &str) -> Result<()> {
    let listed = backend.list(100).await?;
    let record = listed
        .iter()
        .find(|record| record.id == id)
        .context("durable record was not rehydrated")?;
    ensure!(
        record.entry.content == content,
        "rehydrated content changed"
    );
    Ok(())
}

fn attribute<'a>(span: &'a SpanData, name: &str) -> Option<&'a Value> {
    span.attributes
        .iter()
        .find(|attribute| attribute.key.as_str() == name)
        .map(|attribute| &attribute.value)
}

fn assert_string_attribute(span: &SpanData, name: &str, expected: &str) {
    assert!(matches!(
        attribute(span, name),
        Some(Value::String(value)) if value.as_str() == expected
    ));
}

fn record_benchmark(operation: &str, corpus_size: usize, operations: usize, elapsed: Duration) {
    let duration_ms = elapsed.as_millis().try_into().unwrap_or(u64::MAX);
    let operations_per_second = operations as f64 / elapsed.as_secs_f64().max(f64::EPSILON);
    let span = info_span!(
        "memory.benchmark.operation",
        memory.backend = "naive_memory_system",
        memory.operation = operation,
        memory.entry_count = corpus_size,
        memory.duration_ms = tracing::field::Empty,
        benchmark.operations_per_second = tracing::field::Empty
    );
    span.record("memory.duration_ms", duration_ms);
    span.record("benchmark.operations_per_second", operations_per_second);
    info!(
        parent: &span,
        memory.operation = operation,
        memory.entry_count = corpus_size,
        memory.duration_ms = duration_ms,
        benchmark.operations = operations,
        benchmark.operations_per_second = operations_per_second,
        "Memory benchmark measurement"
    );
}

#[tokio::test]
async fn memory_backend_conformance_suite_passes_for_builtin_and_custom_backends() -> Result<()> {
    let _subscriber_lock = crate::test_telemetry::lock_test_subscriber();
    let naive_dir = TestDirectory::new("naive-conformance");
    let naive_config_path =
        naive_dir.write_config("naive_memory_system", naive_dir.path().join("naive-store"));
    let naive_config = MemorySystemConfig::load(naive_config_path)?;
    let naive = MemoryRegistry::new().build(&naive_config)?;
    ensure!(naive.backend_name() == "naive_memory_system");
    let (naive_id, naive_content) = run_backend_conformance_suite(Arc::clone(&naive)).await?;
    drop(naive);
    let reopened_naive = NaiveMemorySystem::open(&naive_config)?;
    assert_rehydrated(&reopened_naive, &naive_id, &naive_content).await?;

    let palace_dir = TestDirectory::new("mempalace-conformance");
    let palace_config_path =
        palace_dir.write_config("mempalace_stoolap", palace_dir.path().join("palace-store"));
    let palace_config = MemorySystemConfig::load(palace_config_path)?;
    let palace = MemoryRegistry::new().build(&palace_config)?;
    ensure!(palace.backend_name() == "mempalace_stoolap");
    let (palace_id, palace_content) = run_backend_conformance_suite(Arc::clone(&palace)).await?;
    drop(palace);
    let reopened_palace = MemPalaceStoolapMemory::open(&palace_config)?;
    assert_rehydrated(&reopened_palace, &palace_id, &palace_content).await?;

    let custom_dir = TestDirectory::new("custom-conformance");
    let custom_config_path =
        custom_dir.write_config("test_custom", custom_dir.path().join("custom-store"));
    let custom_config = MemorySystemConfig::load(custom_config_path)?;
    let custom = custom_registry().build(&custom_config)?;
    ensure!(custom.backend_name() == "test_custom");
    let (custom_id, custom_content) = run_backend_conformance_suite(Arc::clone(&custom)).await?;
    drop(custom);
    let reopened_custom = RegisteredMemoryBackend::open(&custom_config)?;
    assert_rehydrated(&reopened_custom, &custom_id, &custom_content).await?;
    Ok(())
}

#[tokio::test]
async fn memory_config_selects_naive_noop_and_custom_backends() -> Result<()> {
    let _subscriber_lock = crate::test_telemetry::lock_test_subscriber();
    let directory = TestDirectory::new("backend-selection");
    let registry = custom_registry();
    for backend_name in [
        "naive_memory_system",
        "mempalace_stoolap",
        "noop",
        "test_custom",
    ] {
        let config_path = directory.write_config(
            backend_name,
            directory.path().join(format!("{backend_name}-store")),
        );
        let config = MemorySystemConfig::load(config_path)?;
        let backend = registry.build(&config)?;
        ensure!(backend.backend_name() == backend_name);
        backend.health_check().await?;
        if backend_name == "noop" {
            let id = backend
                .remember(entry(MemoryCategory::Fact, "noop-marker"))
                .await?;
            ensure!(id == "noop");
            ensure!(backend.list(10).await?.is_empty());
        }
    }
    Ok(())
}

#[tokio::test]
async fn memory_backend_concurrent_readers_and_writers_complete_without_errors() -> Result<()> {
    let _subscriber_lock = crate::test_telemetry::lock_test_subscriber();
    let directory = TestDirectory::new("concurrency");
    let config = MemorySystemConfig {
        storage_dir: directory.path().join("store"),
        ..MemorySystemConfig::default()
    };
    let backend = MemoryRegistry::new().build(&config)?;
    let mut writers: Vec<JoinHandle<Result<()>>> = Vec::new();
    let mut readers: Vec<JoinHandle<Result<()>>> = Vec::new();

    for writer in 0..4 {
        let backend = Arc::clone(&backend);
        writers.push(tokio::spawn(async move {
            for item in 0..10 {
                backend
                    .remember(MemoryEntry {
                        category: MemoryCategory::Fact,
                        content: format!("concurrency shared writer {writer} item {item}"),
                        tags: vec!["concurrency-shared".to_owned()],
                        scope: Some("concurrency".to_owned()),
                        metadata: Default::default(),
                    })
                    .await?;
            }
            Ok(())
        }));
    }
    for _ in 0..4 {
        let backend = Arc::clone(&backend);
        readers.push(tokio::spawn(async move {
            for _ in 0..20 {
                backend
                    .recall(MemoryQuery {
                        text: "concurrency-shared".to_owned(),
                        limit: Some(10),
                        ..Default::default()
                    })
                    .await?;
                backend.list(20).await?;
            }
            Ok(())
        }));
    }

    for writer in writers {
        writer.await.context("writer task panicked")??;
    }
    for reader in readers {
        reader.await.context("reader task panicked")??;
    }
    let records = backend.list(100).await?;
    ensure!(records.len() == 40, "concurrent writes were lost");
    let ids = records
        .iter()
        .map(|record| &record.id)
        .collect::<BTreeSet<_>>();
    ensure!(ids.len() == records.len(), "concurrent writes reused an ID");
    backend.health_check().await?;
    Ok(())
}

#[test]
fn memory_corrupt_store_returns_an_error_without_panicking() -> Result<()> {
    let _subscriber_lock = crate::test_telemetry::lock_test_subscriber();
    let directory = TestDirectory::new("corrupt-store");
    let storage = directory.path().join("store");
    fs::create_dir_all(&storage)?;
    fs::write(storage.join("memories.ron"), "this is not valid RON")?;
    let config = MemorySystemConfig {
        storage_dir: storage,
        ..MemorySystemConfig::default()
    };
    ensure!(
        NaiveMemorySystem::open(&config).is_err(),
        "corrupted memory storage should return a MemoryError"
    );
    Ok(())
}

#[test]
fn otel_memory_config_registry_and_backend_operations_export_attributes() -> Result<()> {
    let _subscriber_lock = crate::test_telemetry::lock_test_subscriber();
    let telemetry = OtelCapture::new();
    let mut invalid_config = MemorySystemConfig::default();
    invalid_config.recall_limit = 0;
    ensure!(invalid_config.validate().is_err());

    let directory = TestDirectory::new("otel-conformance");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let config_path =
            directory.write_config("naive_memory_system", directory.path().join("store"));
        let config = MemorySystemConfig::load(config_path)?;
        let registry = MemoryRegistry::new();
        let backend = registry.build(&config)?;
        let mut invalid_options = config.clone();
        invalid_options
            .backend_options
            .insert("file_nam".to_owned(), "memories.ron".to_owned());
        ensure!(
            registry
                .build(&invalid_options)
                .err()
                .is_some_and(|failure| failure.to_string().contains("file_name")),
            "backend option validation should report valid options"
        );
        let id = backend
            .remember(entry(MemoryCategory::Preference, "otel-memory-marker"))
            .await?;
        let recall = backend
            .recall(MemoryQuery {
                text: "otel-memory-marker".to_owned(),
                ..Default::default()
            })
            .await?;
        backend.format_for_context(&recall).await?;
        backend.list(10).await?;
        backend.forget(&id).await?;
        backend.health_check().await?;
        Ok::<(), anyhow::Error>(())
    })?;
    drop(runtime);

    let spans = telemetry.finish();
    let config_span = spans
        .iter()
        .find(|span| span.name.as_ref() == "memory.config.load")
        .with_context(|| {
            format!(
                "missing memory config load span; exported spans: {:?}",
                spans
                    .iter()
                    .map(|span| span.name.as_ref())
                    .collect::<Vec<_>>()
            )
        })?;
    assert_string_attribute(config_span, "memory.backend", "naive_memory_system");
    assert_string_attribute(config_span, "memory.outcome", "ok");

    let validation_span = spans
        .iter()
        .find(|span| {
            span.name.as_ref() == "memory.config.validate"
                && matches!(
                    attribute(span, "memory.outcome"),
                    Some(Value::String(value)) if value.as_str() == "error"
                )
        })
        .context("missing failed memory config validation span")?;
    assert_string_attribute(validation_span, "memory.backend", "naive_memory_system");

    let registry_span = spans
        .iter()
        .find(|span| span.name.as_ref() == "memory.backend.build")
        .context("missing memory backend build span")?;
    assert_string_attribute(registry_span, "memory.backend", "naive_memory_system");
    assert_string_attribute(registry_span, "memory.outcome", "ok");

    let failed_registry_span = spans
        .iter()
        .find(|span| {
            span.name.as_ref() == "memory.backend.build"
                && matches!(
                    attribute(span, "memory.outcome"),
                    Some(Value::String(value)) if value.as_str() == "error"
                )
        })
        .context("missing failed memory backend validation span")?;
    assert_string_attribute(
        failed_registry_span,
        "memory.backend",
        "naive_memory_system",
    );

    for operation in [
        "remember",
        "recall",
        "format_for_context",
        "list",
        "forget",
        "health_check",
    ] {
        let span = spans
            .iter()
            .find(|span| {
                matches!(
                    attribute(span, "memory.operation"),
                    Some(Value::String(value)) if value.as_str() == operation
                )
            })
            .with_context(|| format!("missing span for memory operation '{operation}'"))?;
        assert_string_attribute(span, "memory.backend", "naive_memory_system");
        assert_string_attribute(span, "memory.outcome", "ok");
        ensure!(attribute(span, "memory.duration_ms").is_some());
        ensure!(attribute(span, "memory.entry_count").is_some());
    }
    Ok(())
}

#[tokio::test]
#[ignore = "manual performance benchmark; emits measurements as OTel span attributes"]
async fn memory_benchmark_operation_latency_and_corpus_scaling() -> Result<()> {
    let _subscriber_lock = crate::test_telemetry::lock_test_subscriber();
    let telemetry = OtelCapture::new();
    let directory = TestDirectory::new("benchmark");
    for batch_size in [100, 1_000, 5_000] {
        let batch_config = MemorySystemConfig {
            storage_dir: directory.path().join(format!("batch-{batch_size}")),
            recall_limit: 20,
            ..MemorySystemConfig::default()
        };
        let backend = MemoryRegistry::new().build(&batch_config)?;
        let mut inserted = Vec::with_capacity(batch_size);
        let mut write_duration = Duration::ZERO;

        for index in 0..batch_size {
            let started = Instant::now();
            let category = match index % 4 {
                0 => MemoryCategory::Preference,
                1 => MemoryCategory::Fact,
                2 => MemoryCategory::Feedback,
                _ => MemoryCategory::Reference,
            };
            inserted.push(
                backend
                    .remember(MemoryEntry {
                        category,
                        content: format!("shared-corpus marker{index} indexed memory"),
                        tags: vec![
                            "shared-fanout-tag".to_owned(),
                            format!("unique-tag-{index}"),
                        ],
                        scope: Some("benchmark".to_owned()),
                        metadata: [("index".to_owned(), index.to_string())].into(),
                    })
                    .await?,
            );
            let corpus_size = index + 1;
            write_duration += started.elapsed();
            if batch_size == 5_000 && [50, 500, 5_000].contains(&corpus_size) {
                record_benchmark("remember_at_corpus", corpus_size, 1, started.elapsed());

                let started = Instant::now();
                let cold = backend
                    .recall(MemoryQuery {
                        text: "shared-fanout-tag".to_owned(),
                        limit: Some(20),
                        ..Default::default()
                    })
                    .await?;
                record_benchmark("recall_first", corpus_size, 1, started.elapsed());
                ensure!(!cold.records.is_empty(), "recall should match shared tags");

                let started = Instant::now();
                for _ in 0..5 {
                    backend
                        .recall(MemoryQuery {
                            text: "shared-fanout-tag".to_owned(),
                            limit: Some(20),
                            ..Default::default()
                        })
                        .await?;
                }
                record_benchmark("recall_warm", corpus_size, 5, started.elapsed());

                let started = Instant::now();
                backend.list(10).await?;
                record_benchmark("list_small", corpus_size, 1, started.elapsed());
                let started = Instant::now();
                backend.list(corpus_size).await?;
                record_benchmark("list_large", corpus_size, 1, started.elapsed());
            }
        }
        record_benchmark("remember_batch", batch_size, batch_size, write_duration);

        if batch_size == 5_000 {
            let started = Instant::now();
            ensure!(
                backend.forget(&inserted[inserted.len() - 1]).await?,
                "forget should remove the last inserted record"
            );
            record_benchmark("forget", batch_size, 1, started.elapsed());
        }
    }

    let spans = telemetry.finish();
    for operation in [
        "remember_batch",
        "remember_at_corpus",
        "recall_first",
        "recall_warm",
        "list_small",
        "list_large",
        "forget",
    ] {
        ensure!(
            spans.iter().any(|span| {
                span.name.as_ref() == "memory.benchmark.operation"
                    && matches!(
                        attribute(span, "memory.operation"),
                        Some(Value::String(value)) if value.as_str() == operation
                    )
            }),
            "benchmark telemetry is missing operation '{operation}'"
        );
    }
    Ok(())
}
