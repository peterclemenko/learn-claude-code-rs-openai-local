use std::{
    env,
    future::Future,
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    path::Path,
    sync::OnceLock,
    time::Instant,
};

use anyhow::{Context, Result};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{Resource, trace::SdkTracerProvider};
use tracing::{Instrument, info, info_span};
use tracing_subscriber::{Layer, layer::SubscriberExt, util::SubscriberInitExt};

struct TestTelemetry {
    provider: Option<SdkTracerProvider>,
}

static TEST_TELEMETRY: OnceLock<std::result::Result<TestTelemetry, String>> = OnceLock::new();
static TEST_RUNTIME: OnceLock<std::result::Result<tokio::runtime::Runtime, String>> =
    OnceLock::new();

pub(crate) fn run_test_case(name: &'static str, test: impl FnOnce()) {
    let telemetry = test_telemetry();
    let span = test_span(name);
    let started = Instant::now();
    let result = catch_unwind(AssertUnwindSafe(|| {
        span.in_scope(|| {
            info!(test.name = name, "Test started");
            test();
        });
    }));

    record_test_result(telemetry, &span, name, started.elapsed(), result.is_ok());
    if let Err(panic) = result {
        resume_unwind(panic);
    }
}

pub(crate) async fn run_async_test_case<F>(name: &'static str, test: F)
where
    F: Future<Output = ()> + Send + 'static,
{
    let telemetry = test_telemetry();
    let span = test_span(name);
    let started = Instant::now();
    let result = tokio::spawn(test.instrument(span.clone())).await;

    record_test_result(telemetry, &span, name, started.elapsed(), result.is_ok());
    match result {
        Ok(()) => {}
        Err(error) if error.is_panic() => resume_unwind(error.into_panic()),
        Err(error) => panic!("Test task was cancelled: {error}"),
    }
}

fn test_span(name: &'static str) -> tracing::Span {
    info_span!(
        "unit_test",
        test.name = name,
        test.status = tracing::field::Empty,
        test.duration_ms = tracing::field::Empty,
        test.failure = tracing::field::Empty,
    )
}

fn record_test_result(
    telemetry: &TestTelemetry,
    span: &tracing::Span,
    name: &'static str,
    duration: std::time::Duration,
    passed: bool,
) {
    let status = if passed { "passed" } else { "failed" };
    span.record("test.status", status);
    span.record("test.duration_ms", duration.as_millis() as u64);
    if !passed {
        span.record("test.failure", "test panicked");
    }
    span.in_scope(|| {
        info!(
            test.name = name,
            test.status = status,
            test.duration_ms = duration.as_millis() as u64,
            "Test completed"
        );
    });

    if let Some(provider) = &telemetry.provider
        && let Err(error) = provider.force_flush()
    {
        eprintln!("Failed to export test telemetry for '{name}': {error}");
    }
}

fn test_telemetry() -> &'static TestTelemetry {
    TEST_TELEMETRY
        .get_or_init(|| initialize_test_tracing().map_err(|error| format!("{error:#}")))
        .as_ref()
        .unwrap_or_else(|error| panic!("Failed to initialize test telemetry: {error}"))
}

fn initialize_test_tracing() -> Result<TestTelemetry> {
    let format_layer = tracing_subscriber::fmt::layer()
        .with_test_writer()
        .with_filter(tracing_subscriber::filter::LevelFilter::INFO);
    if should_export_telemetry()? {
        let runtime = test_runtime()?;
        let _runtime_guard = runtime.enter();
        let project_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .context("Package manifest directory has no workspace parent")?;
        let config = crate::types::AgentRonConfig::load(project_root.join("config/agent.ron"))?;
        let endpoint = match env::var("OTEL_EXPORTER_OTLP_ENDPOINT") {
            Ok(endpoint) => endpoint,
            Err(env::VarError::NotPresent) => config.telemetry.otlp_endpoint.clone(),
            Err(error) => {
                return Err(error).context("Failed to read OTEL_EXPORTER_OTLP_ENDPOINT");
            }
        };
        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(&endpoint)
            .build()
            .context("Failed to configure test OTLP span exporter")?;
        let service_name = format!("{}-tests", config.telemetry.service_name);
        let provider = SdkTracerProvider::builder()
            .with_batch_exporter(exporter)
            .with_resource(
                Resource::builder_empty()
                    .with_service_name(service_name.clone())
                    .build(),
            )
            .build();
        tracing_subscriber::registry()
            .with(format_layer)
            .with(tracing_opentelemetry::layer().with_tracer(provider.tracer(service_name)))
            .try_init()
            .context("Failed to initialize test tracing subscriber")?;
        Ok(TestTelemetry {
            provider: Some(provider),
        })
    } else {
        tracing_subscriber::registry()
            .with(format_layer)
            .try_init()
            .context("Failed to initialize test tracing subscriber")?;
        Ok(TestTelemetry { provider: None })
    }
}

fn should_export_telemetry() -> Result<bool> {
    let disabled = match env::var("OTEL_SDK_DISABLED") {
        Ok(value) => Some(value),
        Err(env::VarError::NotPresent) => None,
        Err(error) => return Err(error).context("Failed to read OTEL_SDK_DISABLED"),
    };
    telemetry_export_enabled(
        disabled.as_deref(),
        env::var_os("OTEL_EXPORTER_OTLP_ENDPOINT").is_some(),
    )
}

fn telemetry_export_enabled(disabled: Option<&str>, has_endpoint: bool) -> Result<bool> {
    match disabled {
        Some(value) => Ok(!parse_disabled_value(value)?),
        None => Ok(has_endpoint),
    }
}

fn parse_disabled_value(value: &str) -> Result<bool> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" => Ok(true),
        "0" | "false" | "no" => Ok(false),
        _ => anyhow::bail!(
            "OTEL_SDK_DISABLED must be one of 1/0, true/false, or yes/no; got '{value}'"
        ),
    }
}

fn test_runtime() -> Result<&'static tokio::runtime::Runtime> {
    TEST_RUNTIME
        .get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(|error| format!("{error:#}"))
        })
        .as_ref()
        .map_err(|error| anyhow::anyhow!("Failed to create test Tokio runtime: {error}"))
}

#[cfg(test)]
mod tests {
    use super::{parse_disabled_value, telemetry_export_enabled};

    #[test]
    fn test_telemetry_disable_values_are_explicit_and_unambiguous() {
        assert!(parse_disabled_value("true").unwrap());
        assert!(parse_disabled_value("YES").unwrap());
        assert!(parse_disabled_value("1").unwrap());
        assert!(!parse_disabled_value("false").unwrap());
        assert!(!parse_disabled_value("no").unwrap());
        assert!(!parse_disabled_value("0").unwrap());
        assert!(parse_disabled_value("maybe").is_err());
    }

    #[test]
    fn telemetry_export_defaults_to_disabled_unless_an_endpoint_is_configured() {
        assert!(!telemetry_export_enabled(None, false).unwrap());
        assert!(telemetry_export_enabled(None, true).unwrap());
    }

    #[test]
    fn telemetry_disabled_environment_override_takes_precedence() {
        assert!(!telemetry_export_enabled(Some("true"), true).unwrap());
        assert!(telemetry_export_enabled(Some("false"), false).unwrap());
    }
}
