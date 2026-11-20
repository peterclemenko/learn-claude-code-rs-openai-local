use anyhow::{Context, Result};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{Resource, trace::SdkTracerProvider};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub(crate) struct TelemetryProvider {
    provider: Option<SdkTracerProvider>,
}

impl TelemetryProvider {
    pub(crate) fn shutdown(&self) -> Result<()> {
        if let Some(provider) = &self.provider {
            provider
                .shutdown()
                .context("Failed to flush OpenTelemetry spans")?;
        }
        Ok(())
    }
}

pub(crate) fn init_telemetry(
    enabled: bool,
    service_name: &str,
    fallback_endpoint: &str,
) -> Result<TelemetryProvider> {
    let telemetry_enabled = telemetry_enabled(enabled)?;
    if !telemetry_enabled {
        tracing_subscriber::registry()
            .try_init()
            .context("Failed to initialize OpenTelemetry-only tracing subscriber")?;
        return Ok(TelemetryProvider { provider: None });
    }

    let endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
        .unwrap_or_else(|_| fallback_endpoint.to_owned());
    let service_name =
        std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| service_name.to_owned());
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()
        .context("Failed to configure OTLP span exporter")?;
    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(
            Resource::builder_empty()
                .with_service_name(service_name.clone())
                .build(),
        )
        .build();
    let tracer = provider.tracer(service_name);
    tracing_subscriber::registry()
        .with(tracing_opentelemetry::layer().with_tracer(tracer))
        .try_init()
        .context("Failed to initialize OpenTelemetry tracing subscriber")?;
    Ok(TelemetryProvider {
        provider: Some(provider),
    })
}

fn telemetry_enabled(config_enabled: bool) -> Result<bool> {
    match std::env::var("OTEL_SDK_DISABLED") {
        Ok(value) => resolve_telemetry_enabled(config_enabled, Some(&value)),
        Err(std::env::VarError::NotPresent) => resolve_telemetry_enabled(config_enabled, None),
        Err(error) => Err(error).context("Failed to read OTEL_SDK_DISABLED"),
    }
}

fn resolve_telemetry_enabled(
    config_enabled: bool,
    otel_sdk_disabled: Option<&str>,
) -> Result<bool> {
    match otel_sdk_disabled {
        Some(value) => Ok(!parse_otel_sdk_disabled(value)?),
        None => Ok(config_enabled),
    }
}

fn parse_otel_sdk_disabled(value: &str) -> Result<bool> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" => Ok(false),
        "0" | "false" | "no" => Ok(true),
        _ => anyhow::bail!(
            "OTEL_SDK_DISABLED must be one of 1/0, true/false, or yes/no; got '{value}'"
        ),
    }
}
