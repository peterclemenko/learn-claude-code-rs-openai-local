mod agent;
mod context_window;
mod subagent;
mod tools;
mod types;

use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::{Context, Result};
use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestUserMessageArgs,
};
use clap::Parser;
use inquire::Text;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{Resource, trace::SdkTracerProvider};
use tracing_subscriber::{Layer, layer::SubscriberExt, util::SubscriberInitExt};

use crate::{
    agent::Agent,
    types::{AgentRonConfig, TodoState},
};

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    /// Path to the RON agent configuration
    #[arg(long, default_value = "config/agent.ron")]
    config: PathBuf,

    /// Prompt to send without starting the interactive session
    #[arg(short, long)]
    prompt: Option<String>,

    /// Enable debug-level logs
    #[arg(short, long)]
    verbose: bool,
}

fn init_tracing(config: &AgentRonConfig, verbose: bool) -> Result<Option<SdkTracerProvider>> {
    let telemetry_enabled = telemetry_enabled(config.telemetry.enabled)?;
    let level = if verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };
    let format_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_filter(tracing_subscriber::filter::LevelFilter::from_level(level));

    if !telemetry_enabled {
        tracing_subscriber::registry()
            .with(format_layer)
            .try_init()
            .context("Failed to initialize tracing subscriber")?;
        return Ok(None);
    }

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&config.telemetry.otlp_endpoint)
        .build()
        .context("Failed to configure OTLP span exporter")?;
    let service_name = config.telemetry.service_name.clone();
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
        .with(format_layer)
        .with(tracing_opentelemetry::layer().with_tracer(tracer))
        .try_init()
        .context("Failed to initialize tracing and OpenTelemetry subscribers")?;

    Ok(Some(provider))
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
    value
        .to_ascii_lowercase()
        .parse::<bool>()
        .with_context(|| format!("OTEL_SDK_DISABLED must be 'true' or 'false', got '{value}'"))
}

async fn run_prompt(agent: &Agent, prompt: &str) -> Result<()> {
    let answer = agent.run_prompt(prompt).await?;
    println!("{answer}");
    Ok(())
}

async fn run_repl(agent: &Agent, config: &AgentRonConfig) -> Result<()> {
    let mut messages: Vec<ChatCompletionRequestMessage> = vec![
        ChatCompletionRequestSystemMessageArgs::default()
            .content(config.system_prompt.as_str())
            .build()?
            .into(),
    ];

    loop {
        let prompt = Text::new("--- How can I help you?")
            .prompt()
            .context("Failed to read prompt")?;
        if prompt.trim() == "exit()" {
            break;
        }

        messages.push(
            ChatCompletionRequestUserMessageArgs::default()
                .content(prompt)
                .build()?
                .into(),
        );
        let answer = agent.run_turns(&mut messages).await?;
        println!("--- Final response:\n{answer}");
    }
    Ok(())
}

async fn run() -> Result<()> {
    let args = Args::parse();
    let config = AgentRonConfig::load(&args.config)?;
    let provider = init_tracing(&config, args.verbose)?;
    let todo_state = Arc::new(Mutex::new(TodoState::default()));
    let agent = Agent::new(config.clone(), todo_state);

    let result = if let Some(prompt) = args.prompt {
        run_prompt(&agent, &prompt).await
    } else {
        run_repl(&agent, &config).await
    };

    if let Some(provider) = provider {
        provider
            .shutdown()
            .context("Failed to flush OpenTelemetry spans")?;
    }
    result
}

#[tokio::main]
async fn main() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(3600), run())
        .await
        .context("Agent session exceeded the one-hour timeout")?
}

#[cfg(test)]
mod tests {
    use super::{parse_otel_sdk_disabled, resolve_telemetry_enabled};

    #[test]
    fn parses_otel_sdk_disabled_environment_values() {
        assert!(parse_otel_sdk_disabled("true").unwrap());
        assert!(!parse_otel_sdk_disabled("false").unwrap());
        assert!(parse_otel_sdk_disabled("TRUE").unwrap());
        assert!(parse_otel_sdk_disabled("yes").is_err());
    }

    #[test]
    fn telemetry_defaults_to_config_and_environment_can_override() {
        assert!(!resolve_telemetry_enabled(false, None).unwrap());
        assert!(resolve_telemetry_enabled(true, None).unwrap());
        assert!(!resolve_telemetry_enabled(true, Some("true")).unwrap());
        assert!(resolve_telemetry_enabled(false, Some("false")).unwrap());
    }
}
