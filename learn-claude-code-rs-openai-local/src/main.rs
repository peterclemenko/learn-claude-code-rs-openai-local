mod agent;
mod tools;
mod types;

use std::{path::PathBuf, time::Duration};

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

use crate::{agent::Agent, types::AgentRonConfig};

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
    let level = if verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };
    let format_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_filter(tracing_subscriber::filter::LevelFilter::from_level(level));

    if !config.telemetry.enabled {
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
    let agent = Agent::new(config.clone());

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
