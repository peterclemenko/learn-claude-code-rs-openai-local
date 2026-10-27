mod agent;

use std::{collections::HashMap, fs, path::PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::Deserialize;
use tracing::info;
use tracing_subscriber::prelude::*;

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    /// Path to the RON configuration file
    #[arg(long, default_value = "config.ron")]
    config: PathBuf,

    /// Prompt to send to the configured model; omit for interactive mode
    #[arg(short, long)]
    prompt: Option<String>,

    /// Enable debug logging
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct AgentConfig {
    pub(crate) default_model: String,
    pub(crate) models: HashMap<String, LlmConfig>,
    pub(crate) model_roles: HashMap<String, String>,
    pub(crate) telemetry: TelemetryConfig,
}

#[derive(Debug, Deserialize)]
pub(crate) struct LlmConfig {
    pub(crate) base_url: String,
    pub(crate) api_key: String,
    pub(crate) model: String,
    pub(crate) temperature: f32,
    pub(crate) request_timeout_secs: u64,
    pub(crate) max_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct TelemetryConfig {
    pub(crate) enabled: bool,
    pub(crate) service_name: String,
    pub(crate) otlp_endpoint: String,
}

fn load_config(path: &PathBuf) -> Result<AgentConfig> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("Failed to read config: {}", path.display()))?;
    let config: AgentConfig =
        ron::from_str(&contents).context("Failed to parse RON configuration")?;

    if !config.models.contains_key(&config.default_model) {
        bail!("Default model profile '{}' is not defined", config.default_model);
    }
    for (role, profile) in &config.model_roles {
        if !config.models.contains_key(profile) {
            bail!("Model role '{role}' references undefined profile '{profile}'");
        }
    }
    Ok(config)
}

fn init_tracing(verbose: bool) -> Result<()> {
    let filter = if verbose { tracing::Level::DEBUG } else { tracing::Level::INFO };
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(std::io::stderr)
                .with_filter(tracing_subscriber::filter::LevelFilter::from_level(filter)),
        )
        .try_init()
        .context("Failed to initialize tracing subscriber")?;
    Ok(())
}

async fn run() -> Result<()> {
    let args = Args::parse();
    init_tracing(args.verbose)?;
    let config = load_config(&args.config)?;

    info!(
        service = %config.telemetry.service_name,
        telemetry_enabled = config.telemetry.enabled,
        otlp_endpoint = %config.telemetry.otlp_endpoint,
        "Application configuration loaded"
    );

    agent::run_agent(&config, args.prompt.as_deref()).await
}

#[tokio::main]
async fn main() -> Result<()> {
    run().await
}
