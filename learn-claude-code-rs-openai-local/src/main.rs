use std::{collections::HashMap, fs, path::PathBuf, time::Duration};

use anyhow::{Context, Result, bail};

use async_openai::{
    Client,
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
    },
};
use clap::Parser;
use serde::Deserialize;
use tracing::{debug, info};
use tracing_subscriber::prelude::*;

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    /// Path to the RON configuration file
    #[arg(long, default_value = "config.ron")]
    config: PathBuf,

    /// Prompt to send to the configured model
    #[arg(short, long)]
    prompt: Option<String>,

    /// Enable debug logging
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Debug, Deserialize)]
struct AgentConfig {
    default_model: String,
    models: HashMap<String, LlmConfig>,
    model_roles: HashMap<String, String>,
    telemetry: TelemetryConfig,
}

#[derive(Debug, Deserialize)]
struct LlmConfig {
    base_url: String,
    api_key: String,
    model: String,
    temperature: f32,
    request_timeout_secs: u64,
    max_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct TelemetryConfig {
    enabled: bool,
    service_name: String,
    otlp_endpoint: String,
}

fn load_config(path: &PathBuf) -> Result<AgentConfig> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("Failed to read config: {}", path.display()))?;

    let config: AgentConfig =
        ron::from_str(&contents).context("Failed to parse RON configuration")?;

    if !config.models.contains_key(&config.default_model) {
        bail!(
            "Default model profile '{}' is not defined",
            config.default_model
        );
    }

    for (role, profile) in &config.model_roles {
        if !config.models.contains_key(profile) {
            bail!("Model role '{role}' references undefined profile '{profile}'");
        }
    }

    Ok(config)
}

fn init_tracing(verbose: bool) -> Result<()> {
    let filter = if verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };

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

fn build_client(config: &LlmConfig) -> Client<OpenAIConfig> {
    let api_config = OpenAIConfig::new()
        .with_api_base(&config.base_url)
        .with_api_key(&config.api_key);

    Client::with_config(api_config)
}

async fn run_prompt(config: &AgentConfig, prompt: &str) -> Result<()> {
    let profile_name = config
        .model_roles
        .get("default")
        .unwrap_or(&config.default_model);

    let model_config = config
        .models
        .get(profile_name)
        .context("Resolved model profile does not exist")?;

    let client = build_client(model_config);

    let mut request_args = CreateChatCompletionRequestArgs::default();
    request_args
        .model(&model_config.model)
        .messages([
            ChatCompletionRequestSystemMessageArgs::default()
                .content("You are a helpful coding assistant.")
                .build()?
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content(prompt)
                .build()?
                .into(),
        ])
        .temperature(model_config.temperature);
    if let Some(max_tokens) = model_config.max_tokens {
        request_args.max_completion_tokens(max_tokens);
    }
    let request = request_args.build()?;

    info!(
        profile = %profile_name,
        model = %model_config.model,
        "Sending chat completion request"
    );

    let response = tokio::time::timeout(
        Duration::from_secs(model_config.request_timeout_secs),
        client.chat().create(request),
    )
    .await
    .context("Model request timed out")?
    .context("OpenAI-compatible API request failed")?;

    if let Some(content) = response
        .choices
        .first()
        .and_then(|choice| choice.message.content.as_deref())
    {
        println!("{content}");
    } else {
        debug!("Model returned no text content");
    }

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

    if let Some(prompt) = args.prompt {
        run_prompt(&config, &prompt).await?;
    } else {
        info!(
            profile = %config.default_model,
            "Agent initialized; supply --prompt to send a request"
        );
        println!("Configuration loaded successfully.");
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    run().await
}
