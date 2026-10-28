use std::{collections::HashMap, fs, path::PathBuf, time::Duration};

use anyhow::{Context, Result, bail};

use async_openai::{
    Client,
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
    },
};
use clap::Parser;
use inquire::Text;
use serde::Deserialize;
use tracing::info;
use tracing_subscriber::prelude::*;

const SYSTEM_PROMPT: &str = "You are a helpful coding assistant.";

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

async fn request_completion(
    client: &Client<OpenAIConfig>,
    model_config: &LlmConfig,
    messages: &[ChatCompletionRequestMessage],
) -> Result<String> {
    let mut request_args = CreateChatCompletionRequestArgs::default();
    request_args
        .model(&model_config.model)
        .messages(messages.to_vec())
        .temperature(model_config.temperature);
    if let Some(max_tokens) = model_config.max_tokens {
        request_args.max_completion_tokens(max_tokens);
    }
    let request = request_args.build()?;

    let response = tokio::time::timeout(
        Duration::from_secs(model_config.request_timeout_secs),
        client.chat().create(request),
    )
    .await
    .context("Model request timed out")?
    .context("OpenAI-compatible API request failed")?;

    response
        .choices
        .first()
        .and_then(|choice| choice.message.content.as_deref())
        .map(str::to_owned)
        .context("Model response did not contain text content")
}

fn resolve_model(config: &AgentConfig) -> Result<(&str, &LlmConfig)> {
    let profile_name = config
        .model_roles
        .get("default")
        .unwrap_or(&config.default_model);
    let model_config = config
        .models
        .get(profile_name)
        .context("Resolved model profile does not exist")?;
    Ok((profile_name, model_config))
}

async fn run_prompt(config: &AgentConfig, prompt: &str) -> Result<()> {
    let (profile_name, model_config) = resolve_model(config)?;
    let client = build_client(model_config);
    let messages = vec![
        ChatCompletionRequestSystemMessageArgs::default()
            .content(SYSTEM_PROMPT)
            .build()?
            .into(),
        ChatCompletionRequestUserMessageArgs::default()
            .content(prompt)
            .build()?
            .into(),
    ];

    info!(
        profile = %profile_name,
        model = %model_config.model,
        "Sending chat completion request"
    );
    let response = request_completion(&client, model_config, &messages).await?;
    println!("{response}");

    Ok(())
}

async fn run_repl(config: &AgentConfig) -> Result<()> {
    let (profile_name, model_config) = resolve_model(config)?;
    let client = build_client(model_config);
    let mut messages = vec![
        ChatCompletionRequestSystemMessageArgs::default()
            .content(SYSTEM_PROMPT)
            .build()?
            .into(),
    ];

    info!(
        profile = %profile_name,
        model = %model_config.model,
        "Interactive session started; enter exit() to quit"
    );

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
        let response = request_completion(&client, model_config, &messages).await?;
        println!("--- Final response:\n{response}");
        messages.push(
            ChatCompletionRequestAssistantMessageArgs::default()
                .content(response)
                .build()?
                .into(),
        );
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
        run_repl(&config).await?;
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    run().await
}
