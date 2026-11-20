mod agent;
mod compactor;
mod context_window;
mod hooks;
mod permission;
mod skills;
mod subagent;
mod telemetry;
#[cfg(test)]
mod test_telemetry;
mod tools;
mod types;

use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::{
    agent::Agent,
    compactor::CompactionConfig,
    telemetry::init_telemetry,
    types::{AgentRonConfig, TodoState},
};
use anyhow::{Context, Result};
use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestUserMessageArgs,
};
use clap::Parser;
use inquire::Text;

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    /// Path to the RON agent configuration
    #[arg(long, default_value = "config/agent.ron")]
    config: PathBuf,

    /// Path to the RON context-compaction configuration
    #[arg(long, default_value = "config/compaction.ron")]
    compaction_config: PathBuf,

    /// Prompt to send without starting the interactive session
    #[arg(short, long)]
    prompt: Option<String>,

    /// Enable debug-level logs
    #[arg(short, long)]
    verbose: bool,
}

async fn run_prompt(agent: &Agent, prompt: &str) -> Result<()> {
    let answer = agent.run_prompt(prompt).await?;
    println!("{answer}");
    Ok(())
}

async fn run_repl(agent: &Agent, config: &AgentRonConfig) -> Result<()> {
    agent.session_start().await;
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
        if matches!(
            prompt.split_whitespace().next(),
            Some("/rules" | "/mode" | "/profile")
        ) {
            match agent.handle_permission_command(&prompt) {
                Ok(output) => println!("{output}"),
                Err(error) => eprintln!("{error:#}"),
            }
            continue;
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
    agent.session_end().await;
    Ok(())
}

async fn run() -> Result<()> {
    let args = Args::parse();
    let config = AgentRonConfig::load(&args.config)?;
    let compaction_config = CompactionConfig::load(&args.compaction_config)?;
    let telemetry = init_telemetry(
        config.telemetry.enabled,
        &config.telemetry.service_name,
        &config.telemetry.otlp_endpoint,
    )?;
    let todo_state = Arc::new(Mutex::new(TodoState::default()));
    let agent = Agent::new(config.clone(), todo_state, compaction_config)?;

    let result = if let Some(prompt) = args.prompt {
        run_prompt(&agent, &prompt).await
    } else {
        run_repl(&agent, &config).await
    };

    telemetry.shutdown()?;
    result
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    tokio::time::timeout(Duration::from_secs(3600), run())
        .await
        .context("Agent session exceeded the one-hour timeout")?
}

#[cfg(test)]
mod tests {
    #[test]
    fn main_module_keeps_cli_rendering_separate_from_telemetry() {}
}
