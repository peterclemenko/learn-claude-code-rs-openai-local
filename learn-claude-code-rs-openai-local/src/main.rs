mod agent;
mod compactor;
mod context_window;
mod error_recovery;
mod hooks;
pub mod memory;
mod permission;
pub mod prompt;
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

use crate::prompt::{PromptChangeSource, PromptConfig, PromptManager};
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

    /// Name or .ron filename of the active system prompt
    #[arg(long, alias = "prompt-template", alias = "system-prompt")]
    prompt_name: Option<String>,

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
    let result = async {
        let mut messages: Vec<ChatCompletionRequestMessage> = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content(config.system_prompt.as_str())
                .build()?
                .into(),
        ];

        loop {
            let prompt = match Text::new("--- How can I help you?").prompt() {
                Ok(prompt) => prompt,
                Err(error) if is_normal_repl_exit(&error) => {
                    tracing::info!(reason = %error, "Interactive session ended by user");
                    break;
                }
                Err(error) => return Err(error).context("Failed to read prompt"),
            };
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
        Ok(())
    }
    .await;
    agent.session_end().await;
    result
}

fn is_normal_repl_exit(error: &inquire::error::InquireError) -> bool {
    matches!(
        error,
        inquire::error::InquireError::OperationInterrupted
            | inquire::error::InquireError::OperationCanceled
    )
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
    let prompt_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("config/prompts");
    let prompt_manager = Arc::new(PromptManager::new(prompt_dir)?);
    let agent = Agent::new_with_prompts(
        config.clone(),
        Arc::clone(&todo_state),
        compaction_config.clone(),
        Arc::clone(&prompt_manager),
    )?;
    let prompt_config = std::fs::read_to_string("config/prompts.ron")
        .ok()
        .and_then(|contents| ron::from_str::<PromptConfig>(&contents).ok());
    let legacy_prompt_name = prompt_config
        .as_ref()
        .and_then(|config| config.active_prompt.as_deref());
    if let Some(prompt_name) = config.active_prompt.as_deref().or(legacy_prompt_name) {
        prompt_manager
            .select_prompt(prompt_name, PromptChangeSource::Config)
            .await?;
    }
    if let Some(prompt_name) = args.prompt_name.as_deref() {
        prompt_manager
            .select_prompt(prompt_name, PromptChangeSource::CliFlag)
            .await?;
    }

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
    use inquire::error::InquireError;

    use super::is_normal_repl_exit;

    #[test]
    fn prompt_interruptions_and_cancellations_are_normal_repl_exits() {
        assert!(is_normal_repl_exit(&InquireError::OperationInterrupted));
        assert!(is_normal_repl_exit(&InquireError::OperationCanceled));
        assert!(!is_normal_repl_exit(&InquireError::NotTTY));
    }
}
