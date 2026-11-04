use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessage,
        ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
    },
};
use tracing::{Instrument, info, instrument, warn};

use crate::{
    context_window::{ContextWindowProvider, OllamaContextWindowProvider},
    tools::{BashTool, ReadFileTool, TodoTool, ToolRegistry},
    types::{AgentRonConfig, TodoState},
};

pub struct Agent {
    client: Client<OpenAIConfig>,
    config: AgentRonConfig,
    tools: ToolRegistry,
    todo_state: Arc<Mutex<TodoState>>,
    context_window_provider: Arc<dyn ContextWindowProvider>,
}

impl Agent {
    pub fn new(config: AgentRonConfig, todo_state: Arc<Mutex<TodoState>>) -> Self {
        let context_window_provider =
            Arc::new(OllamaContextWindowProvider::new(&config.ollama_base_url));
        Self::new_with_context_window_provider(config, todo_state, context_window_provider)
    }

    pub fn new_with_context_window_provider(
        config: AgentRonConfig,
        todo_state: Arc<Mutex<TodoState>>,
        context_window_provider: Arc<dyn ContextWindowProvider>,
    ) -> Self {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(&config.base_url)
                .with_api_key(&config.api_key),
        );
        let mut tools = ToolRegistry::new();
        tools.register(BashTool);
        tools.register(ReadFileTool);
        tools.register(TodoTool::new(Arc::clone(&todo_state)));

        Self {
            client,
            config,
            tools,
            todo_state,
            context_window_provider,
        }
    }

    #[instrument(name = "agent.prompt", skip(self, prompt))]
    pub async fn run_prompt(&self, prompt: &str) -> Result<String> {
        let mut messages = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content(self.config.system_prompt.as_str())
                .build()?
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content(prompt)
                .build()?
                .into(),
        ];
        self.run_turns(&mut messages).await
    }

    #[instrument(name = "agent.turn_loop", skip(self, messages), fields(agent = %self.config.name))]
    pub async fn run_turns(
        &self,
        messages: &mut Vec<ChatCompletionRequestMessage>,
    ) -> Result<String> {
        for turn in 1..=self.config.max_turns {
            self.inject_todo_context(messages)?;
            let span = tracing::span!(tracing::Level::INFO, "agent.turn", turn);
            match self.request_turn(messages).instrument(span).await? {
                TurnOutcome::Complete(content) => return Ok(content),
                TurnOutcome::ToolsDispatched(count) => {
                    info!(
                        turn,
                        tool_calls = count,
                        "Tool results added to the conversation"
                    );
                }
            }
        }

        bail!(
            "Agent reached max_turns ({}) before producing a final response",
            self.config.max_turns
        )
    }

    fn inject_todo_context(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<()> {
        let todo_context = self
            .todo_state
            .lock()
            .map_err(|error| anyhow::anyhow!("Todo state mutex is poisoned: {error}"))?
            .render_toon_context();
        let system_prompt = format!(
            "{}\n\nActive TODO task state (TOON):\n{}",
            self.config.system_prompt, todo_context
        );
        let system_message = ChatCompletionRequestSystemMessageArgs::default()
            .content(system_prompt)
            .build()?
            .into();

        if matches!(
            messages.first(),
            Some(ChatCompletionRequestMessage::System(_))
        ) {
            messages[0] = system_message;
        } else {
            messages.insert(0, system_message);
        }
        Ok(())
    }

    #[instrument(
        name = "agent.request_turn",
        skip(self, messages),
        fields(
            model = %self.config.model,
            context_window_provider = self.context_window_provider.provider_name(),
            prompt_tokens = tracing::field::Empty,
            completion_tokens = tracing::field::Empty,
            total_tokens = tracing::field::Empty,
            context_window_tokens = tracing::field::Empty,
            context_utilization_pct = tracing::field::Empty,
        )
    )]
    async fn request_turn(
        &self,
        messages: &mut Vec<ChatCompletionRequestMessage>,
    ) -> Result<TurnOutcome> {
        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.config.model)
            .messages(messages.clone())
            .temperature(self.config.temperature)
            .tools(self.tools.openai_tools())
            .build()?;

        let response = self
            .client
            .chat()
            .create(request)
            .instrument(tracing::span!(
                tracing::Level::INFO,
                "llm.chat_completion",
                model = %self.config.model
            ))
            .await
            .context("Local OpenAI-compatible chat completion request failed")?;
        let context_window_tokens = match self
            .context_window_provider
            .context_window_tokens(&self.config.model)
            .await
        {
            Ok(Some(context_window)) => Some(context_window),
            Ok(None) => {
                warn!(
                    model = %self.config.model,
                    context_window_provider = self.context_window_provider.provider_name(),
                    "Context window is unavailable for the active model"
                );
                None
            }
            Err(error) => {
                warn!(
                    model = %self.config.model,
                    context_window_provider = self.context_window_provider.provider_name(),
                    error = %error,
                    "Could not retrieve the active model context window"
                );
                None
            }
        };
        match response.usage.as_ref() {
            Some(usage) => {
                let metrics = model_usage_metrics(usage, context_window_tokens);
                let span = tracing::Span::current();
                span.record("prompt_tokens", metrics.prompt_tokens);
                span.record("completion_tokens", metrics.completion_tokens);
                span.record("total_tokens", metrics.total_tokens);
                if let Some(capacity) = metrics.context_window_tokens {
                    span.record("context_window_tokens", capacity);
                }
                if let Some(utilization) = metrics.context_utilization_pct {
                    span.record("context_utilization_pct", utilization);
                }
                info!(
                    model = %self.config.model,
                    context_window_provider = self.context_window_provider.provider_name(),
                    prompt_tokens = metrics.prompt_tokens,
                    completion_tokens = metrics.completion_tokens,
                    total_tokens = metrics.total_tokens,
                    context_window_tokens = ?metrics.context_window_tokens,
                    context_utilization_pct = ?metrics.context_utilization_pct,
                    "Model token usage"
                );
            }
            None => warn!(
                model = %self.config.model,
                context_window_provider = self.context_window_provider.provider_name(),
                context_window_tokens = ?context_window_tokens,
                "Backend did not report model token usage"
            ),
        }
        let response_message = response
            .choices
            .first()
            .context("Model returned no completion choices")?
            .message
            .clone();

        if let Some(tool_calls) = response_message
            .tool_calls
            .filter(|tool_calls| !tool_calls.is_empty())
        {
            let count = tool_calls.len();
            messages.push(ChatCompletionRequestMessage::Assistant(
                ChatCompletionRequestAssistantMessage {
                    content: response_message
                        .content
                        .map(ChatCompletionRequestAssistantMessageContent::Text),
                    tool_calls: Some(tool_calls.clone()),
                    ..Default::default()
                },
            ));

            for tool_call in tool_calls {
                let ChatCompletionMessageToolCalls::Function(call) = tool_call else {
                    continue;
                };
                info!(
                    tool = %call.function.name,
                    call_id = %call.id,
                    "Dispatching TOON tool call"
                );
                let result = self
                    .tools
                    .dispatch(&call.function.name, &call.function.arguments)
                    .await?;
                let tool_message = ChatCompletionRequestToolMessageArgs::default()
                    .tool_call_id(call.id)
                    .content(result)
                    .build()?;
                messages.push(ChatCompletionRequestMessage::Tool(tool_message));
            }
            return Ok(TurnOutcome::ToolsDispatched(count));
        }

        let content = response_message.content.unwrap_or_default();
        messages.push(ChatCompletionRequestMessage::Assistant(
            ChatCompletionRequestAssistantMessage {
                content: Some(ChatCompletionRequestAssistantMessageContent::Text(
                    content.clone(),
                )),
                ..Default::default()
            },
        ));
        Ok(TurnOutcome::Complete(content))
    }
}

enum TurnOutcome {
    Complete(String),
    ToolsDispatched(usize),
}

struct ModelUsageMetrics {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
    context_window_tokens: Option<u64>,
    context_utilization_pct: Option<f64>,
}

fn model_usage_metrics(
    usage: &async_openai::types::chat::CompletionUsage,
    context_window_tokens: Option<u64>,
) -> ModelUsageMetrics {
    let context_utilization_pct = context_window_tokens
        .and_then(|capacity| context_window_utilization(usage.prompt_tokens, capacity));
    ModelUsageMetrics {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
        context_window_tokens,
        context_utilization_pct,
    }
}

fn context_window_utilization(prompt_tokens: u32, capacity: u64) -> Option<f64> {
    (capacity > 0).then(|| f64::from(prompt_tokens) / capacity as f64 * 100.0)
}

#[cfg(test)]
mod tests {
    use async_openai::types::chat::CompletionUsage;

    use super::{context_window_utilization, model_usage_metrics};

    #[test]
    fn reports_exact_provider_token_usage_and_context_utilization() {
        let usage = CompletionUsage {
            prompt_tokens: 4096,
            completion_tokens: 256,
            total_tokens: 4352,
            ..Default::default()
        };
        let metrics = model_usage_metrics(&usage, Some(32768));
        assert_eq!(metrics.prompt_tokens, 4096);
        assert_eq!(metrics.completion_tokens, 256);
        assert_eq!(metrics.total_tokens, 4352);
        assert_eq!(metrics.context_window_tokens, Some(32768));
        assert_eq!(metrics.context_utilization_pct, Some(12.5));
        assert_eq!(context_window_utilization(4096, 32768), Some(12.5));
        assert_eq!(context_window_utilization(1, 0), None);
    }
}
