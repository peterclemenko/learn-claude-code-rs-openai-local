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
use tracing::{Instrument, info, instrument};

use crate::{
    tools::{BashTool, ReadFileTool, ToolRegistry},
    types::AgentRonConfig,
};

pub struct Agent {
    client: Client<OpenAIConfig>,
    config: AgentRonConfig,
    tools: ToolRegistry,
}

impl Agent {
    pub fn new(config: AgentRonConfig) -> Self {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(&config.base_url)
                .with_api_key(&config.api_key),
        );
        let mut tools = ToolRegistry::new();
        tools.register(BashTool);
        tools.register(ReadFileTool);

        Self {
            client,
            config,
            tools,
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

    #[instrument(name = "agent.request_turn", skip(self, messages))]
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
