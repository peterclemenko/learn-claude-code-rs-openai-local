use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use anyhow::{Context, Result, bail};
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessage,
        ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequest,
        CreateChatCompletionRequestArgs, CreateChatCompletionResponse,
    },
};
use async_trait::async_trait;
use tracing::{Instrument, info, instrument, warn};

use crate::{
    context_window::{ContextWindowProvider, OllamaContextWindowProvider},
    subagent::DelegateTaskTool,
    tools::{BashTool, ReadFileTool, TodoTool, ToolRegistry},
    types::{AgentRonConfig, SubagentConfig, TodoState},
};

#[async_trait]
pub(crate) trait ChatCompletionClient: Send + Sync {
    async fn complete(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse>;
}

struct OpenAiChatCompletionClient {
    client: Client<OpenAIConfig>,
}

#[async_trait]
impl ChatCompletionClient for OpenAiChatCompletionClient {
    async fn complete(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse> {
        self.client
            .chat()
            .create(request)
            .await
            .context("Local OpenAI-compatible chat completion request failed")
    }
}

#[derive(Clone)]
pub(crate) struct AgentRuntime {
    pub(crate) client: Arc<dyn ChatCompletionClient>,
    pub(crate) context_window_provider: Arc<dyn ContextWindowProvider>,
    pub(crate) subagents: Arc<Vec<SubagentConfig>>,
}

pub struct Agent {
    client: Arc<dyn ChatCompletionClient>,
    config: AgentRonConfig,
    tools: ToolRegistry,
    todo_state: Arc<Mutex<TodoState>>,
    context_window_provider: Arc<dyn ContextWindowProvider>,
    depth: usize,
    turns_completed: AtomicUsize,
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
        let client = Arc::new(OpenAiChatCompletionClient {
            client: Client::with_config(
                OpenAIConfig::new()
                    .with_api_base(&config.base_url)
                    .with_api_key(&config.api_key),
            ),
        });
        Self::new_with_chat_client(config, todo_state, context_window_provider, client)
    }

    fn new_with_chat_client(
        config: AgentRonConfig,
        todo_state: Arc<Mutex<TodoState>>,
        context_window_provider: Arc<dyn ContextWindowProvider>,
        client: Arc<dyn ChatCompletionClient>,
    ) -> Self {
        let runtime = AgentRuntime {
            client,
            context_window_provider,
            subagents: Arc::new(config.subagents.clone()),
        };
        let allowed_tools = config.allowed_tools.clone();
        Self::from_runtime(config, todo_state, runtime, 0, None, allowed_tools)
    }

    pub(crate) fn from_runtime(
        config: AgentRonConfig,
        todo_state: Arc<Mutex<TodoState>>,
        runtime: AgentRuntime,
        depth: usize,
        depth_limit: Option<usize>,
        allowed_tools: Vec<String>,
    ) -> Self {
        let mut tools = ToolRegistry::new();
        for tool in &allowed_tools {
            match tool.as_str() {
                "bash" => tools.register(BashTool),
                "read_file" => tools.register(ReadFileTool),
                "todo_write" => tools.register(TodoTool::new(Arc::clone(&todo_state))),
                "delegate_task" if !runtime.subagents.is_empty() => {
                    tools.register(DelegateTaskTool::new(
                        config.clone(),
                        runtime.clone(),
                        depth,
                        depth_limit,
                    ));
                }
                "delegate_task" => {
                    warn!(
                        agent = %config.name,
                        "Agent allows delegation but no subagent definitions are configured"
                    );
                }
                unknown => {
                    warn!(agent = %config.name, tool = unknown, "Unknown agent tool ignored")
                }
            }
        }

        Self {
            client: Arc::clone(&runtime.client),
            config,
            tools,
            todo_state,
            context_window_provider: runtime.context_window_provider,
            depth,
            turns_completed: AtomicUsize::new(0),
        }
    }

    pub(crate) fn turns_completed(&self) -> usize {
        self.turns_completed.load(Ordering::Relaxed)
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

    #[instrument(name = "agent.turn_loop", skip(self, messages), fields(agent = %self.config.name, depth = self.depth))]
    pub async fn run_turns(
        &self,
        messages: &mut Vec<ChatCompletionRequestMessage>,
    ) -> Result<String> {
        self.turns_completed.store(0, Ordering::Relaxed);
        for turn in 1..=self.config.max_turns {
            self.inject_todo_context(messages)?;
            let span = tracing::span!(tracing::Level::INFO, "agent.turn", turn);
            match self.request_turn(messages).instrument(span).await? {
                TurnOutcome::Complete(content) => {
                    self.turns_completed.store(turn, Ordering::Relaxed);
                    return Ok(content);
                }
                TurnOutcome::ToolsDispatched(count) => {
                    self.turns_completed.store(turn, Ordering::Relaxed);
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
            .complete(request)
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
    use std::{
        collections::VecDeque,
        sync::{Arc, Mutex},
    };

    use anyhow::{Context, Result};
    use async_openai::types::chat::{
        CompletionUsage, CreateChatCompletionRequest, CreateChatCompletionResponse,
    };
    use async_trait::async_trait;
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_sdk::{
        Resource,
        error::OTelSdkResult,
        trace::{SdkTracerProvider, SpanData, SpanExporter},
    };
    use serde_json::{Value, json};
    use tracing::{Instrument, instrument::WithSubscriber};
    use tracing_subscriber::prelude::*;

    use super::{Agent, ChatCompletionClient, context_window_utilization, model_usage_metrics};
    use crate::{
        context_window::ContextWindowProvider,
        types::{AgentRonConfig, SubagentConfig, TelemetryConfig, TodoState},
    };

    #[derive(Default)]
    struct MockChatClient {
        responses: Mutex<VecDeque<CreateChatCompletionResponse>>,
        requests: Mutex<Vec<CreateChatCompletionRequest>>,
    }

    impl MockChatClient {
        fn new(responses: Vec<CreateChatCompletionResponse>) -> Self {
            Self {
                responses: Mutex::new(responses.into()),
                requests: Mutex::new(Vec::new()),
            }
        }

        fn requests(&self) -> Vec<CreateChatCompletionRequest> {
            self.requests.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl ChatCompletionClient for MockChatClient {
        async fn complete(
            &self,
            request: CreateChatCompletionRequest,
        ) -> Result<CreateChatCompletionResponse> {
            self.requests.lock().unwrap().push(request);
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .context("Mock chat response queue is empty")
        }
    }

    struct MockContextWindow;

    #[derive(Clone, Debug, Default)]
    struct CapturingExporter(Arc<Mutex<Vec<SpanData>>>);

    impl SpanExporter for CapturingExporter {
        fn export(
            &self,
            batch: Vec<SpanData>,
        ) -> impl std::future::Future<Output = OTelSdkResult> + Send {
            self.0.lock().unwrap().extend(batch);
            std::future::ready(Ok(()))
        }

        fn set_resource(&mut self, _resource: &Resource) {}
    }

    #[derive(Clone)]
    struct CapturingSpans(Arc<Mutex<Vec<String>>>);

    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for CapturingSpans {
        fn on_new_span(
            &self,
            attributes: &tracing::span::Attributes<'_>,
            _id: &tracing::span::Id,
            _context: tracing_subscriber::layer::Context<'_, S>,
        ) {
            self.0
                .lock()
                .unwrap()
                .push(attributes.metadata().name().to_owned());
        }
    }

    #[async_trait]
    impl ContextWindowProvider for MockContextWindow {
        fn provider_name(&self) -> &'static str {
            "mock"
        }

        async fn context_window_tokens(&self, _model: &str) -> Result<Option<u64>> {
            Ok(Some(16_384))
        }
    }

    fn test_config() -> AgentRonConfig {
        AgentRonConfig {
            name: "parent".to_owned(),
            base_url: "http://127.0.0.1:11434/v1".to_owned(),
            api_key: "test-key".to_owned(),
            model: "parent-model".to_owned(),
            system_prompt: "Parent system prompt.".to_owned(),
            temperature: 0.1,
            max_turns: 5,
            allowed_tools: ["bash", "read_file", "todo_write", "delegate_task"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            subagents: vec![
                SubagentConfig {
                    name: "code_runner".to_owned(),
                    description: "Run and validate local code".to_owned(),
                    allowed_tools: ["bash", "delegate_task"]
                        .into_iter()
                        .map(str::to_owned)
                        .collect(),
                    max_depth: 2,
                    system_prompt_template:
                        "You are {name}: {description}. Parent context: {context}".to_owned(),
                    model: Some("code-model".to_owned()),
                },
                SubagentConfig {
                    name: "reviewer".to_owned(),
                    description: "Read-only reviewer".to_owned(),
                    allowed_tools: vec!["read_file".to_owned()],
                    max_depth: 2,
                    system_prompt_template:
                        "You are {name}: {description}. Parent context: {context}".to_owned(),
                    model: None,
                },
            ],
            ollama_base_url: "http://127.0.0.1:11434".to_owned(),
            telemetry: TelemetryConfig::default(),
        }
    }

    fn response(content: Option<&str>, tool_calls: Vec<Value>) -> CreateChatCompletionResponse {
        serde_json::from_value(json!({
            "id": "mock-completion",
            "object": "chat.completion",
            "created": 1,
            "model": "mock-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                },
                "finish_reason": if content.is_some() { "stop" } else { "tool_calls" },
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }))
        .unwrap()
    }

    fn tool_call(id: &str, name: &str, arguments: Value) -> Value {
        json!({
            "id": id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments.to_string(),
            },
        })
    }

    fn request_tool_names(request: &CreateChatCompletionRequest) -> Vec<String> {
        request
            .tools
            .as_ref()
            .unwrap()
            .iter()
            .map(|tool| match tool {
                async_openai::types::chat::ChatCompletionTools::Function(tool) => {
                    tool.function.name.clone()
                }
                async_openai::types::chat::ChatCompletionTools::Custom(_) => {
                    panic!("Unexpected custom tool")
                }
            })
            .collect()
    }

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

    #[tokio::test]
    async fn nested_subagents_use_mock_local_runtime_context_and_tool_permissions() {
        let client = Arc::new(MockChatClient::new(vec![
            response(
                None,
                vec![tool_call(
                    "parent-delegate",
                    "delegate_task",
                    json!({
                        "agent": "code_runner",
                        "task": "inspect the patch",
                        "context": "Parent's selected files and constraints",
                    }),
                )],
            ),
            response(
                None,
                vec![tool_call(
                    "child-bash",
                    "bash",
                    json!({"command": "printf child-tool-ok"}),
                )],
            ),
            response(
                None,
                vec![tool_call(
                    "child-delegate",
                    "delegate_task",
                    json!({
                        "agent": "reviewer",
                        "task": "review the patch for regressions",
                        "context": "Code runner findings from the local test run",
                    }),
                )],
            ),
            response(Some("Review accepted."), vec![]),
            response(Some("Patch tested and reviewed."), vec![]),
            response(Some("Delegated work complete."), vec![]),
        ]));
        let agent = Agent::new_with_chat_client(
            test_config(),
            Arc::new(Mutex::new(TodoState::default())),
            Arc::new(MockContextWindow),
            client.clone(),
        );

        let answer = agent
            .run_prompt("Please inspect and review this patch.")
            .await
            .unwrap();
        assert_eq!(answer, "Delegated work complete.");

        let requests = client.requests();
        assert_eq!(requests.len(), 6);
        assert_eq!(
            requests
                .iter()
                .map(|request| request.model.as_str())
                .collect::<Vec<_>>(),
            [
                "parent-model",
                "code-model",
                "code-model",
                "code-model",
                "code-model",
                "parent-model",
            ]
        );
        assert_eq!(request_tool_names(&requests[1]), ["bash", "delegate_task"]);
        assert_eq!(request_tool_names(&requests[3]), ["read_file"]);

        let child_request = serde_json::to_value(&requests[1]).unwrap();
        assert!(
            child_request["messages"][0]["content"]
                .as_str()
                .unwrap()
                .contains("Parent context: Parent's selected files and constraints")
        );
        assert_eq!(child_request["messages"][1]["content"], "inspect the patch");
        let bash_followup = serde_json::to_value(&requests[2]).unwrap();
        assert!(
            bash_followup["messages"]
                .as_array()
                .unwrap()
                .iter()
                .any(|message| {
                    message["role"] == "tool"
                        && message["content"]
                            .as_str()
                            .is_some_and(|content| content.contains("child-tool-ok"))
                })
        );
        let nested_request = serde_json::to_value(&requests[3]).unwrap();
        assert!(
            nested_request["messages"][0]["content"]
                .as_str()
                .unwrap()
                .contains("Parent context: Code runner findings from the local test run")
        );

        let parent_turn = serde_json::to_value(&requests[5]).unwrap();
        let tool_response = parent_turn["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|message| message["role"] == "tool")
            .unwrap()["content"]
            .as_str()
            .unwrap();
        let result = crate::subagent::SubagentResult::from_toon(tool_response).unwrap();
        assert_eq!(result.agent, "code_runner");
        assert_eq!(result.status, crate::subagent::SubagentStatus::Completed);
        assert!(result.result.contains("Patch tested and reviewed."));
    }

    #[tokio::test]
    async fn nested_delegation_inherits_the_strictest_persona_depth_limit() {
        let client = Arc::new(MockChatClient::new(vec![
            response(
                None,
                vec![tool_call(
                    "parent-delegate",
                    "delegate_task",
                    json!({
                        "agent": "code_runner",
                        "task": "inspect the patch",
                        "context": "parent context",
                    }),
                )],
            ),
            response(
                None,
                vec![tool_call(
                    "child-delegate",
                    "delegate_task",
                    json!({
                        "agent": "reviewer",
                        "task": "review the patch",
                        "context": "child context",
                    }),
                )],
            ),
            response(Some("The nested request was denied."), vec![]),
            response(Some("Done."), vec![]),
        ]));
        let mut config = test_config();
        config.subagents[0].max_depth = 1;
        let agent = Agent::new_with_chat_client(
            config,
            Arc::new(Mutex::new(TodoState::default())),
            Arc::new(MockContextWindow),
            client.clone(),
        );

        assert_eq!(
            agent.run_prompt("Inspect the patch.").await.unwrap(),
            "Done."
        );
        let requests = client.requests();
        assert_eq!(requests.len(), 4);
        let nested_tool_result = serde_json::to_value(&requests[2]).unwrap();
        assert!(
            nested_tool_result["messages"]
                .as_array()
                .unwrap()
                .iter()
                .any(|message| {
                    message["role"] == "tool"
                        && message["content"]
                            .as_str()
                            .is_some_and(|content| content.contains("exceeds max_depth 1"))
                })
        );
    }

    #[tokio::test]
    async fn nested_delegations_propagate_opentelemetry_trace_and_parent_ids() {
        let client = Arc::new(MockChatClient::new(vec![
            response(
                None,
                vec![tool_call(
                    "parent-delegate",
                    "delegate_task",
                    json!({
                        "agent": "code_runner",
                        "task": "inspect the patch",
                        "context": "parent context",
                    }),
                )],
            ),
            response(
                None,
                vec![tool_call(
                    "child-delegate",
                    "delegate_task",
                    json!({
                        "agent": "reviewer",
                        "task": "review the patch",
                        "context": "nested context",
                    }),
                )],
            ),
            response(Some("Review accepted."), vec![]),
            response(Some("Patch is sound."), vec![]),
            response(Some("Done."), vec![]),
        ]));
        let agent = Agent::new_with_chat_client(
            test_config(),
            Arc::new(Mutex::new(TodoState::default())),
            Arc::new(MockContextWindow),
            client,
        );
        let exporter = CapturingExporter::default();
        let captured_span_names = Arc::new(Mutex::new(Vec::new()));
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        let subscriber = tracing_subscriber::registry()
            .with(CapturingSpans(Arc::clone(&captured_span_names)))
            .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("subagent-test")));
        let dispatch = tracing::Dispatch::new(subscriber);

        async {
            let parent = tracing::info_span!("parent_execution");
            agent
                .run_prompt("Delegate a nested review.")
                .instrument(parent)
                .await
        }
        .with_subscriber(dispatch)
        .await
        .unwrap();
        provider.force_flush().unwrap();

        let spans = exporter.0.lock().unwrap().clone();
        let captured_span_names = captured_span_names.lock().unwrap().clone();
        let root = spans
            .iter()
            .find(|span| span.name == "parent_execution")
            .unwrap();
        let by_span_id = spans
            .iter()
            .map(|span| (span.span_context.span_id(), span))
            .collect::<std::collections::HashMap<_, _>>();
        let subagent_spans = spans
            .iter()
            .filter(|span| span.name == "subagent_execution")
            .collect::<Vec<_>>();
        assert_eq!(
            subagent_spans.len(),
            2,
            "Captured tracing spans: {captured_span_names:?}; exported spans: {spans:#?}"
        );
        let span_with_name = |name: &str| {
            subagent_spans
                .iter()
                .find(|span| {
                    span.attributes.iter().any(|attribute| {
                        attribute.key.as_str() == "subagent_name"
                            && attribute.value.as_str() == name
                    })
                })
                .copied()
                .unwrap()
        };
        let code_runner = span_with_name("code_runner");
        let reviewer = span_with_name("reviewer");
        for span in subagent_spans {
            assert_eq!(span.span_context.trace_id(), root.span_context.trace_id());
            assert!(
                by_span_id.contains_key(&span.parent_span_id),
                "Subagent span parent {:?} was not exported",
                span.parent_span_id
            );
            assert!(
                span.attributes.iter().any(|attribute| {
                    attribute.key.as_str() == "subagent.id"
                        && attribute.value.as_str().starts_with("subagent-")
                }),
                "Subagent span has no delegation id: {:?}",
                span.attributes
            );
            assert!(
                span.attributes
                    .iter()
                    .any(|attribute| attribute.key.as_str() == "subagent.task")
            );
            assert!(
                span.attributes
                    .iter()
                    .any(|attribute| attribute.key.as_str() == "subagent.model")
            );
        }
        assert!(has_ancestor(
            &by_span_id,
            code_runner,
            root.span_context.span_id()
        ));
        assert!(has_ancestor(
            &by_span_id,
            reviewer,
            code_runner.span_context.span_id()
        ));
    }

    fn has_ancestor(
        spans: &std::collections::HashMap<opentelemetry::trace::SpanId, &SpanData>,
        child: &SpanData,
        ancestor_id: opentelemetry::trace::SpanId,
    ) -> bool {
        let mut parent_id = child.parent_span_id;
        while let Some(parent) = spans.get(&parent_id) {
            if parent.span_context.span_id() == ancestor_id {
                return true;
            }
            parent_id = parent.parent_span_id;
        }
        false
    }
}
