// src/agent/loop.rs
use crate::tools::registry::ToolRegistry;
use async_openai::{
    Client,
    types::{
        ChatCompletionMessageToolCall, ChatCompletionRequestMessage,
        CreateChatCompletionRequestArgs, ResponseFormat,
    },
};

pub async fn run_agent_loop(
    client: &Client<async_openai::config::OpenAIConfig>,
    model: &str,
    registry: &ToolRegistry,
    messages: &mut Vec<ChatCompletionRequestMessage>,
) -> Result<(), Box<dyn std::error::Error>> {
    let openai_tools = registry.to_openai_tools();

    let request = CreateChatCompletionRequestArgs::default()
        .model(model)
        .messages(messages.clone())
        .tools(openai_tools)
        .build()?;

    let response = client.chat().create(request).await?;
    let choice = response.choices.first().ok_or("No response from model")?;

    if let Some(tool_calls) = &choice.message.tool_calls {
        // Handle tool execution block
        for tool_call in tool_calls {
            let func = &tool_call.function;
            println!(
                "[Executing tool: {} with args: {}]",
                func.name, func.arguments
            );

            let result = match registry.dispatch(&func.name, &func.arguments).await {
                Ok(output) => output,
                Err(err) => format!("Error: {}", err),
            };

            // Push assistant tool call and tool response back into message history
            // ...
        }
    }

    Ok(())
}
