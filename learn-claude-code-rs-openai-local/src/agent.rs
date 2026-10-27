use std::{
    collections::HashMap,
    fs,
    path::PathBuf,
    process::Stdio,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use async_openai::{Client, config::OpenAIConfig};
use async_trait::async_trait;
use inquire::{Confirm, Select, Text};
use rmcp::{
    RoleClient, ServiceExt,
    model::{CallToolRequestParams, RawContent, ResourceContents, Tool as McpTool},
    service::RunningService,
    transport::{ConfigureCommandExt, TokioChildProcess},
};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tokio::{process::Command, time::timeout};
use tracing::{debug, info, warn};

use crate::{AgentConfig, LlmConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PermissionMode {
    Default,
    Plan,
    Auto,
}

impl PermissionMode {
    fn label(self) -> &'static str {
        match self {
            Self::Default => "default (ask before writes)",
            Self::Plan => "plan (read-only)",
            Self::Auto => "auto (allow ordinary writes)",
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PluginManifest {
    name: String,
    #[serde(default)]
    mcp_servers: HashMap<String, McpServerConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct McpServerConfig {
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: HashMap<String, String>,
}

struct McpConnection {
    server_name: String,
    service: RunningService<RoleClient, ()>,
    tools: Vec<McpTool>,
}

impl McpConnection {
    async fn connect(server_name: String, config: McpServerConfig) -> Result<Self> {
        let command = config.command;
        let args = config.args;
        let env = config.env;
        let transport = TokioChildProcess::builder(Command::new(&command).configure(move |cmd| {
            cmd.args(&args).envs(&env).stderr(Stdio::inherit());
        }))
        .spawn()
        .with_context(|| format!("failed to spawn MCP server {server_name}"))?
        .0;

        let service = ().serve(transport).await
            .with_context(|| format!("failed to initialize MCP server {server_name}"))?;
        let tools = match service.peer().list_all_tools().await {
            Ok(tools) => tools,
            Err(error) => {
                let _ = service.cancel().await;
                return Err(error).with_context(|| format!("failed to list tools from {server_name}"));
            }
        };
        Ok(Self { server_name, service, tools })
    }

    async fn call(&self, tool_name: &str, arguments: Value) -> Result<String> {
        let arguments = match arguments {
            Value::Object(map) => Some(map),
            Value::Null => None,
            other => {
                let mut map = Map::new();
                map.insert("value".to_string(), other);
                Some(map)
            }
        };
        let mut params = CallToolRequestParams::new(tool_name.to_string());
        if let Some(arguments) = arguments {
            params = params.with_arguments(arguments);
        }
        let result = self.service.peer().call_tool(params).await
            .with_context(|| format!("MCP tool call failed: {tool_name}"))?;
        let parts = result.content.iter().filter_map(|content| match &content.raw {
            RawContent::Text(text) => Some(text.text.clone()),
            RawContent::Resource(resource) => match &resource.resource {
                ResourceContents::TextResourceContents { text, .. } => Some(text.clone()),
                _ => None,
            },
            other => serde_json::to_string(other).ok(),
        }).collect::<Vec<_>>();
        Ok(parts.join("\n"))
    }

    async fn shutdown(self) {
        let _ = self.service.cancel().await;
    }
}

/// Runtime boundary: future Rune/Grafeo-backed tools can replace this implementation.
#[async_trait]
trait ToolRuntime {
    fn tool_specs(&self) -> Vec<Value>;
    async fn invoke(&mut self, name: &str, input: &Value) -> Result<String>;
    async fn shutdown(&mut self);
}

#[derive(Default)]
struct LocalToolRuntime {
    mcp: HashMap<String, McpConnection>,
    mcp_tool_index: HashMap<String, (String, String)>,
}

impl LocalToolRuntime {
    async fn discover() -> Self {
        let mut runtime = Self::default();
        let manifest_path = match std::env::current_dir() {
            Ok(cwd) => cwd.join(".claude-plugin").join("plugin.json"),
            Err(error) => {
                warn!(%error, "could not determine current directory");
                return runtime;
            }
        };
        if !manifest_path.exists() {
            info!("No .claude-plugin/plugin.json found; starting without MCP plugins");
            return runtime;
        }

        let manifest = match fs::read_to_string(&manifest_path)
            .with_context(|| format!("failed to read {}", manifest_path.display()))
            .and_then(|raw| serde_json::from_str::<PluginManifest>(&raw).context("invalid plugin manifest"))
        {
            Ok(manifest) => manifest,
            Err(error) => {
                warn!(%error, "MCP plugin manifest could not be loaded");
                return runtime;
            }
        };

        info!(plugin = %manifest.name, "Loading MCP plugin manifest");
        for (server, config) in manifest.mcp_servers {
            let server_name = format!("{}__{}", manifest.name, server);
            match McpConnection::connect(server_name.clone(), config).await {
                Ok(connection) => {
                    info!(server = %server_name, count = connection.tools.len(), "Connected MCP server");
                    for tool in &connection.tools {
                        let exposed = format!("mcp__{}__{}", server_name, tool.name);
                        runtime.mcp_tool_index.insert(exposed, (server_name.clone(), tool.name.to_string()));
                    }
                    runtime.mcp.insert(server_name, connection);
                }
                Err(error) => warn!(server = %server_name, %error, "MCP server connection failed"),
            }
        }
        runtime
    }

    fn native_specs() -> Vec<Value> {
        vec![
            json!({"type":"function","function":{"name":"read_file","description":"Read a UTF-8 text file within the current workspace.","parameters":{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer","description":"Optional maximum number of lines"}},"required":["path"]}}}),
            json!({"type":"function","function":{"name":"write_file","description":"Write a UTF-8 text file within the current workspace. Requires permission.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}),
            json!({"type":"function","function":{"name":"edit_file","description":"Replace the first exact occurrence of text in a workspace file. Requires permission.","parameters":{"type":"object","properties":{"path":{"type":"string"},"old_text":{"type":"string"},"new_text":{"type":"string"}},"required":["path","old_text","new_text"]}}}),
            json!({"type":"function","function":{"name":"bash","description":"Run a shell command in the current workspace. Requires permission and safety checks.","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}),
        ]
    }
}

#[async_trait]
impl ToolRuntime for LocalToolRuntime {
    fn tool_specs(&self) -> Vec<Value> {
        let mut specs = Self::native_specs();
        for (exposed, (server_name, tool_name)) in &self.mcp_tool_index {
            if let Some(server) = self.mcp.get(server_name) {
                if let Some(tool) = server.tools.iter().find(|tool| tool.name.as_ref() == tool_name) {
                    specs.push(json!({
                        "type": "function",
                        "function": {
                            "name": exposed,
                            "description": tool.description.as_deref().unwrap_or("MCP server tool"),
                            "parameters": Value::Object((*tool.input_schema).clone())
                        }
                    }));
                }
            }
        }
        specs
    }

    async fn invoke(&mut self, name: &str, input: &Value) -> Result<String> {
        if let Some((server_name, tool_name)) = self.mcp_tool_index.get(name) {
            let server = self.mcp.get(server_name).context("MCP server is no longer connected")?;
            return server.call(tool_name, input.clone()).await;
        }

        match name {
            "read_file" => {
                let path = workspace_path(input, "path", false)?;
                let content = tokio::fs::read_to_string(&path).await
                    .with_context(|| format!("failed to read {}", path.display()))?;
                let total = content.lines().count();
                let output = if let Some(limit) = input.get("limit").and_then(Value::as_u64) {
                    let limit = limit as usize;
                    let mut lines = content.lines().take(limit).collect::<Vec<_>>().join("\n");
                    if total > limit {
                        lines.push_str(&format!("\n... ({} more lines)", total - limit));
                    }
                    lines
                } else { content };
                Ok(output.chars().take(50_000).collect())
            }
            "write_file" => {
                let path = workspace_path(input, "path", true)?;
                let content = input.get("content").and_then(Value::as_str).context("missing string field 'content'")?;
                if let Some(parent) = path.parent() { tokio::fs::create_dir_all(parent).await?; }
                tokio::fs::write(&path, content).await
                    .with_context(|| format!("failed to write {}", path.display()))?;
                Ok(format!("Wrote {} bytes to {}", content.len(), path.display()))
            }
            "edit_file" => {
                let path = workspace_path(input, "path", false)?;
                let old = input.get("old_text").and_then(Value::as_str).context("missing string field 'old_text'")?;
                let new = input.get("new_text").and_then(Value::as_str).context("missing string field 'new_text'")?;
                let content = tokio::fs::read_to_string(&path).await?;
                if !content.contains(old) { bail!("old_text was not found in {}", path.display()); }
                tokio::fs::write(&path, content.replacen(old, new, 1)).await?;
                Ok(format!("Edited {}", path.display()))
            }
            "bash" => {
                let command = input.get("command").and_then(Value::as_str).context("missing string field 'command'")?;
                let lowered = command.to_ascii_lowercase();
                let blocked = ["rm -rf /", "sudo ", "shutdown", "reboot", "> /dev/"];
                if blocked.iter().any(|pattern| lowered.contains(pattern)) {
                    bail!("command blocked by the native safety filter");
                }
                let child = Command::new("sh")
                    .arg("-c").arg(command)
                    .current_dir(std::env::current_dir()?)
                    .stdout(Stdio::piped()).stderr(Stdio::piped())
                    .kill_on_drop(true).spawn()?;
                let output = timeout(Duration::from_secs(120), child.wait_with_output())
                    .await.context("shell command timed out after 120 seconds")??;
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let combined = format!("{stdout}{stderr}");
                let trimmed = combined.trim();
                if trimmed.is_empty() { Ok(format!("Command exited with status {}", output.status)) }
                else { Ok(trimmed.chars().take(50_000).collect()) }
            }
            _ => bail!("unknown tool: {name}"),
        }
    }

    async fn shutdown(&mut self) {
        for (_, server) in self.mcp.drain() { server.shutdown().await; }
    }
}

fn workspace_path(input: &Value, key: &str, allow_missing_leaf: bool) -> Result<PathBuf> {
    let raw = input.get(key).and_then(Value::as_str).context("missing string path")?;
    let cwd = std::env::current_dir()?.canonicalize()?;
    let candidate = cwd.join(raw);
    let resolved = if candidate.exists() {
        candidate.canonicalize()?
    } else if allow_missing_leaf {
        let parent = candidate.parent().context("path has no parent")?.canonicalize()?;
        if !parent.starts_with(&cwd) { bail!("path escapes workspace"); }
        parent.join(candidate.file_name().context("path has no file name")?)
    } else {
        bail!("path does not exist: {}", candidate.display());
    };
    if !resolved.starts_with(&cwd) { bail!("path escapes workspace"); }
    Ok(resolved)
}

fn is_read_only(name: &str) -> bool {
    name == "read_file" || name.rsplit("__").next().is_some_and(|tool| {
        ["read", "list", "get", "show", "search", "query", "inspect"]
            .iter().any(|prefix| tool.starts_with(prefix))
    })
}

fn is_high_risk(name: &str, input: &Value) -> bool {
    let short = name.rsplit("__").next().unwrap_or(name);
    if ["delete", "remove", "drop", "shutdown"].iter().any(|prefix| short.starts_with(prefix)) {
        return true;
    }
    if name == "bash" {
        let command = input.get("command").and_then(Value::as_str).unwrap_or("").to_ascii_lowercase();
        return ["rm -rf", "sudo", "shutdown", "reboot"].iter().any(|pattern| command.contains(pattern));
    }
    false
}

fn permitted(mode: PermissionMode, name: &str, input: &Value) -> Result<bool> {
    if is_read_only(name) { return Ok(true); }
    if mode == PermissionMode::Plan {
        println!("[DENIED] plan mode blocks write-capable tool {name}");
        return Ok(false);
    }
    if mode == PermissionMode::Auto && !is_high_risk(name, input) { return Ok(true); }
    let preview = serde_json::to_string(input)?;
    Confirm::new(&format!("Allow {name} with arguments {preview}?"))
        .with_default(false)
        .prompt()
        .context("failed to read permission decision")
}

fn build_client(config: &LlmConfig) -> Client<OpenAIConfig> {
    let api_config = OpenAIConfig::new()
        .with_api_base(&config.base_url)
        .with_api_key(&config.api_key);
    Client::with_config(api_config)
}

pub(crate) async fn run_agent(config: &AgentConfig, initial_prompt: Option<&str>) -> Result<()> {
    let profile_name = config.model_roles.get("default").unwrap_or(&config.default_model);
    let model_config = config.models.get(profile_name).context("resolved model profile does not exist")?;
    let client = build_client(model_config);
    let mut runtime = LocalToolRuntime::discover().await;
    let specs = runtime.tool_specs();
    let mode_text = Select::new("Permission mode:", vec!["default", "plan", "auto"])
        .with_starting_cursor(0).prompt().context("failed to choose permission mode")?;
    let mut permission_mode = match mode_text.as_str() {
        "plan" => PermissionMode::Plan,
        "auto" => PermissionMode::Auto,
        _ => PermissionMode::Default,
    };
    println!("[Permission mode: {}]", permission_mode.label());
    println!("[Tool pool: {} native, {} MCP]", 4, specs.len().saturating_sub(4));

    let mut messages: Vec<Value> = vec![json!({
        "role": "system",
        "content": format!(
            "You are a helpful coding agent working in {}. Use tools when useful. Tool permissions are enforced by the host. Never claim a tool succeeded unless its result confirms success.",
            std::env::current_dir()?.display()
        )
    })];

    let mut first = true;
    loop {
        let prompt = if first {
            first = false;
            if let Some(prompt) = initial_prompt { prompt.to_string() }
            else { Text::new("How can I help? (exit to quit, /mode to change permissions)").prompt().context("failed to read prompt")? }
        } else {
            Text::new("How can I help? (exit to quit, /mode to change permissions)").prompt().context("failed to read prompt")?
        };
        let prompt = prompt.trim();
        if prompt.is_empty() { if initial_prompt.is_some() { break; } else { continue; } }
        if matches!(prompt, "q" | "exit" | "exit()") { break; }
        if prompt == "/mode" {
            let selected = Select::new("Permission mode:", vec!["default", "plan", "auto"])
                .with_starting_cursor(0).prompt().context("failed to choose permission mode")?;
            permission_mode = match selected.as_str() {
                "plan" => PermissionMode::Plan,
                "auto" => PermissionMode::Auto,
                _ => PermissionMode::Default,
            };
            println!("[Permission mode: {}]", permission_mode.label());
            continue;
        }

        messages.push(json!({"role":"user","content":prompt}));
        let mut turns = 0usize;
        loop {
            turns += 1;
            if turns > 32 { bail!("agent exceeded 32 model/tool turns; stopping to avoid an infinite loop"); }
            let mut request_json = json!({
                "model": model_config.model,
                "messages": messages,
                "temperature": model_config.temperature,
                "tools": specs,
                "tool_choice": "auto"
            });
            if let Some(max_tokens) = model_config.max_tokens {
                request_json["max_tokens"] = json!(max_tokens);
            }
            let request = serde_json::from_value(request_json)
                .context("failed to construct OpenAI-compatible chat completion request")?;
            info!(profile = %profile_name, model = %model_config.model, "Sending agent request");
            let response = timeout(
                Duration::from_secs(model_config.request_timeout_secs),
                client.chat().completions().create(request),
            ).await.context("model request timed out")?
                .context("OpenAI-compatible API request failed")?;
            let choice = response.choices.first().context("model returned no choices")?;
            messages.push(serde_json::to_value(&choice.message)?);
            let calls = choice.message.tool_calls.clone().unwrap_or_default();
            if calls.is_empty() {
                if let Some(content) = &choice.message.content { println!("{content}"); }
                else { debug!("model returned neither text nor tool calls"); }
                break;
            }

            for call in calls {
                let name = call.function.name;
                let input: Value = serde_json::from_str(&call.function.arguments)
                    .with_context(|| format!("invalid JSON arguments for tool {name}"))?;
                let output = match permitted(permission_mode, &name, &input) {
                    Ok(true) => match runtime.invoke(&name, &input).await {
                        Ok(output) => output,
                        Err(error) => format!("Tool error: {error:#}"),
                    },
                    Ok(false) => "Permission denied by host.".to_string(),
                    Err(error) => format!("Permission prompt failed: {error:#}"),
                };
                println!("[tool:{name}] {}", output.chars().take(240).collect::<String>());
                messages.push(json!({"role":"tool","tool_call_id":call.id,"content":output}));
            }
        }
        if initial_prompt.is_some() { break; }
    }
    runtime.shutdown().await;
    Ok(())
}
