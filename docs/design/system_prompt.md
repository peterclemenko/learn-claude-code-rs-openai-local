# Structured System Prompt Architecture

System prompt templates live in `config/prompts/*.ron`. Each template defines metadata and ordered sections:

- `identity`
- `instructions`
- `tool_guidelines`
- `safety_and_constraints`
- `custom_sections`
- template variables and placeholders

The available placeholders include `{workspace_path}`, `{operating_system}`, `{runtime}`, template variables, and render-context variables.

## Selection precedence

Prompt selection is resolved in this order:

1. Default: `config/prompts/default.ron`
2. Config: `active_prompt` in `config/prompts.ron`
3. CLI: `--prompt-name <name>`, `--prompt-template <name>`, or `--system-prompt <name>`

Later sources override earlier sources. Names may be `default` or `default.ron`, but selections cannot contain separators, absolute paths, `..`, or symlinks outside `config/prompts/`.

## Runtime API

The shared `PromptManager` is thread-safe and is shared by the agent loop and registered tools:

```rust
manager.select_prompt("concise", PromptChangeSource::ProgrammaticApi).await?;
manager.select_prompt_sync("coding", PromptChangeSource::CliFlag)?;
```

Supported sources are `Config`, `CliFlag`, `ProgrammaticApi`, and `AgentTool`. Agent turns render the static prompt and append TODO, memory, skill, and compaction content separately as TOON or existing dynamic-context formats.

## Agent tool

The `select_system_prompt` tool accepts JSON or TOON:

```text
{"action":"list"}
{"action":"switch","name":"concise"}
select_system_prompt[1]{action,name}:
switch,concise
```

Tool dispatch is gated by `PermissionManager` before `PromptManager::select_prompt(..., AgentTool)` runs.

## Strict permissions

The `strict` profile denies `select_system_prompt` with a wildcard deny rule. Trusted Rust callers using `Config`, `CliFlag`, or `ProgrammaticApi` bypass LLM tool permissions and may still switch the active prompt. `default`, `auto`, and `plan` use their existing permission rules and plan-mode state-change behavior.
