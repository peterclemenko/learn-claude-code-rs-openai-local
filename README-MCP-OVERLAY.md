# MCP plugin + local OpenAI-compatible agent overlay

This is an overlay for the existing `learn-claude-code-rs-openai-local` repository.
It is not a standalone replacement for the repository's Nix/CI scaffolding.

## Apply

Extract this archive into the root of your existing checkout and allow it to overwrite
the listed files. Then run:

```bash
cargo fmt --all
cargo check --workspace
cargo test --workspace
```

The original `config.ron` format and model profiles are preserved. The agent continues
to use the configured OpenAI-compatible endpoint. MCP server commands are started as
child processes using the `rmcp` client transport.

## MCP manifest

The example `.claude-plugin/plugin.json` is deliberately empty. Add servers only after
reviewing their commands and arguments. The manifest's `env` values are passed to the
child process; do not commit secrets there.

## Permission model

- `default`: read-like tools are allowed; writes and shell commands ask first.
- `plan`: blocks write-capable tools.
- `auto`: allows ordinary writes but still asks before high-risk operations.
- `/mode` changes the mode for the current run.
- Native file tools restrict paths to the current working directory.

MCP tool names are exposed as `mcp__<plugin>__<server>__<tool>`.
Tool names are not a security boundary: server processes still have the operating-system
permissions of the user running this application.

## Important

This archive was assembled from the repository's public source and adapted source files.
It could not be compiled in this environment because network access to download Rust
dependencies was unavailable. Run the checks above locally before relying on it.
