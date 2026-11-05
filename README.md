Openai local version of the harness from

https://github.com/wulawulu/learn-claude-code-rs/tree/main

this is not a tutorial

this is the harness with improvements

A template Rust project with fully functional and no-frills Nix support, as well as builtin VSCode configuration to get IDE experience without any manual setup (just [install direnv](https://nixos.asia/en/direnv), open in VSCode and accept the suggestions). It uses [crane](https://crane.dev/), via [rust-flake](https://github.com/juspay/rust-flake).

> [!NOTE]
> If you are looking for the original template based on [this blog post](https://srid.ca/rust-nix)'s use of `crate2nix`, browse from [this tag](https://github.com/srid/learn-claude-code-rs-openai-local/tree/crate2nix). The evolution of this template can be gleaned from [releases](https://github.com/srid/learn-claude-code-rs-openai-local/releases).

## Usage

You can use [omnix](https://omnix.page/om/init.html)[^omnix] to initialize this template:
```
nix run nixpkgs#omnix -- init github:srid/learn-claude-code-rs-openai-local -o ~/my-rust-project
```

Run the harness with `just run` (or `cargo run`) to start an interactive session.
Enter `exit()` to quit. To send a single prompt without starting the REPL, run
`cargo run -- --prompt "Your question"`.

The agent settings are loaded from [`config/agent.ron`](./config/agent.ron).
The default endpoint is Ollama's OpenAI-compatible API at
`http://localhost:11434/v1`; set `base_url` and `api_key` for another local
server such as vLLM or llama.cpp. Tools advertise TOON signatures and exchange
TOON results; OpenAI-compatible JSON tool arguments are also accepted.
OpenTelemetry spans are exported to the configured OTLP endpoint when telemetry
is enabled. Export is disabled by default; set `OTEL_SDK_DISABLED=false` to
enable it when a collector is available.
The agent also exposes `todo_write` for maintaining a task list in TOON format;
the latest task state is injected into the system context on every model turn.
Delegation is configured in the same RON file: `delegate_task` starts an isolated
child loop against the configured OpenAI-compatible endpoint, with per-persona
model, tool permissions, prompt template, and maximum depth. A caller supplies
the task and the parent context to pass to the child as TOON or JSON; nested
delegation is available only to child personas that allow `delegate_task`.
Each `max_depth` is an absolute nesting limit inherited by descendants; a child
cannot delegate beyond the strictest limit in its ancestry.
Subagent spans are nested under the delegating tool call and include delegation
identity, task, model, depth, status, result size, and turn count.
Tool spans record duration, success, input/output bytes, and rough token
estimates for every registered tool. Model request spans record exact
prompt/completion/total tokens when the backend reports them, plus prompt
utilization against the active Ollama context size from `/api/ps`. Per-tool
token counts are byte-based estimates, not tokenizer-exact counts. Configure
`ollama_base_url` for Ollama; other runtimes can implement the shared
`ContextWindowProvider` interface.

Run `cargo test --workspace` to exercise local mock subagent delegation,
tool permissions, RON/TOON serialization, and OpenTelemetry trace parenting
without requiring a running LLM service.

[^omnix]: If initializing manually, make sure to:
    - Change `name` in Cargo.toml.
    - Run `cargo generate-lockfile` in the nix shelld

## Adapting this template

- There are two CI workflows, and one of them uses Nix which is slower (unless you configure a cache) than the other one based on rustup. Pick one or the other depending on your trade-offs.

## Development (Flakes)

This repo uses [Flakes](https://nixos.asia/en/flakes) from the get-go.

```bash
# Dev shell
nix develop

# or run via cargo
nix develop -c cargo run

# build
nix build
```

We also provide a [`justfile`](https://just.systems/) for Makefile'esque commands to be run inside of the devShell.

## Tips

- Run `nix flake update` to update all flake inputs.
- Run `nix --accept-flake-config run github:juspay/omnix ci` to build _all_ outputs.
- [pre-commit] hooks will automatically be setup in Nix shell. You can also run `pre-commit run -a` manually to run the hooks (e.g.: to autoformat the project tree using `rustfmt`, `nixpkgs-fmt`, etc.).

## Discussion

- [Zulip](https://nixos.zulipchat.com/#narrow/stream/413950-nix)

## See Also

- [nixos.wiki: Packaging Rust projects with nix](https://nixos.wiki/wiki/Rust#Packaging_Rust_projects_with_nix)
