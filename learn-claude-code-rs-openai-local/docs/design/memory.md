# Durable Memory System

## Architecture

The memory feature is isolated under `learn-claude-code-rs-openai-local/src/memory/`
and is exposed to the agent through the `MemoryBackend` trait. The agent loop does
not know how a backend stores or searches records; it receives an
`Arc<dyn MemoryBackend>` in `AgentRuntime`. Subagents clone the same runtime, so
they share the selected backend and durable store.

The built-in `naivememorysystem` backend stores RON-serialized `MemoryRecord` values in
a local file. It supports lexical recall, optional scope/category filtering,
category weights, and the durable categories `Preference`, `Fact`, `Feedback`,
and `Reference`. Writes use a temporary file followed by rename, and in-memory
state is updated only after the new store is persisted. `noop` provides an empty
backend for isolated runs.

`MemoryRegistry` maps configured backend names to builder functions. Its built-in
registry includes `naivememorysystem` and `noop`; `register_memory_backend` adds a
custom implementation to the process registry before `Agent::new` builds the
selected backend. All backends constructed by the registry are wrapped with
common OpenTelemetry instrumentation.

## Domain model and operations

- `MemoryEntry` contains a category, content, tags, optional scope, and string
  metadata.
- `MemoryRecord` adds a stable ID and RFC 3339 creation/update timestamps.
- `MemoryQuery` contains search text, optional category/scope filters, and an
  optional limit.
- `MemoryRecallResult` retains the query and ranked records.
- `MemoryBackend` defines `recall`, `remember`, `forget`, `list`,
  `format_for_context`, and `health_check`.

The `naivememorysystem` backend scores query tokens against entry content, tags, scope,
and metadata. Category weights multiply the lexical score; matching records
are ordered by score and then update time. An empty query returns the newest
eligible records up to the configured recall limit.

At each agent turn, the latest user text is recalled and formatted before the
chat request. Non-empty results are injected into the system message as a TOON
table. Recall or formatting errors fail the turn explicitly; they are not
silently treated as an empty result.

The `memory` tool exposes `recall`, `remember`, `forget`, `list`, and
`health_check`. It is registered through the normal tool registry, so
PreToolUse/PostToolUse hooks and permission checks still run. Its permission
configuration defaults to `Ask`; operators can change that policy in
`config/permissions/memory.ron`.

Example JSON tool input:

```json
{"action":"remember","category":"Preference","content":"Prefer concise Rust examples","tags":["style"],"scope":"project"}
```

Equivalent TOON input uses the declared signature
`memory[1]{action,category,content,tags,scope,metadata,query,id,limit}:`.
Recall and list results use a TOON table with `id`, `category`, `content`,
`tags`, `scope`, `metadata`, and `created_at` columns.

## Configuration and defaults

The active configuration is `config/memory.ron`:

```ron
(
    backend: "naivememorysystem",
    storage_dir: "../.local/memory",
    recall_limit: 5,
    enabled_categories: [Preference, Fact, Feedback, Reference],
    category_weights: {
        "Preference": 2.0,
        "Fact": 1.0,
        "Feedback": 1.5,
        "Reference": 0.75,
    },
    backend_options: {
        "file_name": "memories.ron",
    },
)
```

Relative `storage_dir` values resolve from the configuration file.
`naivememorysystem` uses `backend_options["file_name"]` (default `memories.ron`) and rejects
absolute paths or path traversal. `backend_options` is a string map reserved for
backend-specific settings. Unknown backend IDs and invalid configuration are
reported as errors. If the config file is absent, validated defaults select
`naivememorysystem` and use `.local/memory`.

`enabled_categories` defaults to all four built-in categories and controls both
recall eligibility and whether new entries can be remembered. `category_weights`
defaults to preference 2.0, fact 1.0, feedback 1.5, and reference 0.75; omitted
category weights are neutral (1.0).

## Adding a backend

1. Implement `MemoryBackend` with `Send + Sync` and its async operations. Keep
   backend-specific configuration in `backend_options` or add typed config
   fields if the format becomes stable.
2. Register a builder before `Agent::new` constructs the runtime:

   ```rust
   register_memory_backend("custom", |config| {
       let endpoint = config
           .backend_options
           .get("endpoint")
           .context("custom memory backend requires an endpoint")?;
       Ok(Arc::new(CustomMemoryBackend::new(endpoint.clone())?))
   })?;
   ```

   The closure returns `Arc<dyn MemoryBackend>` and is registered once per
   process. Do not register the same name twice.
3. Select it in RON without changing the agent loop:

   ```ron
   (
       backend: "custom",
       storage_dir: "../.local/memory",
       recall_limit: 5,
       enabled_categories: [Preference, Fact, Feedback, Reference],
       category_weights: {},
       backend_options: {"endpoint": "http://127.0.0.1:9000"},
   )
   ```

The common registry wrapper instruments custom backend operations and formats
recall context as TOON by default. Override `format_for_context` only when a
backend requires a different, still prompt-safe representation.

## OpenTelemetry

Memory code emits spans and events through `tracing`; it does not install a
terminal formatting layer or write diagnostics to stdout/stderr. The application
telemetry bootstrap remains responsible for bridging tracing to OpenTelemetry
and exporting spans.

Backend construction and operations record:

- `memory.backend` — selected backend name.
- `memory.operation` — `recall`, `remember`, `forget`, `list`,
  `format_for_context`, or `health_check`.
- `memory.entry_count` — records returned or affected.
- `memory.duration_ms` — elapsed operation time.
- `memory.outcome` — `ok`, `error`, or `default` for absent configuration.

Failures are emitted as structured tracing error events and propagated to the
caller. The `otel_memory_tests` module uses an in-memory span exporter and a
subscriber containing only the OpenTelemetry layer; it verifies operation
attributes and provider shutdown without console logging.