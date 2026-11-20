# Durable Memory System

## Architecture

The memory feature is isolated under `learn-claude-code-rs-openai-local/src/memory/`
and is exposed to the agent through the `MemoryBackend` trait. The agent loop does
not know how a backend stores or searches records; it receives an
`Arc<dyn MemoryBackend>` in `AgentRuntime`. Subagents clone the same runtime, so
they share the selected backend and durable store.

The built-in `naive_memory_system` backend stores RON-serialized `MemoryRecord` values in
a local file. It supports lexical recall, optional scope/category filtering,
category weights, and the durable categories `Preference`, `Fact`, `Feedback`,
and `Reference`. Writes use a temporary file followed by rename, and in-memory
state is updated only after the new store is persisted.

`mempalace_stoolap` stores verbatim drawer records in embedded Stoolap tables
with lexical fallback and optional native vector recall. Its hierarchy maps
entry scope and metadata to wings, rooms, and halls; compact closet index rows
provide a bounded wake-up layer before the verbatim memory TOON table.
`noop` provides an empty backend for isolated runs.

`MemoryRegistry` maps configured backend names to builder functions. Its built-in
registry includes `naive_memory_system`, `mempalace_stoolap`, and `noop`;
`register_memory_backend` adds a custom implementation to the process registry
before `Agent::new` builds the selected backend. All backends constructed by the
registry are wrapped with common OpenTelemetry instrumentation.

## Domain model and operations

- `MemoryEntry` contains a category, content, tags, optional scope, and string
  metadata.
- `MemoryRecord` adds a stable ID and RFC 3339 creation/update timestamps.
- `MemoryQuery` contains search text, optional category/scope filters, and an
  optional limit.
- `MemoryRecallResult` retains the query and ranked records.
- `MemoryBackend` defines `recall`, `remember`, `forget`, `list`,
  `format_for_context`, and `health_check`.

The `naive_memory_system` backend scores query tokens against entry content, tags, scope,
and metadata. Category weights multiply the lexical score; matching records
are ordered by score and then update time. An empty query returns the newest
eligible records up to the configured recall limit.

At each agent turn, the latest user text is recalled and formatted before the
chat request. Non-empty results are injected into the system message as a TOON
table. Recall or formatting errors fail the turn explicitly; they are not
silently treated as an empty result.

## MemPalace Stoolap hierarchy and retrieval

The backend maps the existing `MemoryEntry` shape without changing the agent
contract:

- **Wings** are top-level project/domain/person scopes. `metadata["wing"]`
  overrides the entry scope; otherwise scope is used, then the configured
  `default_wing`.
- **Rooms** are topics within a wing, selected by `metadata["room"]` or the
  configured `default_room`.
- **Halls** are memory corridors. `metadata["hall"]` may select one explicitly;
  otherwise `Preference` maps to `preferences`, `Fact` to `facts`, `Feedback`
  to `advice`, and `Reference` to `discoveries`.
- **Drawers** preserve the original content, category, tags, scope, metadata,
  and timestamps. The database never summarizes or rewrites drawer text.
- **Closets** hold bounded index metadata (hall and tag names) per wing/room.
  `format_for_context` emits those wake-up rows as TOON before the normal
  verbatim memory table, limited by `wake_up_limit`.

The embedded schema uses relational `wings`, `rooms`, `halls`, `closets`, and
`drawers` tables plus `palace_settings` for schema/vector compatibility checks.
`drawers.embedding` is Stoolap's configured `VECTOR(N)` column with an HNSW
index using the configured cosine or L2 metric. Persistent mode opens a
`file://` database with full WAL synchronization by default; memory mode uses a
fresh `Database::open_in_memory()` instance. Schema and indexes are initialized
idempotently, and reopening an existing database with different vector
dimensions or metric returns an explicit configuration error.

An application can implement the public `MemoryEmbeddingProvider` trait and
pass it to `MemPalaceStoolapMemory::open_with_embedding_provider`. The provider
dimension must match `vector_dimensions`; returned values are checked for
dimension and finite numbers before storage or search. The default registry
does not require or load an embedding model. When no provider or query embedding
is available, recall uses lexical matches over content/tags/scope/metadata,
category weights, configured scope/category filters, and newest-first ordering
for empty queries. Thus offline operation and config-only backend switching do
not require model downloads or network access.

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

## Configuration and validation

The active RON configuration is [`config/memory.ron`](../../../config/memory.ron).
Its built-in backend IDs are `mempalace_stoolap`, `naive_memory_system`, and
`noop`. Custom IDs are available after registration through
`register_memory_backend` or `MemoryRegistry::register`.

The common configuration fields are:

| Field | Constraints and defaults |
| --- | --- |
| `backend` | Required registered identifier. |
| `storage_dir` | Required, non-empty path; relative paths resolve from the RON file. Default: `.local/memory`. |
| `recall_limit` | Maximum recall/context rows; integer range `1..=i64::MAX`. Default: `5`. |
| `enabled_categories` | Unique built-in categories; defaults to `Preference`, `Fact`, `Feedback`, and `Reference`. |
| `category_weights` | Optional known category names with finite, non-negative multipliers; omitted categories use `1.0`. Defaults: `Preference=2.0`, `Fact=1.0`, `Feedback=1.5`, `Reference=0.75`. |
| `backend_options` | String map validated against the selected backend's allowlist; default is empty. |
| `mempalace_stoolap` | Typed block with strict RON field names and types; defaults are listed below. |

MemPalace settings are only validated as backend-specific settings when
`backend` is `mempalace_stoolap`. `storage_mode` is `persistent` or `memory`;
`database_path` is non-empty in persistent mode and can be relative to
`storage_dir`, absolute, or a Stoolap `file://` URI. `vector_dimensions` must be
positive, `vector_metric` is `cosine` or `l2`, `default_wing` and `default_room`
must be non-empty, and `wake_up_limit` must be positive. Defaults are
`persistent`, `mempalace-db`, `384`, `cosine`, `project`, `general`, and `8`,
respectively. RON rejects unknown fields and type mismatches in this typed
block; the top-level config also rejects unknown fields.

The built-in backend option allowlists are strict:

| Backend | Allowed `backend_options` keys |
| --- | --- |
| `naive_memory_system` | `file_name` (plain filename only; default `memories.ron`) |
| `mempalace_stoolap` | none |
| `noop` | none |

For example, `backend_options: {"file_nam": "memories.ron"}` fails before the
backend is constructed and reports both the invalid key and valid options.
Unknown backend IDs fail with a sorted list of all registered IDs, such as
`["mempalace_stoolap", "naive_memory_system", "noop"]`. The public
`MemoryRegistry::available_backends()` method returns that deterministic list.
Custom backends registered with `register` accept arbitrary string options;
use `register_with_validator` or `register_memory_backend_with_validator` to
reject unknown options for custom schemas.

`MemorySystemConfig::default()` remains independent of the checked-in active
selection: it chooses `naive_memory_system`, uses an empty options map (which
selects the default `memories.ron` file), and validates without relying on the
current `config/memory.ron`. The checked-in-config test builds whichever
registered backend is selected there, and separate RON tests parse and build
each built-in.

For routine switching, keep `backend_options` empty and change only `backend`
in `config/memory.ron`:

```ron
backend: "mempalace_stoolap", // or "naive_memory_system" or "noop"
```

The active checked-in example uses MemPalace with persistent Stoolap storage.
The RON file includes commented ready-to-use examples for each built-in.
Changing to `naive_memory_system` selects the RON-file backend; changing to
`noop` disables persistence. All backends retain the same `MemoryBackend`
contract and are constructed by the registry.

Configuration loading and validation, backend resolution, and construction
errors are recorded through tracing/OpenTelemetry spans and error events.
Registry validation occurs before builder invocation, so invalid options fail
before opening a database or writing a memory file.

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
   process. Do not register the same name twice. Use the validator variants
   when the custom backend needs strict option validation.
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

## Conformance and stress testing

The reusable test-only `run_backend_conformance_suite` helper accepts an
`Arc<dyn MemoryBackend>`. It checks CRUD and health behavior, all four memory
categories, metadata/tags/scope preservation, result limits and filtering,
TOON output for empty and populated results, large entries, and durable
rehydration. The same contract is run against `naive_memory_system`,
`mempalace_stoolap`, and a dynamically registered test backend. Backend
selection tests switch between both durable built-ins, `noop`, and a custom ID
using generated RON configurations and isolated temporary directories.

To validate a custom backend, add its builder to `custom_registry()` in
`src/memory/conformance_tests.rs`, construct it through `MemoryRegistry` from
test RON, and invoke `run_backend_conformance_suite` with its trait object.
This keeps the assertions independent of the concrete backend while retaining
backend-specific tests for persistence or service behavior.

Run the conformance and stress coverage with:

```sh
cargo test --workspace memory::conformance_tests -- --nocapture
```

The suite includes concurrent readers and writers and verifies that corrupted
storage reports an error without panicking. OpenTelemetry-focused tests use an
in-memory exporter and an OTel-only subscriber; run them with the repository
target:

```sh
just test-otel
```

The `just test` and `just test-otel` recipes run the Rust test harness
single-threaded because tests temporarily install per-test tracing dispatchers;
this avoids nondeterministic exporter assertions from parallel subscriber
changes. Production execution is unaffected.

The ignored manual benchmark records OTel measurements for independent
`remember` batches of 100, 1,000, and 5,000 entries, plus per-write latency at
corpus sizes 50, 500, and 5,000, cold/warm recall, small/large listing, and
forget. Measurements are emitted as structured tracing events and benchmark
span attributes—never terminal output. Since the naive backend persists the
full RON store on each write, the 5,000-entry tier can take substantially
longer than the regular test suite. Run it explicitly when comparing backends:

```sh
cargo test --workspace memory_benchmark_operation_latency_and_corpus_scaling \
  -- --ignored --nocapture
```