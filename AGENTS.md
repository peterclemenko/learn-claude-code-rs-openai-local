Markdown
<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **learn-claude-code-rs-openai-local** (469 symbols, 1111 relationships, 40 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST use GitNexus for all refactoring attempts.** Whenever extracting, splitting, moving, or renaming symbols across modules, run `gitnexus_context({name: "symbolName"})` and `gitnexus_impact` first, and execute renames via `gitnexus_rename` to preserve call-graph integrity.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER refactor or reorganize code across files without first mapping upstream/downstream call graphs via GitNexus.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource                                                           | Use for                                  |
| ------------------------------------------------------------------ | ---------------------------------------- |
| `gitnexus://repo/learn-claude-code-rs-openai-local/context`        | Codebase overview, check index freshness |
| `gitnexus://repo/learn-claude-code-rs-openai-local/clusters`       | All functional areas                     |
| `gitnexus://repo/learn-claude-code-rs-openai-local/processes`      | All execution flows                      |
| `gitnexus://repo/learn-claude-code-rs-openai-local/process/{name}` | Step-by-step execution trace             |

## CLI

| Task                                         | Read this skill file                                        |
| -------------------------------------------- | ----------------------------------------------------------- |
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md`       |
| Blast radius / "What breaks if I change X?"  | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?"             | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md`       |
| Rename / extract / split / refactor          | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md`     |
| Tools, resources, schema reference           | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md`           |
| Index, status, clean, wiki CLI commands      | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md`             |

<!-- gitnexus:end -->

---

# Core Agent Architectural & Engineering Directives

## 1. Mandatory Tooling & Verification Defaults

- **GitNexus Graph-First Refactoring & Impact Analysis:**
  - Treat the codebase as a directed dependency graph, not flat text files.
  - Before any edit, bug fix, or refactoring pass, map the exact blast radius with `gitnexus_impact` and execution flows with `gitnexus_query`.
  - Follow `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` for all structural refactors; never guess call sites or perform blind regex/find-and-replace edits.
- **Context7 Live API Documentation:**
  - **ALWAYS query Context7 (`resolve-library-id` -> `get-library-docs`) before writing, refactoring, or debugging code against third-party crates or SDKs** (e.g., `async-openai`, `opentelemetry`, `tracing`, `ron`, `tokio`, `grafeo`).
  - Never rely on stale pre-training memory for crate versions, feature flags, trait signatures, or builder patterns—verify exact APIs against `Cargo.toml` versions via Context7 first.
- **RON Configuration Paradigm:**
  - All agent state, tool bindings, subagent boundaries, and capability manifests **MUST** be serialized and parsed using **RON (Rusty Object Notation)**.
  - Avoid JSON or YAML for internal configuration specs to preserve Rust enum support, strict typing, trailing commas, and inline comments.
- **TOON Serialization:**
  - Use **TOON (Token-Oriented Object Notation)** by default for dense data representations, structured inter-agent payloads, tool response compression, and context compaction prompts to minimize token overhead while preserving exact schema constraints.
- **Default OpenTelemetry Instrumentation:**
  - Every agent turn, subagent delegation, tool invocation (including GitNexus and Context7 calls), and graph memory query **MUST** emit OpenTelemetry spans (`tracing` + `opentelemetry`) tagged with trace context, duration, token usage, and target symbol/crate metadata.
- **Exhaustive Documentation:**
  - Every public struct, trait, function, error variant, and RON configuration block **MUST** include comprehensive `///` rustdoc comments (covering arguments, return values, panics/errors, and architectural *why*) as well as structured `#[instrument]` attributes.

---

## 2. Standard `agent.ron` Specification Template

All agent definitions and subagent manifests must adhere to this heavily documented RON schema:

```ron
// ============================================================================
// Agent Configuration Specification (RON)
// Project: learn-claude-code-rs-openai-local
// ============================================================================

(
    // Unique identifier for the agent instance or persona
    agent_id: "default-local-operator",

    // Operational mandate and architectural boundaries
    description: "Local-first Rust agent harness enforcing GitNexus impact analysis, Context7 API verification, OTel tracing, and TOON serialization.",

    // Execution parameters and context compaction thresholds
    execution: (
        max_turns: 25,
        temperature: 0.2,
        context_window_tokens: 128000,
        enable_compaction: true,
        compaction_format: "toon",
    ),

    // Mandatory pre-flight verification gates before code modification
    verification_gates: (
        require_gitnexus_impact_before_edit: true,
        require_gitnexus_detect_changes_before_commit: true,
        require_context7_for_external_crates: true,
        block_on_critical_blast_radius_without_ack: true,
    ),

    // Observability & OpenTelemetry Configuration
    telemetry: (
        enabled: true,
        service_name: "learn-claude-code-rs-openai-local",
        exporter_endpoint: "[http://127.0.0.1:4317](http://127.0.0.1:4317)", // OTLP gRPC endpoint
        sample_ratio: 1.0,
        include_payload_attributes: true,
        record_token_metrics: true,
    ),

    // Serialized Data Formats
    serialization: (
        preferred_config_format: "ron",
        preferred_inter_agent_format: "toon",
    ),

    // Bound MCP Tools and Core Capabilities
    capabilities: [
        "file_system_operations",
        "gitnexus_code_intelligence",
        "context7_live_documentation",
        "grafeo_graph_memory",
        "otel_tracing_span",
    ],
)
3. OpenTelemetry & Documentation Rust Instrumentation Standard
All new or refactored Rust functions in this repository must follow this documentation and instrumentation pattern (verify exact crate signatures via Context7 before modifying Cargo.toml or imports):

Rust
use opentelemetry::{
    global,
    trace::{FutureExt, TraceContextExt, Tracer},
    Context, KeyValue,
};
use tracing::{info, instrument};

/// Executes a single instrumented agent turn against the local OpenAI-compatible endpoint.
///
/// # Architectural Invariants
/// - Emits an OpenTelemetry span (`agent_turn`) capturing token budgets and execution latency.
/// - Expects structured subagent handoffs and compacted context blocks in TOON format.
///
/// # Arguments
/// * `prompt` - The user or compacted TOON prompt payload for the current turn.
/// * `budget` - Maximum allowed token allocation for this execution step.
///
/// # Errors
/// Returns [`AgentError`] if endpoint inference fails, RON schema validation breaks,
/// or required pre-flight verification gates fail.
#[instrument(
    name = "agent.execute_turn",
    skip(prompt),
    fields(
        otel.kind = "internal",
        token.budget = %budget,
        serialization.format = "toon"
    )
)]
pub async fn execute_agent_turn(prompt: &str, budget: usize) -> Result<AgentResponse, AgentError> {
    let tracer = global::tracer("learn-claude-code-rs-openai-local");
    let span = tracer.start("agent_turn_execution");
    span.set_attribute(KeyValue::new("token.budget", budget as i64));

    let cx = Context::current_with_span(span);

    async move {
        info!(target: "agent::loop", "Executing instrumented agent turn");
        // Core execution logic here...
        Ok(AgentResponse::default())
    }
    .with_context(cx)
    .await
}
4. Mandatory Workflow Rules
TOON Data Exchange Rule:

When passing structured data between subagents, formatting dense symbol/impact reports, or running context compaction, serialize the payload using TOON syntax.

Ensure all field names map 1:1 to the underlying Rust structs and RON schemas.

Pre-Refactor & Pre-Edit Checklist:

Step 1 (Graph Scope): Run gitnexus_impact and gitnexus_context on target symbols. Report blast radius and risk level.

Step 2 (API Verification): Query Context7 for any external crates touched by the change to confirm exact method signatures and feature flags.

Step 3 (Implement with Instrumentation): Write code with full /// rustdoc comments, #[instrument] spans, RON config bindings, and TOON payload serialization.

Step 4 (Verify Scope): Run gitnexus_detect_changes() and cargo check / cargo test before finalizing.