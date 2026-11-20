# Error recovery

The error recovery engine classifies provider and tool failures, selects a bounded
recovery action, and reports each transition through the hook dispatcher. Recovery
diagnostics are also recorded on OpenTelemetry spans; the engine does not write
diagnostics to stdout or stderr.

## Configuration

`config/error_recovery.ron` controls recovery:

```ron
(
    enabled: true,
    hooks_enabled: true,
    provider: (
        max_retries: 2,
        base_delay_ms: 100,
        max_delay_ms: 2000,
        jitter_percent: 20,
    ),
    tool: (
        max_attempts: 2,
    ),
    parse: (
        max_attempts: 2,
    ),
)
```

If the file is absent, the same defaults are used. Invalid RON or invalid policy
limits are returned as startup errors. Provider retry and correction budgets are
capped at ten; provider delay is capped at one minute. Backoff doubles up to the
configured maximum and applies bounded jitter.

Provider timeouts and transient connection, rate-limit, and server failures are
retried within the configured budget. Other provider failures are escalated.
Tool and parse failures are returned to the model as TOON `tool_error` and
`recovery_context` tables for self-correction. Repeated failures of the same tool
exhaust its correction budget and abort the current turn. Successful executions
reset that tool's consecutive-failure count. Permission checks and pre-tool hooks
run before tool execution and are never bypassed by recovery.

## Hook flow

The `Error` hook event carries the classification and diagnostic. The `Recovery`
event carries the selected action, attempt, outcome, and TOON context. Tool
failures additionally retain the existing `PostToolUse` event with the recovery
diagnostics attached. Add rules for `Error` or `Recovery` to `config/hooks.ron`
to invoke configured native or external hooks. `hooks_enabled` controls these
dedicated recovery notifications; OpenTelemetry spans remain available when
they are disabled.

Provider retries happen in the agent request path. Tool failures are not replayed
automatically, since a tool may have side effects; instead, the tool error is
returned to the model for bounded self-correction. Once a budget is exhausted,
the error is escalated to the caller.

## OpenTelemetry attributes

The `error.recovery` span and associated `hook.dispatch` spans record:

| Attribute | Meaning |
| --- | --- |
| `error.kind` | `provider`, `tool`, `parse`, `timeout`, or `fatal` |
| `error.recoverable` | Whether the classification supports a recovery attempt |
| `recovery.action` | `retry`, `self_correct`, `fallback`, or `escalate` |
| `recovery.attempt` | One-based failure attempt |
| `recovery.outcome` | `retrying`, `feedback_injected`, `fallback_returned`, `escalated`, or `recovered` |

Error and recovery events are emitted with `tracing` under these spans, so the
existing OpenTelemetry subscriber exports the diagnostics uniformly with the
hook lifecycle.
