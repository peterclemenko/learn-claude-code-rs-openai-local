use std::{process::Stdio, time::Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::{io::AsyncWriteExt, process::Command};
use tracing::{error, info, instrument, warn};

use super::config::{HookBackend, HookPayloadFormat, HookRule};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) struct RecoveryDiagnostics {
    pub(crate) error_kind: String,
    pub(crate) recoverable: bool,
    pub(crate) recovery_action: String,
    pub(crate) attempt: u32,
    pub(crate) outcome: String,
    pub(crate) message: String,
    pub(crate) context_toon: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) struct HookContext {
    pub(crate) event: String,
    #[serde(default)]
    pub(crate) tool_name: Option<String>,
    #[serde(default)]
    pub(crate) tool_input: Option<String>,
    #[serde(default)]
    pub(crate) tool_output: Option<String>,
    #[serde(default)]
    pub(crate) prompt: Option<String>,
    #[serde(default)]
    pub(crate) subagent_name: Option<String>,
    #[serde(default)]
    pub(crate) recovery: Option<RecoveryDiagnostics>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq, Eq)]
pub(crate) struct HookResponse {
    #[serde(default)]
    pub(crate) decision: Option<String>,
    #[serde(default)]
    pub(crate) reason: Option<String>,
    #[serde(default)]
    pub(crate) modified_arguments: Option<String>,
    #[serde(default)]
    pub(crate) modified_prompt: Option<String>,
    #[serde(default)]
    pub(crate) reminder: Option<String>,
}

#[derive(Debug)]
pub(crate) struct ExternalHookOutput {
    pub(crate) exit_code: i64,
    pub(crate) stdout: String,
    pub(crate) stderr: String,
    pub(crate) duration_ms: u64,
    pub(crate) timed_out: bool,
}

pub(crate) struct HookRunner;

impl HookRunner {
    #[instrument(
        name = "hook.run_external",
        skip(rule, payload),
        fields(
            hook.id = %rule.id,
            hook.name = %rule.name,
            hook.exit_code = tracing::field::Empty,
            hook.duration_ms = tracing::field::Empty,
            hook.timed_out = tracing::field::Empty,
            hook.stdout = tracing::field::Empty,
            hook.stderr = tracing::field::Empty,
        )
    )]
    pub(crate) async fn run_external(
        rule: &HookRule,
        backend: &HookBackend,
        payload: &HookContext,
    ) -> Result<(ExternalHookOutput, HookResponse)> {
        let HookBackend::External {
            command,
            args,
            payload_format,
        } = backend
        else {
            anyhow::bail!("Runner expected an external hook backend");
        };

        let payload_bytes = match payload_format {
            HookPayloadFormat::Json => {
                serde_json::to_vec(payload).context("Failed to serialize hook JSON payload")?
            }
            HookPayloadFormat::Ron => ron::ser::to_string(payload)
                .context("Failed to serialize hook RON payload")?
                .into_bytes(),
        };

        let mut command = Command::new(command);
        command
            .args(args)
            .kill_on_drop(true)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let traceparent = current_traceparent();
        if let Some(traceparent) = traceparent {
            command.env("TRACEPARENT", traceparent);
        }

        let started = Instant::now();
        let mut child = command
            .spawn()
            .context("Failed to spawn external hook process")?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(&payload_bytes)
                .await
                .context("Failed to write hook payload")?;
        }

        let wait = child.wait_with_output();
        match tokio::time::timeout(rule.timeout(), wait).await {
            Ok(result) => {
                let output = result.context("Failed to wait for external hook")?;
                let duration_ms = started.elapsed().as_millis().try_into().unwrap_or(u64::MAX);
                let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
                let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
                let exit_code = output.status.code().map(i32::from).unwrap_or(-1) as i64;
                let span = tracing::Span::current();
                span.record("hook.exit_code", exit_code);
                span.record("hook.duration_ms", duration_ms);
                span.record("hook.timed_out", false);
                span.record("hook.stdout", stdout.as_str());
                span.record("hook.stderr", stderr.as_str());
                let response = parse_hook_response(payload_format, &stdout)
                    .unwrap_or_else(|error| {
                        warn!(hook.id = %rule.id, error = %error, "External hook returned invalid response; defaulting to noop");
                        HookResponse::default()
                    });
                info!(
                    hook.id = %rule.id,
                    hook.exit_code = exit_code,
                    hook.duration_ms = duration_ms,
                    hook.stdout = %stdout,
                    hook.stderr = %stderr,
                    "External hook completed"
                );
                Ok((
                    ExternalHookOutput {
                        exit_code,
                        stdout,
                        stderr,
                        duration_ms,
                        timed_out: false,
                    },
                    response,
                ))
            }
            Err(_) => {
                let duration_ms = started.elapsed().as_millis().try_into().unwrap_or(u64::MAX);
                let span = tracing::Span::current();
                span.record("hook.exit_code", -1i64);
                span.record("hook.duration_ms", duration_ms);
                span.record("hook.timed_out", true);
                error!(
                    hook.id = %rule.id,
                    hook.exit_code = -1i64,
                    hook.timeout_ms = rule.timeout_ms,
                    hook.duration_ms = duration_ms,
                    hook.timed_out = true,
                    "External hook timed out"
                );
                Ok((
                    ExternalHookOutput {
                        exit_code: -1,
                        stdout: String::new(),
                        stderr: String::new(),
                        duration_ms,
                        timed_out: true,
                    },
                    HookResponse::default(),
                ))
            }
        }
    }
}

fn parse_hook_response(format: &HookPayloadFormat, stdout: &str) -> Result<HookResponse> {
    match format {
        HookPayloadFormat::Json => {
            serde_json::from_str(stdout).context("Invalid hook JSON response")
        }
        HookPayloadFormat::Ron => ron::from_str(stdout).context("Invalid hook RON response"),
    }
}

fn current_traceparent() -> Option<String> {
    use opentelemetry::trace::TraceContextExt;
    use tracing_opentelemetry::OpenTelemetrySpanExt;

    let context = tracing::Span::current().context();
    let span = context.span();
    let span_context = span.span_context();
    if !span_context.is_valid() {
        return None;
    }
    Some(format!(
        "00-{}-{}-{:02x}",
        span_context.trace_id(),
        span_context.span_id(),
        if span_context.is_sampled() { 1 } else { 0 }
    ))
}
