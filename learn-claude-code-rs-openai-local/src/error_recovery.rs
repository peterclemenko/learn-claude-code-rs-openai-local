use std::{
    fs,
    path::Path,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use serde::Deserialize;
use tracing::{error, info, instrument, warn};

use crate::hooks::{HookContext, HookDispatcher, HookEvent, RecoveryDiagnostics};

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct ErrorRecoveryConfig {
    pub(crate) enabled: bool,
    pub(crate) hooks_enabled: bool,
    pub(crate) provider: ProviderRecoveryPolicy,
    pub(crate) tool: CorrectionPolicy,
    pub(crate) parse: CorrectionPolicy,
}

impl ErrorRecoveryConfig {
    pub(crate) fn load_or_default(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let contents = match fs::read_to_string(path) {
            Ok(contents) => contents,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self::default());
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("Failed to read error recovery config {}", path.display())
                });
            }
        };
        let config: Self = ron::from_str(&contents)
            .with_context(|| format!("Failed to parse error recovery RON {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.provider.max_retries <= 10,
            "Error recovery provider.max_retries must not exceed 10"
        );
        anyhow::ensure!(
            self.provider.base_delay_ms <= self.provider.max_delay_ms,
            "Error recovery provider.base_delay_ms must not exceed max_delay_ms"
        );
        anyhow::ensure!(
            self.provider.max_delay_ms <= 60_000,
            "Error recovery provider.max_delay_ms must not exceed 60000"
        );
        anyhow::ensure!(
            self.provider.jitter_percent <= 100,
            "Error recovery provider.jitter_percent must not exceed 100"
        );
        anyhow::ensure!(
            self.tool.max_attempts <= 10,
            "Error recovery tool.max_attempts must not exceed 10"
        );
        anyhow::ensure!(
            self.parse.max_attempts <= 10,
            "Error recovery parse.max_attempts must not exceed 10"
        );
        Ok(())
    }
}

impl Default for ErrorRecoveryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            hooks_enabled: true,
            provider: ProviderRecoveryPolicy::default(),
            tool: CorrectionPolicy::default(),
            parse: CorrectionPolicy::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct ProviderRecoveryPolicy {
    pub(crate) max_retries: u32,
    pub(crate) base_delay_ms: u64,
    pub(crate) max_delay_ms: u64,
    pub(crate) jitter_percent: u8,
}

impl Default for ProviderRecoveryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 2,
            base_delay_ms: 100,
            max_delay_ms: 2_000,
            jitter_percent: 20,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct CorrectionPolicy {
    pub(crate) max_attempts: u32,
}

impl Default for CorrectionPolicy {
    fn default() -> Self {
        Self { max_attempts: 2 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ErrorKind {
    Provider,
    Tool,
    Parse,
    Timeout,
    Fatal,
}

impl ErrorKind {
    pub(crate) fn is_recoverable(self) -> bool {
        matches!(
            self,
            Self::Provider | Self::Tool | Self::Parse | Self::Timeout
        )
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Provider => "provider",
            Self::Tool => "tool",
            Self::Parse => "parse",
            Self::Timeout => "timeout",
            Self::Fatal => "fatal",
        }
    }

    pub(crate) fn classify_provider_error(error: &anyhow::Error) -> Self {
        let message = error
            .chain()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" ")
            .to_ascii_lowercase();
        if message.contains("timeout") || message.contains("timed out") {
            Self::Timeout
        } else if [
            "connection",
            "temporarily unavailable",
            "rate limit",
            "429",
            "500",
            "502",
            "503",
            "504",
        ]
        .iter()
        .any(|marker| message.contains(marker))
        {
            Self::Provider
        } else {
            Self::Fatal
        }
    }

    pub(crate) fn classify_tool_error(message: &str) -> Self {
        let message = message.to_ascii_lowercase();
        if [
            "parse",
            "deserialize",
            "serialization",
            "invalid format",
            "malformed",
        ]
        .iter()
        .any(|marker| message.contains(marker))
        {
            Self::Parse
        } else {
            Self::Tool
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecoveryAction {
    Retry,
    SelfCorrect,
    Fallback,
    Escalate,
}

impl RecoveryAction {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Retry => "retry",
            Self::SelfCorrect => "self_correct",
            Self::Fallback => "fallback",
            Self::Escalate => "escalate",
        }
    }

    pub(crate) fn outcome(self) -> &'static str {
        match self {
            Self::Retry => "retrying",
            Self::SelfCorrect => "feedback_injected",
            Self::Fallback => "fallback_returned",
            Self::Escalate => "escalated",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ErrorRecoveryEngine {
    config: ErrorRecoveryConfig,
}

impl ErrorRecoveryEngine {
    pub(crate) fn new(config: ErrorRecoveryConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    pub(crate) fn provider_action(&self, kind: ErrorKind, attempt: u32) -> RecoveryAction {
        if self.config.enabled
            && matches!(kind, ErrorKind::Provider | ErrorKind::Timeout)
            && attempt <= self.config.provider.max_retries
        {
            RecoveryAction::Retry
        } else {
            RecoveryAction::Escalate
        }
    }

    pub(crate) fn tool_action(&self, kind: ErrorKind, attempt: u32) -> RecoveryAction {
        if !self.config.enabled {
            return RecoveryAction::Fallback;
        }
        let policy = match kind {
            ErrorKind::Tool => &self.config.tool,
            ErrorKind::Parse => &self.config.parse,
            _ => return RecoveryAction::Escalate,
        };
        if attempt <= policy.max_attempts {
            RecoveryAction::SelfCorrect
        } else {
            RecoveryAction::Escalate
        }
    }

    pub(crate) fn retry_delay(&self, attempt: u32) -> Duration {
        let policy = &self.config.provider;
        let exponent = 1u64
            .checked_shl(attempt.saturating_sub(1).min(63))
            .unwrap_or(u64::MAX);
        let base = policy
            .base_delay_ms
            .saturating_mul(exponent)
            .min(policy.max_delay_ms);
        let jitter_range = base.saturating_mul(u64::from(policy.jitter_percent)) / 100;
        let jitter_span = jitter_range.saturating_mul(2).saturating_add(1);
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let jitter = i128::from(seed % jitter_span) - i128::from(jitter_range);
        let delay = (i128::from(base) + jitter).clamp(0, i128::from(policy.max_delay_ms)) as u64;
        Duration::from_millis(delay)
    }

    #[instrument(
        name = "error.recovery",
        skip(self, hooks, diagnostic),
        fields(
            error.kind = %diagnostic.error_kind,
            error.recoverable = diagnostic.recoverable,
            recovery.action = %diagnostic.recovery_action,
            recovery.attempt = diagnostic.attempt as i64,
            recovery.outcome = %diagnostic.outcome,
        )
    )]
    pub(crate) async fn notify(
        &self,
        hooks: &HookDispatcher,
        event: HookEvent,
        tool_name: Option<String>,
        diagnostic: RecoveryDiagnostics,
    ) {
        let span = tracing::Span::current();
        if self.config.hooks_enabled {
            hooks
                .dispatch(HookContext {
                    event: event.as_name().to_owned(),
                    tool_name,
                    tool_input: None,
                    tool_output: None,
                    prompt: None,
                    subagent_name: None,
                    recovery: Some(diagnostic.clone()),
                })
                .await;
        }

        match event {
            HookEvent::Error => error!(
                parent: &span,
                error.kind = %diagnostic.error_kind,
                error.recoverable = diagnostic.recoverable,
                recovery.action = %diagnostic.recovery_action,
                recovery.attempt = diagnostic.attempt,
                recovery.outcome = %diagnostic.outcome,
                error.message = %diagnostic.message,
                context = %diagnostic.context_toon,
                "Error recovery classified failure"
            ),
            HookEvent::Recovery
                if diagnostic.recovery_action == RecoveryAction::Escalate.as_str() =>
            {
                warn!(
                    parent: &span,
                    error.kind = %diagnostic.error_kind,
                    recovery.action = %diagnostic.recovery_action,
                    recovery.attempt = diagnostic.attempt,
                    recovery.outcome = %diagnostic.outcome,
                    "Error recovery escalated failure"
                )
            }
            HookEvent::Recovery => info!(
                parent: &span,
                error.kind = %diagnostic.error_kind,
                recovery.action = %diagnostic.recovery_action,
                recovery.attempt = diagnostic.attempt,
                recovery.outcome = %diagnostic.outcome,
                "Error recovery transition completed"
            ),
            _ => {}
        }
    }
}

impl Default for ErrorRecoveryEngine {
    fn default() -> Self {
        Self {
            config: ErrorRecoveryConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use opentelemetry::Value;
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider, SpanData};
    use tracing::instrument::WithSubscriber;
    use tracing_subscriber::prelude::*;

    use super::*;
    use crate::hooks::{HookConfig, HookDispatcher};

    fn attribute<'a>(span: &'a SpanData, name: &str) -> Option<&'a Value> {
        span.attributes
            .iter()
            .find(|attribute| attribute.key.as_str() == name)
            .map(|attribute| &attribute.value)
    }

    #[test]
    fn classifies_transient_and_fatal_provider_errors_and_bounds_retries() {
        let transient = anyhow::anyhow!("request timed out");
        let fatal = anyhow::anyhow!("provider rejected the API key");
        let provider_unavailable = anyhow::anyhow!("HTTP 503 service unavailable");
        assert_eq!(
            ErrorKind::classify_provider_error(&transient),
            ErrorKind::Timeout
        );
        assert_eq!(
            ErrorKind::classify_provider_error(&provider_unavailable),
            ErrorKind::Provider
        );
        assert_eq!(ErrorKind::classify_provider_error(&fatal), ErrorKind::Fatal);
        assert_eq!(
            ErrorKind::classify_tool_error("Failed to parse malformed arguments"),
            ErrorKind::Parse
        );

        let engine = ErrorRecoveryEngine::default();
        assert_eq!(
            engine.provider_action(ErrorKind::Timeout, 1),
            RecoveryAction::Retry
        );
        assert_eq!(
            engine.provider_action(ErrorKind::Timeout, 2),
            RecoveryAction::Retry
        );
        assert_eq!(
            engine.provider_action(ErrorKind::Timeout, 3),
            RecoveryAction::Escalate
        );
        assert_eq!(
            engine.provider_action(ErrorKind::Fatal, 1),
            RecoveryAction::Escalate
        );
    }

    #[test]
    fn missing_config_uses_defaults_and_invalid_limits_are_rejected() {
        let missing_path = std::env::temp_dir().join(format!(
            "error-recovery-missing-{}-{}.ron",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let config = ErrorRecoveryConfig::load_or_default(missing_path).unwrap();
        assert!(config.enabled);
        assert_eq!(config.provider.max_retries, 2);
        assert_eq!(config.tool.max_attempts, 2);

        let mut invalid_config = ErrorRecoveryConfig::default();
        invalid_config.provider.max_delay_ms = 50;
        assert!(ErrorRecoveryEngine::new(invalid_config).is_err());
    }

    #[test]
    fn retry_delay_is_jittered_within_the_configured_cap() {
        let engine = ErrorRecoveryEngine::default();
        let initial_delay = engine.retry_delay(1).as_millis();
        let second_delay = engine.retry_delay(2).as_millis();
        assert!((80..=120).contains(&initial_delay));
        assert!((160..=240).contains(&second_delay));
        for attempt in 1..=8 {
            assert!(engine.retry_delay(attempt) <= Duration::from_millis(2_000));
        }
    }

    #[tokio::test]
    async fn otel_recovery_diagnostics_are_dispatched_with_attributes() {
        let _subscriber_lock = crate::test_telemetry::lock_test_subscriber();
        let exporter = InMemorySpanExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        let subscriber = tracing_subscriber::registry().with(
            tracing_opentelemetry::layer().with_tracer(provider.tracer("otel-error-recovery")),
        );
        let engine = ErrorRecoveryEngine::default();
        let hooks = HookDispatcher::from_config(
            ron::from_str::<HookConfig>(
                r#"(
                    rules: [
                        (
                            id: "error-observer",
                            name: "Error observer",
                            event: Error,
                            matcher: All,
                            timeout_ms: 1000,
                            failure_mode: FailOpen,
                            backend: Native(name: "noop"),
                        ),
                        (
                            id: "recovery-observer",
                            name: "Recovery observer",
                            event: Recovery,
                            matcher: All,
                            timeout_ms: 1000,
                            failure_mode: FailOpen,
                            backend: Native(name: "noop"),
                        ),
                    ],
                )"#,
            )
            .unwrap(),
        );
        let diagnostic = RecoveryDiagnostics {
            error_kind: ErrorKind::Tool.as_str().to_owned(),
            recoverable: true,
            recovery_action: RecoveryAction::SelfCorrect.as_str().to_owned(),
            attempt: 1,
            outcome: RecoveryAction::SelfCorrect.outcome().to_owned(),
            message: "tool argument rejected".to_owned(),
            context_toon: "recovery_context[1]{tool}:read_file".to_owned(),
        };

        let error_diagnostic = diagnostic.clone();
        async {
            engine
                .notify(
                    &hooks,
                    HookEvent::Error,
                    Some("read_file".to_owned()),
                    error_diagnostic,
                )
                .await;
            engine
                .notify(
                    &hooks,
                    HookEvent::Recovery,
                    Some("read_file".to_owned()),
                    diagnostic,
                )
                .await;
        }
        .with_subscriber(subscriber)
        .await;
        provider.force_flush().unwrap();
        let spans = exporter.get_finished_spans().unwrap();
        provider.shutdown().unwrap();

        let recovery_span = spans
            .iter()
            .find(|span| span.name.as_ref() == "error.recovery")
            .expect("error recovery span");
        assert!(matches!(
            attribute(recovery_span, "error.kind"),
            Some(Value::String(value)) if value.as_str() == "tool"
        ));
        assert!(matches!(
            attribute(recovery_span, "error.recoverable"),
            Some(Value::Bool(true))
        ));
        assert!(matches!(
            attribute(recovery_span, "recovery.action"),
            Some(Value::String(value)) if value.as_str() == "self_correct"
        ));
        let attempt = attribute(recovery_span, "recovery.attempt");
        assert!(
            matches!(attempt, Some(Value::I64(1))),
            "unexpected recovery attempt attribute: {attempt:?}"
        );
        assert!(matches!(
            attribute(recovery_span, "recovery.outcome"),
            Some(Value::String(value)) if value.as_str() == "feedback_injected"
        ));
        assert!(
            spans
                .iter()
                .any(|span| span.name.as_ref() == "hook.dispatch")
        );
        let hook_span = spans
            .iter()
            .find(|span| {
                span.name.as_ref() == "hook.dispatch"
                    && matches!(
                        attribute(span, "hook.event"),
                        Some(Value::String(value)) if value.as_str() == "Recovery"
                    )
            })
            .expect("recovery hook dispatch span");
        assert!(matches!(
            attribute(hook_span, "error.kind"),
            Some(Value::String(value)) if value.as_str() == "tool"
        ));
        assert!(matches!(
            attribute(hook_span, "recovery.action"),
            Some(Value::String(value)) if value.as_str() == "self_correct"
        ));
        assert!(
            spans
                .iter()
                .filter(|span| span.name.as_ref() == "hook.rule")
                .count()
                >= 2
        );
    }
}
