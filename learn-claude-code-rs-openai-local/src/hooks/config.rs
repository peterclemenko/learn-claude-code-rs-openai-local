use std::{fs, path::Path, time::Duration};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) enum HookEvent {
    SessionStart,
    SessionEnd,
    UserPromptSubmit,
    PreToolUse,
    PostToolUse,
    Error,
    Recovery,
    SubagentStart,
    SubagentStop,
}

impl HookEvent {
    pub(crate) fn as_name(self) -> &'static str {
        match self {
            Self::SessionStart => "SessionStart",
            Self::SessionEnd => "SessionEnd",
            Self::UserPromptSubmit => "UserPromptSubmit",
            Self::PreToolUse => "PreToolUse",
            Self::PostToolUse => "PostToolUse",
            Self::Error => "Error",
            Self::Recovery => "Recovery",
            Self::SubagentStart => "SubagentStart",
            Self::SubagentStop => "SubagentStop",
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) enum HookMatcher {
    All,
    Glob(String),
    Regex(String),
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) enum HookPayloadFormat {
    Json,
    Ron,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) enum HookBackend {
    External {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        payload_format: HookPayloadFormat,
    },
    Native {
        name: String,
    },
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) enum FailureMode {
    FailOpen,
    FailClosed,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) struct HookRule {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) event: HookEvent,
    #[serde(default)]
    pub(crate) matcher: HookMatcher,
    #[serde(default = "default_timeout_ms")]
    pub(crate) timeout_ms: u64,
    #[serde(default)]
    pub(crate) failure_mode: FailureMode,
    pub(crate) backend: HookBackend,
}

impl HookRule {
    pub(crate) fn timeout(&self) -> Duration {
        Duration::from_millis(self.timeout_ms)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) struct HookConfig {
    #[serde(default)]
    pub(crate) rules: Vec<HookRule>,
}

impl HookConfig {
    pub(crate) fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path)
            .with_context(|| format!("Failed to read hook config {}", path.display()))?;
        let config: Self = ron::from_str(&contents)
            .with_context(|| format!("Failed to parse RON hook config {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        for rule in &self.rules {
            anyhow::ensure!(!rule.id.trim().is_empty(), "Hook rule id must not be empty");
            anyhow::ensure!(
                !rule.name.trim().is_empty(),
                "Hook rule name must not be empty"
            );
            if let HookBackend::External { command, .. } = &rule.backend {
                anyhow::ensure!(
                    !command.trim().is_empty(),
                    "External hook command must not be empty"
                );
            }
            if rule.timeout_ms == 0 {
                anyhow::ensure!(
                    false,
                    "Hook rule '{}' timeout_ms must be greater than zero",
                    rule.id
                );
            }
        }
        Ok(())
    }
}

impl Default for FailureMode {
    fn default() -> Self {
        Self::FailOpen
    }
}

impl Default for HookMatcher {
    fn default() -> Self {
        Self::All
    }
}

impl Default for HookPayloadFormat {
    fn default() -> Self {
        Self::Json
    }
}

fn default_timeout_ms() -> u64 {
    5000
}
