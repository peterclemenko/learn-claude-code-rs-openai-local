# Specification: Permission System & Profile Engine

## 1. Overview
This subsystem implements a fine-grained, declarative security and permission evaluation harness for tool execution in `learn-claude-code-rs-openai-local`. It decouples permission definitions into RON files, supports hierarchical profile overrides, integrates OpenTelemetry tracing natively, and includes comprehensive test harnesses.

This is modernized from the s07 tutorial at https://github.com/wulawulu/learn-claude-code-rs/tree/main/s07_permission_system 
---

## 2. Directory Layout
The permission configuration tree must reside under `config/permissions/`:

```text
config/
└── permissions/
    ├── bash.ron                  # Default permission rules for bash commands
    ├── file_write.ron            # Default permission rules for file writing
    ├── file_read.ron             # Default permission rules for file reading
    └── profiles/
        ├── strict/
        │   ├── profile.ron       # Profile metadata & inheritance rules
        │   └── permissions/
        │       └── bash.ron      # Profile-specific overrides for bash
        └── autonomous/
            ├── profile.ron
            └── permissions/
                └── file_write.ron
3. RON Schema Definitions
Default Permission Rule (config/permissions/<tool or cmd>.ron)
Rust
(
    action: Prompt, // Options: Allow, Deny, Prompt
    description: "Description of what this tool or command is permitted to do.",
    allowed_patterns: [
        "^cargo test.*",
        "^git status\$",
    ],
    denied_patterns: [
        "^rm -rf /.*",
        "^sudo.*",
    ],
)
Profile Configuration (config/permissions/profiles/<profilename>/profile.ron)
Rust
(
    name: "strict",
    description: "Restricted sandbox profile requiring confirmation for mutating operations.",
    inherits_default: true,
)
4. Evaluation and Resolution Logic
Base Lookup: Locate the default permission file at config/permissions/{tool_name}.ron. If missing, fallback to a safe default (PermissionAction::Prompt).

Profile Override Resolution: If an active profile is specified, inspect config/permissions/profiles/{profile}/permissions/{tool_name}.ron.

Merging Rules:

If a profile override exists, its action overrides the default.

Profile-specific allowed_patterns and denied_patterns take precedence or extend the base rule set.

Pattern Matching Priority:

Evaluate regex checks against the target execution string (e.g., command arguments or file paths).

Deny Match: If any regex in denied_patterns matches, immediately return PermissionAction::Deny.

Allow Match: If any regex in allowed_patterns matches, return PermissionAction::Allow.

Default Fallback: If no patterns match, return the configured base action.

5. Rust Core Implementation
Rust
use opentelemetry::{
    global,
    trace::{TraceContextExt, Tracer},
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::{info, instrument};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PermissionAction {
    Allow,
    Deny,
    Prompt,
}

impl ToString for PermissionAction {
    fn to_string(&self) -> String {
        match self {
            Self::Allow => "allow".into(),
            Self::Deny => "deny".into(),
            Self::Prompt => "prompt".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionRule {
    pub action: PermissionAction,
    pub description: String,
    #[serde(default)]
    pub allowed_patterns: Vec<String>,
    #[serde(default)]
    pub denied_patterns: Vec<String>,
}

pub struct PermissionManager {
    base_dir: PathBuf,
    active_profile: Option<String>,
}

impl PermissionManager {
    pub fn new(base_dir: impl Into<PathBuf>, active_profile: Option<String>) -> Self {
        Self {
            base_dir: base_dir.into(),
            active_profile,
        }
    }

    /// Evaluates permissions with full OTel telemetry context and pattern matching.
    #[instrument(skip(self), fields(tool.name = %tool_name, target.input = %target_input, profile = ?self.active_profile))]
    pub async fn evaluate(
        &self,
        tool_name: &str,
        target_input: &str,
    ) -> Result<PermissionAction, Box<dyn std::error::Error + Send + Sync>> {
        let tracer = global::tracer("learn-claude-code-rs-openai-local::permissions");
        let _span = tracer.start("permission_evaluation");

        // 1. Load default rule
        let default_path = self.base_dir.join("permissions").join(format!("{}.ron", tool_name));
        let mut rule = self.load_rule(&default_path).await.unwrap_or_else(|_| PermissionRule {
            action: PermissionAction::Prompt,
            description: "Default fallback permission".into(),
            allowed_patterns: vec![],
            denied_patterns: vec![],
        });

        // 2. Profile Override Resolution
        if let Some(ref profile) = self.active_profile {
            let profile_rule_path = self.base_dir
                .join("permissions")
                .join("profiles")
                .join(profile)
                .join("permissions")
                .join(format!("{}.ron", tool_name));

            if profile_rule_path.exists() {
                let profile_rule = self.load_rule(&profile_rule_path).await?;
                rule.action = profile_rule.action;
                if !profile_rule.allowed_patterns.is_empty() {
                    rule.allowed_patterns = profile_rule.allowed_patterns;
                }
                if !profile_rule.denied_patterns.is_empty() {
                    rule.denied_patterns = profile_rule.denied_patterns;
                }
            }
        }

        // 3. Pattern Matching Evaluation
        let mut final_action = rule.action;

        for pattern in &rule.denied_patterns {
            let re = regex::Regex::new(pattern)?;
            if re.is_match(target_input) {
                final_action = PermissionAction::Deny;
                info!(tool = tool_name, pattern = pattern, "Permission DENIED by pattern match");
                break;
            }
        }

        if final_action != PermissionAction::Deny {
            for pattern in &rule.allowed_patterns {
                let re = regex::Regex::new(pattern)?;
                if re.is_match(target_input) {
                    final_action = PermissionAction::Allow;
                    info!(tool = tool_name, pattern = pattern, "Permission ALLOWED by pattern match");
                    break;
                }
            }
        }

        tracing::Span::current().record("permission.action", &final_action.to_string().as_str());

        Ok(final_action)
    }

    async fn load_rule(&self, path: &Path) -> Result<PermissionRule, Box<dyn std::error::Error + Send + Sync>> {
        let content = fs::read_to_string(path).await?;
        let rule: PermissionRule = ron::from_str(&content)?;
        Ok(rule)
    }
}
6. OpenTelemetry Observability & Testing
Telemetry Spans
Every call to PermissionManager::evaluate is wrapped in an instrumented OpenTelemetry span.

Recorded Attributes:

tool.name: Name of the tool being executed (e.g., bash, file_write).

target.input: Command string or file path payload.

profile: Active security profile (if any).

permission.action: Resulting enforcement action (allow, deny, prompt).

Native Test Harnesses (tests/permission_tests.rs)
Rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tokio::fs;

    async fn setup_test_env() -> TempDir {
        let dir = TempDir::new().unwrap();
        let perm_dir = dir.path().join("permissions");
        let profile_dir = perm_dir.join("profiles").join("strict").join("permissions");

        fs::create_dir_all(&perm_dir).await.unwrap();
        fs::create_dir_all(&profile_dir).await.unwrap();

        let default_bash = r#"
        (
            action: Prompt,
            description: "Bash execution rule",
            allowed_patterns: [
                "^cargo test.*",
            ],
            denied_patterns: [
                "^rm -rf /.*",
            ],
        )
        "#;
        fs::write(perm_dir.join("bash.ron"), default_bash).await.unwrap();

        let strict_bash = r#"
        (
            action: Deny,
            description: "Strict override",
            allowed_patterns: [
                "^cargo test --workspace\$",
            ],
            denied_patterns: [],
        )
        "#;
        fs::write(profile_dir.join("bash.ron"), strict_bash).await.unwrap();

        dir
    }

    #[tokio::test]
    async fn test_default_allow_pattern() {
        let tmp = setup_test_env().await;
        let manager = PermissionManager::new(tmp.path(), None);
        let action = manager.evaluate("bash", "cargo test --lib").await.unwrap();
        assert_eq!(action, PermissionAction::Allow);
    }

    #[tokio::test]
    async fn test_default_deny_pattern() {
        let tmp = setup_test_env().await;
        let manager = PermissionManager::new(tmp.path(), None);
        let action = manager.evaluate("bash", "rm -rf /home/user").await.unwrap();
        assert_eq!(action, PermissionAction::Deny);
    }

    #[tokio::test]
    async fn test_profile_override_strict() {
        let tmp = setup_test_env().await;
        let manager = PermissionManager::new(tmp.path(), Some("strict".into()));

        let action = manager.evaluate("bash", "cargo test --lib").await.unwrap();
        assert_eq!(action, PermissionAction::Deny);

        let action_strict = manager.evaluate("bash", "cargo test --workspace").await.unwrap();
        assert_eq!(action_strict, PermissionAction::Allow);
    }
}