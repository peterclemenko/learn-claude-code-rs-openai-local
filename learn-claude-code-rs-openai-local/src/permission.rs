use std::{
    collections::BTreeMap,
    fmt, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use inquire::Select;
use regex::Regex;
use serde::{Deserialize, Serialize};
use wildmatch::WildMatch;

const READ_ONLY_TOOLS: &[&str] = &["read_file"];

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
enum PermissionMode {
    Default,
    Plan,
    Auto,
}

impl PermissionMode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "default" => Ok(Self::Default),
            "plan" => Ok(Self::Plan),
            "auto" => Ok(Self::Auto),
            _ => bail!("Unknown permission mode '{value}'. Use default, plan, or auto."),
        }
    }
}

impl fmt::Display for PermissionMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Default => "default",
            Self::Plan => "plan",
            Self::Auto => "auto",
        })
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
enum PermissionAction {
    Allow,
    Deny,
    Ask,
}

impl fmt::Display for PermissionAction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Allow => "allow",
            Self::Deny => "deny",
            Self::Ask => "ask",
        })
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PermissionRule {
    action: PermissionAction,
    pattern: String,
    reason: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ToolPermissionConfig {
    tool_name: String,
    default_behavior: PermissionAction,
    safety_checks_enabled: bool,
    rules: Vec<PermissionRule>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct PermissionDefaults {
    fallback_mode: PermissionMode,
    #[serde(default)]
    active_profile: Option<String>,
    #[serde(default = "default_persist_always_allow")]
    persist_always_allow: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileConfig {
    name: String,
    description: String,
    fallback_mode: PermissionMode,
    enforce_strict_bash_checks: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PermissionBehavior {
    Allow,
    Deny,
    Ask,
}

#[derive(Debug)]
struct SafetyRule {
    name: &'static str,
    regex: Regex,
    deny: bool,
}

#[derive(Debug)]
pub(crate) struct PermissionManager {
    root: PathBuf,
    defaults: PermissionDefaults,
    tool_configs: BTreeMap<String, ToolPermissionConfig>,
    active_profile: Option<ProfileConfig>,
    mode_override: Option<PermissionMode>,
    consecutive_denials: usize,
    bash_safety_rules: Vec<SafetyRule>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UserChoice {
    AllowOnce,
    Deny,
    AlwaysAllow,
}

impl fmt::Display for UserChoice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::AllowOnce => "allow once",
            Self::Deny => "deny",
            Self::AlwaysAllow => "always allow this tool",
        })
    }
}

impl PermissionManager {
    pub(crate) fn load(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let defaults_path = root.join("default.ron");
        let defaults = read_ron::<PermissionDefaults>(&defaults_path)?;
        let active_profile = defaults.active_profile.clone();
        let tool_configs = read_tool_configs(&root)?;

        let mut manager = Self {
            root,
            defaults,
            tool_configs,
            active_profile: None,
            mode_override: None,
            consecutive_denials: 0,
            bash_safety_rules: make_bash_safety_rules()?,
        };
        if let Some(profile) = active_profile {
            manager.switch_profile(&profile)?;
        }
        Ok(manager)
    }

    #[cfg(test)]
    pub(crate) fn allow_all_for_tests() -> Self {
        let tool_configs = ["bash", "read_file", "todo_write", "delegate_task"]
            .into_iter()
            .map(|tool_name| {
                (
                    tool_name.to_owned(),
                    ToolPermissionConfig {
                        tool_name: tool_name.to_owned(),
                        default_behavior: PermissionAction::Allow,
                        safety_checks_enabled: false,
                        rules: vec![PermissionRule {
                            action: PermissionAction::Allow,
                            pattern: "*".to_owned(),
                            reason: "Test-only permission".to_owned(),
                        }],
                    },
                )
            })
            .collect();
        Self {
            root: PathBuf::new(),
            defaults: PermissionDefaults {
                fallback_mode: PermissionMode::Auto,
                active_profile: None,
                persist_always_allow: false,
            },
            tool_configs,
            active_profile: None,
            mode_override: None,
            consecutive_denials: 0,
            bash_safety_rules: Vec::new(),
        }
    }

    fn mode(&self) -> PermissionMode {
        self.mode_override.unwrap_or_else(|| {
            self.active_profile
                .as_ref()
                .map_or(self.defaults.fallback_mode, |profile| profile.fallback_mode)
        })
    }

    pub(crate) fn check(&self, tool_name: &str, payload: &str) -> (PermissionBehavior, String) {
        let config = self.tool_configs.get(tool_name);
        let value = match permission_value(tool_name, payload) {
            Ok(value) => value,
            Err(error) => {
                return (
                    PermissionBehavior::Ask,
                    format!(
                        "Tool arguments could not be parsed for permission matching: {error:#}"
                    ),
                );
            }
        };

        if let Some(rule) = matching_rule(config, PermissionAction::Deny, &value) {
            return (PermissionBehavior::Deny, rule.reason.clone());
        }

        match self.mode() {
            PermissionMode::Plan if !READ_ONLY_TOOLS.contains(&tool_name) => {
                return (
                    PermissionBehavior::Deny,
                    "Plan mode blocks tools that may change state".to_owned(),
                );
            }
            PermissionMode::Plan => {
                return (
                    PermissionBehavior::Allow,
                    "Plan mode allows read-only tools".to_owned(),
                );
            }
            PermissionMode::Default | PermissionMode::Auto => {}
        }

        if tool_name == "bash"
            && (config.is_some_and(|config| config.safety_checks_enabled)
                || self
                    .active_profile
                    .as_ref()
                    .is_some_and(|profile| profile.enforce_strict_bash_checks))
        {
            let failures = self
                .bash_safety_rules
                .iter()
                .filter(|rule| rule.regex.is_match(&value))
                .collect::<Vec<_>>();
            if !failures.is_empty() {
                let names = failures
                    .iter()
                    .map(|rule| rule.name)
                    .collect::<Vec<_>>()
                    .join(", ");
                let behavior = if failures.iter().any(|rule| rule.deny) {
                    PermissionBehavior::Deny
                } else {
                    PermissionBehavior::Ask
                };
                return (behavior, format!("Bash safety checks flagged: {names}"));
            }
        }

        if let Some(rule) = matching_rule(config, PermissionAction::Ask, &value) {
            return (PermissionBehavior::Ask, rule.reason.clone());
        }

        if let Some(rule) = matching_rule(config, PermissionAction::Allow, &value) {
            return (PermissionBehavior::Allow, rule.reason.clone());
        }

        if self.mode() == PermissionMode::Auto {
            if READ_ONLY_TOOLS.contains(&tool_name) {
                return (
                    PermissionBehavior::Allow,
                    "Auto mode allows read-only tools".to_owned(),
                );
            }
            return (
                PermissionBehavior::Ask,
                format!("Auto mode requires approval for '{tool_name}'"),
            );
        }

        let behavior = config.map_or(PermissionAction::Ask, |config| config.default_behavior);
        (
            to_behavior(behavior),
            format!("Tool '{tool_name}' default behavior is {behavior}"),
        )
    }

    pub(crate) fn ask_user(&mut self, tool_name: &str, payload: &str) -> Result<bool> {
        let preview = truncate(payload, 200);
        let prompt = format!("[Permission] {tool_name}: {preview}");
        let choice = Select::new(
            &prompt,
            vec![
                UserChoice::AllowOnce,
                UserChoice::Deny,
                UserChoice::AlwaysAllow,
            ],
        )
        .prompt()
        .context("Failed to read permission decision")?;

        match choice {
            UserChoice::AllowOnce => {
                self.consecutive_denials = 0;
                Ok(true)
            }
            UserChoice::Deny => {
                self.consecutive_denials += 1;
                if self.consecutive_denials >= 3 {
                    println!(
                        "[{} consecutive denials -- consider switching to plan mode]",
                        self.consecutive_denials
                    );
                }
                Ok(false)
            }
            UserChoice::AlwaysAllow => {
                self.add_allow_rule(tool_name)?;
                self.consecutive_denials = 0;
                Ok(true)
            }
        }
    }

    pub(crate) fn handle_command(&mut self, command: &str) -> Result<String> {
        let mut parts = command.split_whitespace();
        match parts.next() {
            Some("/rules") if parts.next().is_none() => Ok(self.render_rules()),
            Some("/mode") => {
                let Some(mode) = parts.next() else {
                    bail!("Usage: /mode <default|plan|auto>");
                };
                if parts.next().is_some() {
                    bail!("Usage: /mode <default|plan|auto>");
                }
                let mode = PermissionMode::parse(mode)?;
                self.mode_override = Some(mode);
                Ok(format!("Permission mode set to {mode}."))
            }
            Some("/profile") => {
                let Some(name) = parts.next() else {
                    bail!("Usage: /profile <name>");
                };
                if parts.next().is_some() {
                    bail!("Usage: /profile <name>");
                }
                self.switch_profile(name)?;
                let profile = self.active_profile.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("Profile switch completed without an active profile")
                })?;
                Ok(format!(
                    "Active profile: {} ({}) [mode: {}]",
                    profile.name,
                    profile.description,
                    self.mode()
                ))
            }
            _ => bail!("Usage: /rules, /mode <default|plan|auto>, or /profile <name>"),
        }
    }

    fn switch_profile(&mut self, name: &str) -> Result<()> {
        if name.is_empty()
            || !name.chars().all(|character| {
                character.is_ascii_alphanumeric() || matches!(character, '-' | '_')
            })
        {
            bail!("Invalid profile name '{name}'");
        }

        let profile_dir = self.root.join("profiles").join(name);
        let profile_path = profile_dir.join("profile.ron");
        let profile = read_ron::<ProfileConfig>(&profile_path)?;
        if profile.name != name {
            bail!(
                "Profile name mismatch in {}: expected '{name}', got '{}'",
                profile_path.display(),
                profile.name
            );
        }

        let mut tool_configs = read_tool_configs(&self.root)?;
        let overrides_dir = profile_dir.join("permissions");
        match fs::read_dir(&overrides_dir) {
            Ok(entries) => {
                for entry in entries {
                    let entry = entry.with_context(|| {
                        format!(
                            "Failed to read profile directory {}",
                            overrides_dir.display()
                        )
                    })?;
                    let path = entry.path();
                    if path.extension().is_some_and(|extension| extension == "ron") {
                        let config = read_ron::<ToolPermissionConfig>(&path)?;
                        let expected_name = path
                            .file_stem()
                            .and_then(|stem| stem.to_str())
                            .context("Profile permission filename is not valid UTF-8")?;
                        validate_tool_name(&config, expected_name, &path)?;
                        tool_configs.insert(config.tool_name.clone(), config);
                    }
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "Failed to read profile directory {}",
                        overrides_dir.display()
                    )
                });
            }
        }

        self.active_profile = Some(profile);
        self.tool_configs = tool_configs;
        self.mode_override = None;
        self.consecutive_denials = 0;
        Ok(())
    }

    fn add_allow_rule(&mut self, tool_name: &str) -> Result<()> {
        let config = self
            .tool_configs
            .entry(tool_name.to_owned())
            .or_insert_with(|| ToolPermissionConfig {
                tool_name: tool_name.to_owned(),
                default_behavior: PermissionAction::Ask,
                safety_checks_enabled: false,
                rules: Vec::new(),
            });
        let mut updated = config.clone();
        updated.rules.push(PermissionRule {
            action: PermissionAction::Allow,
            pattern: "*".to_owned(),
            reason: "Always allowed by user".to_owned(),
        });

        if self.defaults.persist_always_allow
            && let Some(profile) = &self.active_profile
        {
            let path = self
                .root
                .join("profiles")
                .join(&profile.name)
                .join("permissions")
                .join(format!("{tool_name}.ron"));
            let parent = path
                .parent()
                .context("Profile permission path has no parent")?;
            fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create profile directory {}", parent.display())
            })?;
            let serialized =
                ron::ser::to_string_pretty(&updated, ron::ser::PrettyConfig::default())
                    .context("Failed to serialize permission rules to RON")?;
            fs::write(&path, serialized).with_context(|| {
                format!("Failed to persist permission rules to {}", path.display())
            })?;
        }
        self.tool_configs.insert(tool_name.to_owned(), updated);
        Ok(())
    }

    fn render_rules(&self) -> String {
        let mut output = format!("Permission mode: {}\n", self.mode());
        if let Some(profile) = &self.active_profile {
            output.push_str(&format!(
                "Profile: {} ({})\n",
                profile.name, profile.description
            ));
        } else {
            output.push_str("Profile: none\n");
        }
        for (tool_name, config) in &self.tool_configs {
            output.push_str(&format!(
                "\n{tool_name}: default={}, safety_checks_enabled={}\n",
                config.default_behavior, config.safety_checks_enabled
            ));
            for rule in &config.rules {
                output.push_str(&format!(
                    "  {} {} -- {}\n",
                    rule.action, rule.pattern, rule.reason
                ));
            }
        }
        output
    }
}

fn default_persist_always_allow() -> bool {
    true
}

fn to_behavior(action: PermissionAction) -> PermissionBehavior {
    match action {
        PermissionAction::Allow => PermissionBehavior::Allow,
        PermissionAction::Deny => PermissionBehavior::Deny,
        PermissionAction::Ask => PermissionBehavior::Ask,
    }
}

fn matching_rule<'a>(
    config: Option<&'a ToolPermissionConfig>,
    action: PermissionAction,
    value: &str,
) -> Option<&'a PermissionRule> {
    config?
        .rules
        .iter()
        .find(|rule| rule.action == action && WildMatch::new(&rule.pattern).matches(value))
}

fn permission_value(tool_name: &str, payload: &str) -> Result<String> {
    let field = match tool_name {
        "bash" => "command",
        "read_file" | "write_file" | "edit_file" => "path",
        _ => return Ok(payload.to_owned()),
    };
    crate::types::parse_toon_field(payload, tool_name, field)
}

fn make_bash_safety_rules() -> Result<Vec<SafetyRule>> {
    [
        ("sudo", r"\bsudo\b", true),
        ("recursive_rm", r"\brm\s+(-[a-zA-Z]*)?r", true),
        ("shell_metacharacters", r"[;&|`$]", false),
        ("command_substitution", r"\$\(", false),
        ("ifs_assignment", r"\bIFS\s*=", false),
    ]
    .into_iter()
    .map(|(name, pattern, deny)| {
        Ok(SafetyRule {
            name,
            regex: Regex::new(pattern)
                .with_context(|| format!("Failed to compile bash safety pattern '{name}'"))?,
            deny,
        })
    })
    .collect()
}

fn read_ron<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let contents =
        fs::read_to_string(path).with_context(|| format!("Failed to read {}", path.display()))?;
    ron::from_str(&contents).with_context(|| format!("Failed to parse RON {}", path.display()))
}

fn read_tool_configs(root: &Path) -> Result<BTreeMap<String, ToolPermissionConfig>> {
    let entries = fs::read_dir(root)
        .with_context(|| format!("Failed to read permission directory {}", root.display()))?;
    let mut configs = BTreeMap::new();
    for entry in entries {
        let entry = entry
            .with_context(|| format!("Failed to read permission directory {}", root.display()))?;
        let path = entry.path();
        if path.file_name().is_some_and(|name| name == "default.ron")
            || path.extension().is_none_or(|extension| extension != "ron")
        {
            continue;
        }

        let expected_name = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .context("Permission filename is not valid UTF-8")?;
        let config = read_ron::<ToolPermissionConfig>(&path)?;
        validate_tool_name(&config, expected_name, &path)?;
        configs.insert(config.tool_name.clone(), config);
    }
    Ok(configs)
}

fn validate_tool_name(config: &ToolPermissionConfig, expected: &str, path: &Path) -> Result<()> {
    if config.tool_name != expected {
        bail!(
            "Tool name mismatch in {}: filename requires '{expected}', got '{}'",
            path.display(),
            config.tool_name
        );
    }
    Ok(())
}

fn truncate(value: &str, limit: usize) -> String {
    let mut chars = value.chars();
    let preview = chars.by_ref().take(limit).collect::<String>();
    if chars.next().is_some() {
        format!("{preview}...")
    } else {
        preview
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::{PermissionBehavior, PermissionManager};

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "permission-manager-test-{}-{nonce}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn write(&self, path: &str, contents: &str) {
            let path = self.0.join(path);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, contents).unwrap();
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    fn permission_files(directory: &TestDirectory) {
        directory.write(
            "default.ron",
            "(fallback_mode: Default, persist_always_allow: true)",
        );
        directory.write(
            "bash.ron",
            r#"ToolPermissionConfig(
                tool_name: "bash",
                default_behavior: Ask,
                safety_checks_enabled: true,
                rules: [
                    PermissionRule(action: Deny, pattern: "sudo *", reason: "No sudo"),
                ],
            )"#,
        );
        directory.write(
            "read_file.ron",
            r#"ToolPermissionConfig(
                tool_name: "read_file",
                default_behavior: Deny,
                safety_checks_enabled: false,
                rules: [],
            )"#,
        );
    }

    #[test]
    fn denies_bash_safety_flags_and_falls_back_to_tool_behavior() {
        crate::test_telemetry::run_test_case(
            "permission::tests::denies_bash_safety_flags_and_falls_back_to_tool_behavior",
            || {
                let directory = TestDirectory::new();
                permission_files(&directory);
                let manager = PermissionManager::load(&directory.0).unwrap();

                assert_eq!(
                    manager.check("bash", r#"{"command":"sudo rm -rf /tmp"}"#).0,
                    PermissionBehavior::Deny
                );
                assert_eq!(
                    manager.check("bash", r#"{"command":"echo hello"}"#).0,
                    PermissionBehavior::Ask
                );
                assert_eq!(
                    manager.check("bash", "bash[1]{command}:\nrm -rf /").0,
                    PermissionBehavior::Deny
                );
            },
        );
    }

    #[test]
    fn profile_overrides_tool_rules_and_fallback_mode() {
        crate::test_telemetry::run_test_case(
            "permission::tests::profile_overrides_tool_rules_and_fallback_mode",
            || {
                let directory = TestDirectory::new();
                permission_files(&directory);
                directory.write(
                    "profiles/ci/profile.ron",
                    r#"ProfileConfig(
                name: "ci",
                description: "Allow selected commands without a prompt",
                fallback_mode: Auto,
                enforce_strict_bash_checks: false,
            )"#,
                );
                directory.write(
                    "profiles/ci/permissions/bash.ron",
                    r#"ToolPermissionConfig(
                tool_name: "bash",
                default_behavior: Ask,
                safety_checks_enabled: false,
                rules: [
                    PermissionRule(
                        action: Allow,
                        pattern: "printf *",
                        reason: "CI permits printf commands.",
                    ),
                ],
            )"#,
                );
                let mut manager = PermissionManager::load(&directory.0).unwrap();
                manager.handle_command("/profile ci").unwrap();

                assert_eq!(
                    manager.check("bash", r#"{"command":"printf hello"}"#).0,
                    PermissionBehavior::Allow
                );
                assert_eq!(
                    manager.check("bash", r#"{"command":"echo hello"}"#).0,
                    PermissionBehavior::Ask
                );
                assert_eq!(
                    manager.check("read_file", r#"{"path":"README.md"}"#).0,
                    PermissionBehavior::Allow
                );
            },
        );
    }

    #[test]
    fn default_config_can_activate_a_profile_on_startup() {
        crate::test_telemetry::run_test_case(
            "permission::tests::default_config_can_activate_a_profile_on_startup",
            || {
                let directory = TestDirectory::new();
                permission_files(&directory);
                directory.write(
                    "default.ron",
                    "(fallback_mode: Default, active_profile: Some(\"strict\"), persist_always_allow: true)",
                );
                directory.write(
                    "profiles/strict/profile.ron",
                    r#"ProfileConfig(
                name: "strict",
                description: "Read-only profile",
                fallback_mode: Plan,
                enforce_strict_bash_checks: true,
            )"#,
                );

                let manager = PermissionManager::load(&directory.0).unwrap();

                assert_eq!(manager.mode(), super::PermissionMode::Plan);
                assert_eq!(
                    manager.check("bash", r#"{"command":"echo hello"}"#).0,
                    PermissionBehavior::Deny
                );
            },
        );
    }

    #[test]
    fn checked_in_strict_profile_loads_and_blocks_bash() {
        crate::test_telemetry::run_test_case(
            "permission::tests::checked_in_strict_profile_loads_and_blocks_bash",
            || {
                let permissions_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .unwrap()
                    .join("config/permissions");
                let mut manager = PermissionManager::load(permissions_dir).unwrap();
                manager.handle_command("/profile strict").unwrap();

                assert_eq!(
                    manager.check("bash", r#"{"command":"echo hello"}"#).0,
                    PermissionBehavior::Deny
                );
                assert_eq!(
                    manager.check("read_file", r#"{"path":"README.md"}"#).0,
                    PermissionBehavior::Allow
                );
            },
        );
    }

    #[test]
    fn always_allow_is_persisted_to_the_active_profile() {
        crate::test_telemetry::run_test_case(
            "permission::tests::always_allow_is_persisted_to_the_active_profile",
            || {
                let directory = TestDirectory::new();
                permission_files(&directory);
                directory.write(
                    "profiles/lax/profile.ron",
                    r#"ProfileConfig(
                name: "lax",
                description: "Interactive writes",
                fallback_mode: Default,
                enforce_strict_bash_checks: false,
            )"#,
                );
                directory.write(
                    "profiles/lax/permissions/bash.ron",
                    r#"ToolPermissionConfig(
                tool_name: "bash",
                default_behavior: Ask,
                safety_checks_enabled: false,
                rules: [],
            )"#,
                );
                let mut manager = PermissionManager::load(&directory.0).unwrap();
                manager.handle_command("/profile lax").unwrap();
                manager.add_allow_rule("bash").unwrap();

                assert!(
                    directory
                        .0
                        .join("profiles/lax/permissions/bash.ron")
                        .is_file()
                );
                let mut reloaded = PermissionManager::load(&directory.0).unwrap();
                reloaded.handle_command("/profile lax").unwrap();
                assert_eq!(
                    reloaded.check("bash", r#"{"command":"echo hello"}"#).0,
                    PermissionBehavior::Allow
                );
            },
        );
    }

    #[test]
    fn plan_mode_never_prompts_for_bash_and_auto_mode_asks_for_unconfigured_writes() {
        crate::test_telemetry::run_test_case(
            "permission::tests::plan_mode_never_prompts_for_bash_and_auto_mode_asks_for_unconfigured_writes",
            || {
                let directory = TestDirectory::new();
                permission_files(&directory);
                let mut manager = PermissionManager::load(&directory.0).unwrap();
                manager.handle_command("/mode plan").unwrap();
                assert_eq!(
                    manager.check("bash", r#"{"command":"echo hello | cat"}"#).0,
                    PermissionBehavior::Deny
                );

                manager.handle_command("/mode auto").unwrap();
                assert_eq!(
                    manager.check("bash", r#"{"command":"echo hello"}"#).0,
                    PermissionBehavior::Ask
                );
            },
        );
    }
}
