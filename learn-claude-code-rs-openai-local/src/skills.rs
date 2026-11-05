use std::{
    collections::BTreeSet,
    fs,
    path::{Component, Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use tracing::{info, info_span, instrument};
use walkdir::WalkDir;

use crate::types::toon_string;

const MIN_MATCH_SCORE: f64 = 0.25;
const SKILL_COLUMNS: &[&str] = &["name", "version", "score"];

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct SkillManifest {
    pub schema_version: u32,
    pub name: String,
    pub version: String,
    pub description: String,
    pub trigger_keywords: Vec<String>,
    pub instructions: PathBuf,
    #[serde(default)]
    pub required_tools: Vec<String>,
}

#[derive(Debug, Clone)]
struct Skill {
    manifest: SkillManifest,
    instructions: String,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SkillRegistry {
    skills: Vec<Skill>,
}

#[derive(Debug)]
struct SkillMatch<'a> {
    skill: &'a Skill,
    score: f64,
}

impl SkillRegistry {
    #[instrument(name = "skills.discover", skip(directory), fields(skill_count = tracing::field::Empty))]
    pub(crate) fn load_dir(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let root = directory.canonicalize().with_context(|| {
            format!("Failed to resolve skills directory {}", directory.display())
        })?;
        if !root.is_dir() {
            bail!("Skills path {} is not a directory", root.display());
        }

        let mut skills = Vec::new();
        let mut skill_names = BTreeSet::new();
        for entry in WalkDir::new(&root).follow_links(false) {
            let entry = entry
                .with_context(|| format!("Failed to scan skills directory {}", root.display()))?;
            if !entry.file_type().is_file() || entry.file_name() != "skill.ron" {
                continue;
            }

            let manifest_path = entry.path();
            let contents = fs::read_to_string(manifest_path).with_context(|| {
                format!("Failed to read skill manifest {}", manifest_path.display())
            })?;
            let manifest: SkillManifest = ron::from_str(&contents).with_context(|| {
                format!("Failed to parse skill manifest {}", manifest_path.display())
            })?;
            validate_manifest(&manifest, manifest_path)?;
            if !skill_names.insert(manifest.name.clone()) {
                bail!("Duplicate skill name '{}'", manifest.name);
            }

            let skill_directory = manifest_path
                .parent()
                .context("Skill manifest has no parent directory")?
                .canonicalize()
                .with_context(|| {
                    format!(
                        "Failed to resolve skill directory for {}",
                        manifest_path.display()
                    )
                })?;
            let instruction_path = skill_directory.join(&manifest.instructions);
            let resolved_instruction_path = instruction_path.canonicalize().with_context(|| {
                format!(
                    "Failed to resolve instructions for skill '{}'",
                    manifest.name
                )
            })?;
            if !resolved_instruction_path.starts_with(&skill_directory) {
                bail!(
                    "Skill '{}' instructions must remain inside its skill directory",
                    manifest.name
                );
            }
            if resolved_instruction_path
                .extension()
                .is_none_or(|ext| ext != "md")
            {
                bail!(
                    "Skill '{}' instructions must point to a Markdown file",
                    manifest.name
                );
            }
            let instructions =
                fs::read_to_string(&resolved_instruction_path).with_context(|| {
                    format!("Failed to read instructions for skill '{}'", manifest.name)
                })?;
            if instructions.trim().is_empty() {
                bail!("Skill '{}' instructions must not be empty", manifest.name);
            }
            skills.push(Skill {
                manifest,
                instructions,
            });
        }

        skills.sort_by(|left, right| left.manifest.name.cmp(&right.manifest.name));
        tracing::Span::current().record("skill_count", skills.len());
        info!(skill_count = skills.len(), "Discovered local skills");
        Ok(Self { skills })
    }

    #[instrument(
        name = "skills.intent_screen",
        skip(self, task, available_tools),
        fields(task_bytes = task.len(), selected_skills = tracing::field::Empty)
    )]
    fn matching_skills<'a>(
        &'a self,
        task: &str,
        available_tools: &[String],
    ) -> Vec<SkillMatch<'a>> {
        let task_tokens = normalized_tokens(task);
        let available_tools = available_tools.iter().cloned().collect::<BTreeSet<_>>();
        let mut matches = self
            .skills
            .iter()
            .filter(|skill| {
                skill
                    .manifest
                    .required_tools
                    .iter()
                    .all(|tool| available_tools.contains(tool))
            })
            .filter_map(|skill| {
                let matched = skill
                    .manifest
                    .trigger_keywords
                    .iter()
                    .filter(|keyword| phrase_matches(&task_tokens, keyword))
                    .count();
                let score = matched as f64 / skill.manifest.trigger_keywords.len() as f64;
                (matched > 0 && score >= MIN_MATCH_SCORE).then_some(SkillMatch { skill, score })
            })
            .collect::<Vec<_>>();
        matches.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.skill.manifest.name.cmp(&right.skill.manifest.name))
        });
        tracing::Span::current().record("selected_skills", matches.len());
        info!(
            selected_skills = matches.len(),
            "Screened task against skill triggers"
        );
        matches
    }

    pub(crate) fn render_context(&self, task: &str, available_tools: &[String]) -> String {
        let matches = self.matching_skills(task, available_tools);
        if matches.is_empty() {
            return String::new();
        }

        let mut output = format!("Matched skills (TOON):\n{}", render_skill_matches(&matches));
        for matched in matches {
            let skill = &matched.skill;
            let span = info_span!(
                "skill.invocation",
                "skill.name" = %skill.manifest.name,
                "skill.version" = %skill.manifest.version,
                "skill.match_score" = matched.score,
                "skill.duration_ms" = tracing::field::Empty,
            );
            let started = Instant::now();
            let section = span.in_scope(|| {
                format!(
                    "\n\n--- Skill: {} (v{}) ---\n{}",
                    skill.manifest.name, skill.manifest.version, skill.instructions
                )
            });
            span.record(
                "skill.duration_ms",
                started.elapsed().as_secs_f64() * 1000.0,
            );
            span.in_scope(|| {
                info!(
                    skill_name = %skill.manifest.name,
                    skill_version = %skill.manifest.version,
                    match_score = matched.score,
                    "Injected matched skill instructions"
                )
            });
            output.push_str(&section);
        }
        output
    }
}

fn validate_manifest(manifest: &SkillManifest, path: &Path) -> Result<()> {
    if manifest.schema_version != 1 {
        bail!(
            "Unsupported skill schema_version {} in {}",
            manifest.schema_version,
            path.display()
        );
    }
    if manifest.name.is_empty()
        || !manifest
            .name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        bail!(
            "Skill name in {} must contain only ASCII letters, digits, '-' or '_'",
            path.display()
        );
    }
    if manifest.version.trim().is_empty() || manifest.description.trim().is_empty() {
        bail!(
            "Skill '{}' must declare a version and description",
            manifest.name
        );
    }
    if manifest.trigger_keywords.is_empty()
        || manifest
            .trigger_keywords
            .iter()
            .any(|keyword| normalized_tokens(keyword).is_empty())
    {
        bail!(
            "Skill '{}' must declare non-empty trigger keywords",
            manifest.name
        );
    }
    if manifest.instructions.is_absolute()
        || manifest
            .instructions
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
        || manifest
            .instructions
            .extension()
            .is_none_or(|ext| ext != "md")
    {
        bail!(
            "Skill '{}' instructions must be a relative Markdown path without traversal",
            manifest.name
        );
    }

    let allowed_tools = ["bash", "read_file", "todo_write", "delegate_task"];
    let mut seen_tools = BTreeSet::new();
    for tool in &manifest.required_tools {
        if !allowed_tools.contains(&tool.as_str()) {
            bail!("Skill '{}' declares unknown tool '{tool}'", manifest.name);
        }
        if !seen_tools.insert(tool) {
            bail!("Skill '{}' declares duplicate tool '{tool}'", manifest.name);
        }
    }
    Ok(())
}

fn normalized_tokens(value: &str) -> Vec<String> {
    value
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn phrase_matches(task_tokens: &[String], keyword: &str) -> bool {
    let keyword_tokens = normalized_tokens(keyword);
    !keyword_tokens.is_empty()
        && task_tokens
            .windows(keyword_tokens.len())
            .any(|window| window == keyword_tokens)
}

fn render_skill_matches(matches: &[SkillMatch<'_>]) -> String {
    let mut output = format!("skills[{}]{{{}}}:", matches.len(), SKILL_COLUMNS.join(","));
    for matched in matches {
        output.push_str(&format!(
            "\n{},{},{:.3}",
            toon_string(&matched.skill.manifest.name),
            toon_string(&matched.skill.manifest.version),
            matched.score
        ));
    }
    output
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use super::{SkillManifest, SkillRegistry};

    #[test]
    fn parses_ron_skill_manifest_and_scores_trigger_phrases() {
        let manifest: SkillManifest = ron::from_str(
            r#"(
                schema_version: 1,
                name: "rust-testing",
                version: "1.2.0",
                description: "Guidance for Rust tests",
                trigger_keywords: ["cargo test", "rust tests"],
                instructions: "instructions.md",
                required_tools: ["bash"],
            )"#,
        )
        .unwrap();
        assert_eq!(manifest.name, "rust-testing");

        let registry = SkillRegistry::load_dir(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("skills"),
        )
        .unwrap();
        let matches = registry.matching_skills(
            "Please run CARGO TEST before reviewing the changes.",
            &["bash".to_owned()],
        );
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].skill.manifest.name, "rust-testing");
        assert_eq!(matches[0].score, 0.5);
        assert!(
            registry
                .matching_skills("Please review the changes.", &["bash".to_owned()])
                .is_empty()
        );
        assert!(
            registry
                .matching_skills("cargo test", &["read_file".to_owned()])
                .is_empty()
        );
    }

    #[test]
    fn rejects_instruction_paths_that_escape_the_skill_directory() {
        let temp = std::env::temp_dir().join(format!("skill-path-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        let skill_dir = temp.join("escape");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("skill.ron"),
            r#"(schema_version: 1, name: "escape", version: "1", description: "test", trigger_keywords: ["test"], instructions: "../outside.md")"#,
        )
        .unwrap();
        fs::write(temp.join("outside.md"), "outside").unwrap();
        assert!(SkillRegistry::load_dir(&temp).is_err());
        fs::remove_dir_all(&temp).unwrap();
    }
}
