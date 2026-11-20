use std::{
    collections::BTreeMap,
    fmt, fs,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};

use anyhow::Result;
use serde::Deserialize;
use tracing::{info, instrument};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptChangeSource {
    Config,
    CliFlag,
    ProgrammaticApi,
    AgentTool,
}

impl fmt::Display for PromptChangeSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Config => "Config",
            Self::CliFlag => "CliFlag",
            Self::ProgrammaticApi => "ProgrammaticApi",
            Self::AgentTool => "AgentTool",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptSwitchOutcome {
    pub name: String,
    pub path: PathBuf,
    pub rendered_prompt: String,
}

#[derive(Debug)]
pub enum PromptError {
    InvalidSelection(String),
    PathTraversal(String),
    NotFound(String),
    Io(String),
    Parse(String),
    Render(String),
}

impl fmt::Display for PromptError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSelection(message)
            | Self::PathTraversal(message)
            | Self::NotFound(message)
            | Self::Io(message)
            | Self::Parse(message)
            | Self::Render(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for PromptError {}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromptMetadata {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct PromptRenderContext {
    #[serde(default)]
    pub workspace_path: String,
    #[serde(default)]
    pub operating_system: String,
    #[serde(default)]
    pub runtime: String,
    #[serde(default)]
    pub variables: BTreeMap<String, String>,
}

impl PromptRenderContext {
    pub fn from_current_environment() -> Self {
        Self {
            workspace_path: std::env::current_dir()
                .ok()
                .and_then(|path| path.to_str().map(str::to_owned))
                .unwrap_or_else(|| ".".to_owned()),
            operating_system: std::env::consts::OS.to_owned(),
            runtime: "rust".to_owned(),
            variables: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromptTemplate {
    pub metadata: PromptMetadata,
    pub identity: String,
    pub instructions: String,
    pub tool_guidelines: String,
    pub safety_and_constraints: String,
    #[serde(default)]
    pub custom_sections: BTreeMap<String, String>,
    #[serde(default)]
    pub variables: BTreeMap<String, String>,
}

impl PromptTemplate {
    #[instrument(name = "prompt.load", skip(path))]
    pub fn load(path: impl AsRef<Path>) -> Result<Self, PromptError> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path).map_err(|error| {
            PromptError::Io(format!("Failed to read prompt {}: {error}", path.display()))
        })?;
        let template: Self = ron::from_str(&contents).map_err(|error| {
            PromptError::Parse(format!(
                "Failed to parse prompt RON {}: {error}",
                path.display()
            ))
        })?;
        if template.metadata.name.trim().is_empty() {
            return Err(PromptError::InvalidSelection(
                "Prompt metadata name must not be empty".to_owned(),
            ));
        }
        info!(
            prompt.name = %template.metadata.name,
            prompt.path = %path.display(),
            "Prompt template loaded"
        );
        Ok(template)
    }

    #[instrument(
        name = "prompt.render",
        fields(prompt.name = %self.metadata.name),
        skip(self, context)
    )]
    pub fn render(&self, context: &PromptRenderContext) -> Result<String, PromptError> {
        let mut output = format!("# {}\n\n{}", self.metadata.name, self.metadata.description,);
        if let Some(version) = &self.metadata.version {
            output.push_str("\n\nPrompt version: ");
            output.push_str(version);
        }
        if !self.metadata.tags.is_empty() {
            output.push_str("\nPrompt tags: ");
            output.push_str(&self.metadata.tags.join(", "));
        }
        output.push_str(&format!(
            "\n\n## Identity\n{}\n\n## Instructions\n{}\n\n## Tool Guidelines\n{}\n\n## Safety and Constraints\n{}",
            self.substitute(&self.identity, context),
            self.substitute(&self.instructions, context),
            self.substitute(&self.tool_guidelines, context),
            self.substitute(&self.safety_and_constraints, context),
        ));
        for (name, content) in &self.custom_sections {
            output.push_str("\n\n## ");
            output.push_str(name);
            output.push('\n');
            output.push_str(&self.substitute(content, context));
        }
        info!(
            prompt.name = %self.metadata.name,
            prompt.outcome = "rendered",
            "Prompt template rendered"
        );
        Ok(output)
    }

    fn substitute(&self, text: &str, context: &PromptRenderContext) -> String {
        let mut rendered = text
            .replace("{workspace_path}", &context.workspace_path)
            .replace("{operating_system}", &context.operating_system)
            .replace("{runtime}", &context.runtime);
        for (key, value) in &self.variables {
            rendered = rendered.replace(&format!("{{{key}}}"), value);
        }
        for (key, value) in &context.variables {
            rendered = rendered.replace(&format!("{{{key}}}"), value);
        }
        rendered
    }
}

#[derive(Debug, Clone)]
struct ActivePromptState {
    name: String,
    template: PromptTemplate,
}

#[derive(Debug, Clone)]
pub struct PromptManager {
    directory: Arc<PathBuf>,
    active: Arc<RwLock<ActivePromptState>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromptConfig {
    #[serde(default)]
    pub active_prompt: Option<String>,
}

impl PromptManager {
    pub fn select_prompt_sync(
        &self,
        selection: &str,
        source: PromptChangeSource,
    ) -> Result<PromptSwitchOutcome, PromptError> {
        let (name, path) = self.resolve_selection(selection)?;
        let template = PromptTemplate::load(&path)?;
        let rendered = template.render(&PromptRenderContext::from_current_environment())?;
        let mut active = self.active.write().map_err(|error| {
            PromptError::Render(format!("Prompt manager lock is poisoned: {error}"))
        })?;
        *active = ActivePromptState {
            name: name.clone(),
            template,
        };
        info!(
            prompt.name = %name,
            prompt.path = %path.display(),
            prompt.source = %source,
            prompt.outcome = "switched",
            "Active system prompt switched"
        );
        Ok(PromptSwitchOutcome {
            name,
            path,
            rendered_prompt: rendered,
        })
    }
    #[instrument(name = "prompt.manager.load", skip(directory))]
    pub fn new(directory: impl AsRef<Path>) -> Result<Self, PromptError> {
        let directory = directory.as_ref();
        let directory = fs::canonicalize(directory).map_err(|error| {
            PromptError::Io(format!(
                "Failed to resolve prompt directory {}: {error}",
                directory.display()
            ))
        })?;
        if !directory.is_dir() {
            return Err(PromptError::NotFound(format!(
                "Prompt directory {} is not a directory",
                directory.display()
            )));
        }
        let default_path = directory.join("default.ron");
        let template = PromptTemplate::load(&default_path)?;
        let name = template.metadata.name.clone();
        Ok(Self {
            directory: Arc::new(directory),
            active: Arc::new(RwLock::new(ActivePromptState { name, template })),
        })
    }

    pub fn resolve_selection(&self, selection: &str) -> Result<(String, PathBuf), PromptError> {
        let selection = selection.trim();
        if selection.is_empty() {
            return Err(PromptError::InvalidSelection(
                "Prompt name must not be empty".to_owned(),
            ));
        }
        if selection.contains('\\')
            || selection.contains('/')
            || selection.starts_with('.')
            || Path::new(selection).is_absolute()
        {
            return Err(PromptError::PathTraversal(format!(
                "Prompt selection '{selection}' must be a filename inside the prompt directory"
            )));
        }
        let name = selection.strip_suffix(".ron").unwrap_or(selection);
        if name.is_empty() || name.contains("..") {
            return Err(PromptError::PathTraversal(format!(
                "Prompt selection '{selection}' contains traversal or is invalid"
            )));
        }
        if name
            .chars()
            .any(|character| !character.is_ascii_alphanumeric() && !matches!(character, '-' | '_'))
        {
            return Err(PromptError::InvalidSelection(format!(
                "Prompt name '{name}' contains invalid characters"
            )));
        }
        let path = self.directory.join(format!("{name}.ron"));
        let canonical = fs::canonicalize(&path).map_err(|error| {
            PromptError::NotFound(format!(
                "Prompt '{}' was not found in {}: {error}",
                name,
                self.directory.display()
            ))
        })?;
        if canonical
            .parent()
            .is_none_or(|parent| parent != self.directory.as_ref())
        {
            return Err(PromptError::PathTraversal(format!(
                "Prompt '{}' resolves outside the prompt directory",
                name
            )));
        }
        Ok((name.to_owned(), canonical))
    }

    pub fn list_available(&self) -> Result<Vec<String>, PromptError> {
        let entries = fs::read_dir(self.directory.as_ref()).map_err(|error| {
            PromptError::Io(format!("Failed to list prompt directory: {error}"))
        })?;
        let mut names = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|error| {
                PromptError::Io(format!("Failed to inspect prompt directory: {error}"))
            })?;
            let path = entry.path();
            if path.extension().is_some_and(|extension| extension == "ron")
                && let Some(name) = path.file_stem().and_then(|stem| stem.to_str())
            {
                names.push(name.to_owned());
            }
        }
        names.sort();
        info!(
            prompt.count = names.len(),
            "Available prompt templates listed"
        );
        Ok(names)
    }

    #[instrument(
        name = "prompt.switch",
        fields(prompt.selection = selection, prompt.source = %source)
    )]
    pub async fn select_prompt(
        &self,
        selection: &str,
        source: PromptChangeSource,
    ) -> Result<PromptSwitchOutcome, PromptError> {
        let (name, path) = self.resolve_selection(selection)?;
        let template = PromptTemplate::load(&path)?;
        let rendered = template.render(&PromptRenderContext::from_current_environment())?;
        let mut active = self.active.write().map_err(|error| {
            PromptError::Render(format!("Prompt manager lock is poisoned: {error}"))
        })?;
        *active = ActivePromptState {
            name: name.clone(),
            template,
        };
        info!(
            prompt.name = %name,
            prompt.path = %path.display(),
            prompt.source = %source,
            prompt.outcome = "switched",
            "Active system prompt switched"
        );
        Ok(PromptSwitchOutcome {
            name,
            path,
            rendered_prompt: rendered,
        })
    }

    pub fn active_name(&self) -> Result<String, PromptError> {
        self.active
            .read()
            .map(|active| active.name.clone())
            .map_err(|error| {
                PromptError::Render(format!("Prompt manager lock is poisoned: {error}"))
            })
    }

    pub fn render_active(&self) -> Result<String, PromptError> {
        let active = self.active.read().map_err(|error| {
            PromptError::Render(format!("Prompt manager lock is poisoned: {error}"))
        })?;
        active
            .template
            .render(&PromptRenderContext::from_current_environment())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};
    use tracing_subscriber::prelude::*;

    fn project_prompt_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("config/prompts")
    }

    #[test]
    fn renders_default_and_switches_to_concise() {
        crate::test_telemetry::run_test_case(
            "prompt::tests::renders_default_and_switches_to_concise",
            || {
                let manager = PromptManager::new(project_prompt_dir()).unwrap();
                let default = manager.render_active().unwrap();
                assert!(default.contains("## Identity"));
                assert!(default.contains("helpful local coding assistant"));

                let switched = manager
                    .select_prompt_sync("concise.ron", PromptChangeSource::ProgrammaticApi)
                    .unwrap();
                assert_eq!(switched.name, "concise");
                assert!(
                    switched
                        .rendered_prompt
                        .contains("concise local coding assistant")
                );
                assert_eq!(manager.active_name().unwrap(), "concise");
            },
        );
    }

    #[test]
    fn loads_snarky_joker_prompt() {
        crate::test_telemetry::run_test_case("prompt::tests::loads_snarky_joker_prompt", || {
            let path = project_prompt_dir().join("snarky_joker.ron");
            let template = PromptTemplate::load(path).unwrap();
            let rendered = template
                .render(&PromptRenderContext::from_current_environment())
                .unwrap();
            assert!(rendered.contains("The Glitch"));
            assert!(rendered.contains("Catchphrases"));
        });
    }

    #[test]
    fn rejects_prompt_path_traversal() {
        crate::test_telemetry::run_test_case(
            "prompt::tests::rejects_prompt_path_traversal",
            || {
                let manager = PromptManager::new(project_prompt_dir()).unwrap();
                for selection in ["../default", "/etc/passwd", r"..\default", "default/other"] {
                    assert!(matches!(
                        manager.resolve_selection(selection),
                        Err(PromptError::PathTraversal(_))
                    ));
                }
            },
        );
    }

    #[test]
    fn otel_switch_emits_prompt_span_in_memory() {
        let exporter = InMemorySpanExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        let subscriber = tracing_subscriber::registry()
            .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("otel-prompt-tests")));

        let manager = PromptManager::new(project_prompt_dir()).unwrap();
        let _subscriber_guard = tracing::subscriber::set_default(subscriber);
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            manager
                .select_prompt("concise", PromptChangeSource::CliFlag)
                .await
                .unwrap();
        });

        provider.force_flush().unwrap();
        let spans = exporter.get_finished_spans().unwrap();
        assert!(
            spans
                .iter()
                .any(|span| span.name.as_ref() == "prompt.switch")
        );
    }
}
