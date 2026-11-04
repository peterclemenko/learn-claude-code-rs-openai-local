use std::{fmt, fs, path::Path, str::FromStr};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

impl fmt::Display for TodoStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Pending => "Pending",
            Self::InProgress => "InProgress",
            Self::Completed => "Completed",
            Self::Failed => "Failed",
        })
    }
}

impl FromStr for TodoStatus {
    type Err = anyhow::Error;

    fn from_str(status: &str) -> Result<Self> {
        match status.trim().to_ascii_lowercase().as_str() {
            "pending" => Ok(Self::Pending),
            "inprogress" => Ok(Self::InProgress),
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            _ => bail!("Unknown todo status '{status}'"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TodoItem {
    pub id: usize,
    pub description: String,
    pub status: TodoStatus,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TodoState {
    pub tasks: Vec<TodoItem>,
}

impl TodoState {
    pub fn render_toon_context(&self) -> String {
        let header = format!("tasks[{}]{{id,description,status}}:", self.tasks.len());
        if self.tasks.is_empty() {
            return header;
        }

        let rows = self
            .tasks
            .iter()
            .map(|task| {
                format!(
                    "{},{},{}",
                    task.id,
                    toon_cell(&task.description),
                    task.status
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("{header}\n{rows}")
    }
}

fn toon_cell(value: &str) -> String {
    if value.is_empty()
        || value.trim() != value
        || value.chars().any(|character| {
            matches!(character, ',' | '"' | '\\' | '\n' | '\r' | '\t') || character.is_control()
        })
    {
        toon_string(value)
    } else {
        value.to_owned()
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct AgentRonConfig {
    pub name: String,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub system_prompt: String,
    pub temperature: f32,
    pub max_turns: usize,
    #[serde(default = "default_ollama_base_url")]
    pub ollama_base_url: String,
    #[serde(default)]
    pub telemetry: TelemetryConfig,
}

fn default_ollama_base_url() -> String {
    "http://localhost:11434".to_owned()
}

#[derive(Debug, Deserialize, Clone)]
pub struct TelemetryConfig {
    pub enabled: bool,
    pub service_name: String,
    pub otlp_endpoint: String,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            service_name: "local-agent".to_owned(),
            otlp_endpoint: "http://localhost:4317".to_owned(),
        }
    }
}

impl AgentRonConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path)
            .with_context(|| format!("Failed to read agent config {}", path.display()))?;
        let config: Self = ron::from_str(&contents)
            .with_context(|| format!("Failed to parse RON agent config {}", path.display()))?;

        if config.name.trim().is_empty() {
            bail!("Agent name must not be empty");
        }
        if config.base_url.trim().is_empty() {
            bail!("Agent base_url must not be empty");
        }
        if config.model.trim().is_empty() {
            bail!("Agent model must not be empty");
        }
        if config.ollama_base_url.trim().is_empty() {
            bail!("Agent ollama_base_url must not be empty");
        }
        if !config.temperature.is_finite() {
            bail!("Agent temperature must be finite");
        }
        if config.max_turns == 0 {
            bail!("Agent max_turns must be greater than zero");
        }

        Ok(config)
    }
}

pub fn parse_toon_field(payload: &str, tool: &str, field: &str) -> Result<String> {
    let payload = payload.trim();
    if payload.starts_with('{') {
        let object: serde_json::Value =
            serde_json::from_str(payload).context("Invalid JSON tool arguments")?;
        let value = object
            .get(field)
            .with_context(|| format!("JSON tool arguments are missing field '{field}'"))?;
        return json_scalar(value, field);
    }

    let (header, inline_value) = payload
        .split_once(':')
        .with_context(|| format!("Expected a TOON header for {tool}[1]{{{field}}}"))?;

    let row_value = if let Some(open) = header.find('{') {
        let close = header[open + 1..]
            .find('}')
            .map(|index| open + 1 + index)
            .context("Malformed TOON table header")?;
        let tool_name = header[..open].split('[').next().unwrap_or_default().trim();
        if tool_name != tool {
            bail!("Expected TOON tool '{tool}', received '{tool_name}'");
        }
        let fields = header[open + 1..close]
            .split(',')
            .map(str::trim)
            .collect::<Vec<_>>();
        let field_index = fields
            .iter()
            .position(|candidate| *candidate == field)
            .with_context(|| format!("TOON payload is missing field '{field}'"))?;
        let mut row = inline_value.trim().to_owned();
        if row.is_empty() {
            row = payload
                .split_once(':')
                .map(|(_, rest)| rest.trim())
                .unwrap_or_default()
                .to_owned();
        }
        if row.is_empty() {
            bail!("TOON table for '{tool}' has no data row");
        }

        let values = split_toon_row(&row)?;
        values
            .get(field_index)
            .with_context(|| format!("TOON row is missing value for '{field}'"))?
            .to_owned()
    } else {
        if header.trim() != field {
            bail!(
                "Expected TOON field '{field}', received '{}'",
                header.trim()
            );
        }
        let mut value = inline_value.trim().to_owned();
        if value.is_empty() {
            value = payload
                .split_once(':')
                .map(|(_, rest)| rest.trim())
                .unwrap_or_default()
                .to_owned();
        }
        value
    };

    decode_toon_string(row_value)
}

fn split_toon_row(row: &str) -> Result<Vec<String>> {
    let mut values = Vec::new();
    let mut value = String::new();
    let mut quoted = false;
    let mut escaped = false;

    for character in row.chars() {
        if escaped {
            value.push(character);
            escaped = false;
            continue;
        }
        match character {
            '\\' if quoted => {
                value.push(character);
                escaped = true;
            }
            '"' => {
                value.push(character);
                quoted = !quoted;
            }
            ',' | '\t' if !quoted => {
                values.push(value.trim().to_owned());
                value.clear();
            }
            _ => value.push(character),
        }
    }
    if quoted {
        bail!("Unterminated quoted string in TOON row");
    }
    values.push(value.trim().to_owned());
    Ok(values)
}

fn decode_toon_string(value: String) -> Result<String> {
    if value.starts_with('"') {
        serde_json::from_str(&value).context("Invalid quoted string in TOON payload")
    } else {
        Ok(value)
    }
}

fn json_scalar(value: &serde_json::Value, field: &str) -> Result<String> {
    match value {
        serde_json::Value::String(value) => Ok(value.clone()),
        serde_json::Value::Number(value) => Ok(value.to_string()),
        serde_json::Value::Bool(value) => Ok(value.to_string()),
        _ => bail!("JSON tool argument '{field}' must be a scalar value"),
    }
}

pub fn toon_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character.is_control() => {
                use std::fmt::Write as _;
                let _ = write!(escaped, "\\u{:04x}", character as u32);
            }
            character => escaped.push(character),
        }
    }
    escaped.push('"');
    escaped
}

pub fn toon_table(name: &str, columns: &[&str], row: &[String]) -> String {
    let values = row
        .iter()
        .map(|value| toon_string(value))
        .collect::<Vec<_>>()
        .join(",");
    format!("{name}[1]{{{}}}:\n{values}", columns.join(","))
}

pub(crate) fn parse_toon_table(
    payload: &str,
    expected_name: &str,
    expected_columns: &[&str],
) -> Result<Vec<Vec<String>>> {
    let payload = payload.trim();
    if payload.starts_with('{') || payload.starts_with('[') {
        let value: serde_json::Value =
            serde_json::from_str(payload).context("Invalid JSON tool arguments")?;
        let tasks = match &value {
            serde_json::Value::Array(tasks) => tasks,
            serde_json::Value::Object(object) => match object.get("tasks") {
                Some(serde_json::Value::Array(tasks)) => tasks,
                Some(_) => bail!("JSON tool argument 'tasks' must be an array"),
                None if expected_columns
                    .iter()
                    .all(|column| object.contains_key(*column)) =>
                {
                    return Ok(vec![
                        expected_columns
                            .iter()
                            .map(|column| json_scalar(&object[*column], column))
                            .collect::<Result<Vec<_>>>()?,
                    ]);
                }
                None => bail!("JSON tool arguments are missing the '{expected_name}' task list"),
            },
            _ => bail!("JSON tool arguments must be an object or array"),
        };
        return tasks
            .iter()
            .map(|task| {
                let object = task
                    .as_object()
                    .context("Each JSON todo task must be an object")?;
                expected_columns
                    .iter()
                    .map(|column| {
                        let value = object
                            .get(*column)
                            .with_context(|| format!("JSON todo task is missing '{column}'"))?;
                        json_scalar(value, column)
                    })
                    .collect()
            })
            .collect();
    }

    let (header, data) = payload
        .split_once(':')
        .context("Expected a TOON table header")?;
    let (name, definition) = header
        .trim()
        .split_once('[')
        .context("Malformed TOON table header")?;
    if name.trim() != expected_name {
        bail!(
            "Expected TOON table '{expected_name}', received '{}'",
            name.trim()
        );
    }

    let (count, columns) = definition
        .split_once("]{")
        .context("Malformed TOON table header")?;
    let count = count
        .parse::<usize>()
        .context("TOON row count must be a non-negative integer")?;
    let columns = columns
        .strip_suffix('}')
        .context("Malformed TOON table columns")?
        .split(',')
        .map(str::trim)
        .collect::<Vec<_>>();
    if columns != expected_columns {
        bail!(
            "Expected TOON columns {{{}}}, received {{{}}}",
            expected_columns.join(","),
            columns.join(",")
        );
    }

    let rows = data
        .lines()
        .map(str::trim)
        .filter(|row| !row.is_empty())
        .collect::<Vec<_>>();
    if rows.len() != count {
        bail!(
            "TOON table declares {count} rows but contains {}",
            rows.len()
        );
    }

    rows.into_iter()
        .map(|row| {
            let values = split_toon_row(row)?;
            if values.len() != expected_columns.len() {
                bail!(
                    "TOON row has {} values; expected {}",
                    values.len(),
                    expected_columns.len()
                );
            }
            values.into_iter().map(decode_toon_string).collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        AgentRonConfig, TodoItem, TodoState, TodoStatus, parse_toon_field, parse_toon_table,
        toon_string, toon_table,
    };

    #[test]
    fn renders_todo_state_as_compact_toon() {
        let state = TodoState {
            tasks: vec![
                TodoItem {
                    id: 1,
                    description: "Audit file tree".to_owned(),
                    status: TodoStatus::Completed,
                },
                TodoItem {
                    id: 2,
                    description: "Execute compile test".to_owned(),
                    status: TodoStatus::InProgress,
                },
            ],
        };
        assert_eq!(
            state.render_toon_context(),
            "tasks[2]{id,description,status}:\n1,Audit file tree,Completed\n2,Execute compile test,InProgress"
        );
    }

    #[test]
    fn parses_todo_rows_and_escaped_descriptions() {
        let rows = parse_toon_table(
            "todo_write[1]{id,description,status}:\n1,\"Audit, then test\",Pending",
            "todo_write",
            &["id", "description", "status"],
        )
        .unwrap();
        assert_eq!(rows, vec![vec!["1", "Audit, then test", "Pending"]]);
        assert_eq!(
            "InProgress".parse::<TodoStatus>().unwrap(),
            TodoStatus::InProgress
        );
        assert!("unknown".parse::<TodoStatus>().is_err());
        let json_rows = parse_toon_table(
            r#"{"tasks":[{"id":2,"description":"Run tests","status":"InProgress"}]}"#,
            "todo_write",
            &["id", "description", "status"],
        )
        .unwrap();
        assert_eq!(json_rows, vec![vec!["2", "Run tests", "InProgress"]]);
    }

    #[test]
    fn parses_inline_and_quoted_toon_fields() {
        assert_eq!(
            parse_toon_field("bash[1]{command}: echo hello", "bash", "command").unwrap(),
            "echo hello"
        );
        assert_eq!(
            parse_toon_field(
                "read_file[1]{path}: \"a path/file.txt\"",
                "read_file",
                "path"
            )
            .unwrap(),
            "a path/file.txt"
        );
        assert_eq!(
            parse_toon_field(r#"{"command":"cargo test"}"#, "bash", "command").unwrap(),
            "cargo test"
        );
    }

    #[test]
    fn serializes_toon_table_rows() {
        let result = toon_table(
            "exec_result",
            &["exit_code", "stdout", "stderr"],
            &["0".to_owned(), "hello\n".to_owned(), String::new()],
        );
        assert_eq!(
            result,
            "exec_result[1]{exit_code,stdout,stderr}:\n\"0\",\"hello\\n\",\"\""
        );
        assert_eq!(toon_string("a\"b"), "\"a\\\"b\"");
    }

    #[test]
    fn loads_slotted_ron_configuration() {
        let config: AgentRonConfig = ron::from_str(
            r#"AgentRonConfig(
                name: "test",
                base_url: "http://localhost:11434/v1",
                api_key: "local",
                model: "test-model",
                system_prompt: "Be helpful.",
                temperature: 0.2,
                max_turns: 4,
            )"#,
        )
        .unwrap();
        assert_eq!(config.model, "test-model");
        assert_eq!(config.ollama_base_url, "http://localhost:11434");
        assert!(!config.telemetry.enabled);
    }
}
