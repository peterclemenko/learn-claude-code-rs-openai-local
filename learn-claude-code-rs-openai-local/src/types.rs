use std::{fs, path::Path};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct AgentRonConfig {
    pub name: String,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub system_prompt: String,
    pub temperature: f32,
    pub max_turns: usize,
    #[serde(default)]
    pub telemetry: TelemetryConfig,
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

#[cfg(test)]
mod tests {
    use super::{AgentRonConfig, parse_toon_field, toon_string, toon_table};

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
        assert!(!config.telemetry.enabled);
    }
}
