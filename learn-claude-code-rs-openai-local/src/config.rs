use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct OtelConfig {
    pub service_name: String,
    pub endpoint: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    pub model: String,
    pub endpoint: String,
    pub api_key: String,
    pub temperature: f32,
    pub max_tokens: Option<u32>,
    pub otel: OtelConfig,
}

impl AppConfig {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: AppConfig = ron::from_str(&content)?;
        Ok(config)
    }
}