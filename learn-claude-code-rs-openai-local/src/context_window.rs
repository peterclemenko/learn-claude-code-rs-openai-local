use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::Value;

#[async_trait]
pub trait ContextWindowProvider: Send + Sync {
    fn provider_name(&self) -> &'static str;

    async fn context_window_tokens(&self, model: &str) -> Result<Option<u64>>;
}

pub struct OllamaContextWindowProvider {
    base_url: String,
    client: reqwest::Client,
}

impl OllamaContextWindowProvider {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl ContextWindowProvider for OllamaContextWindowProvider {
    fn provider_name(&self) -> &'static str {
        "ollama"
    }

    async fn context_window_tokens(&self, model: &str) -> Result<Option<u64>> {
        let response = self
            .client
            .get(format!("{}/api/ps", self.base_url))
            .send()
            .await
            .context("Failed to query Ollama running models")?
            .error_for_status()
            .context("Ollama /api/ps returned an unsuccessful status")?
            .json::<Value>()
            .await
            .context("Failed to parse Ollama /api/ps response")?;

        Ok(ollama_context_window(&response, model))
    }
}

fn ollama_context_window(response: &Value, model: &str) -> Option<u64> {
    response
        .get("models")?
        .as_array()?
        .iter()
        .find(|entry| {
            let running_model = entry
                .get("name")
                .or_else(|| entry.get("model"))
                .and_then(Value::as_str);
            running_model.is_some_and(|running_model| same_model(running_model, model))
        })?
        .get("context_length")?
        .as_u64()
        .filter(|context_length| *context_length > 0)
}

fn same_model(running_model: &str, requested_model: &str) -> bool {
    running_model == requested_model
        || running_model.strip_suffix(":latest") == Some(requested_model)
        || requested_model.strip_suffix(":latest") == Some(running_model)
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read, Write},
        net::TcpListener,
        thread,
    };

    use serde_json::json;

    use super::{
        ContextWindowProvider, OllamaContextWindowProvider, ollama_context_window, same_model,
    };

    #[test]
    fn reads_allocated_context_for_matching_running_model() {
        let response = json!({
            "models": [
                {"name": "other:latest", "context_length": 4096},
                {"name": "local-model:latest", "context_length": 32768}
            ]
        });
        assert_eq!(
            ollama_context_window(&response, "local-model:latest"),
            Some(32768)
        );
        assert_eq!(ollama_context_window(&response, "local-model"), Some(32768));
        assert_eq!(ollama_context_window(&response, "not-loaded"), None);
    }

    #[test]
    fn rejects_missing_or_invalid_context_lengths() {
        assert_eq!(ollama_context_window(&json!({"models": []}), "model"), None);
        assert_eq!(
            ollama_context_window(
                &json!({"models": [{"name": "model", "context_length": 0}]}),
                "model"
            ),
            None
        );
    }

    #[test]
    fn only_equates_unqualified_names_with_the_latest_tag() {
        assert!(same_model("model:latest", "model"));
        assert!(!same_model("model:7b", "model"));
        assert!(!same_model("model-v2:latest", "model-v2-extra"));
    }

    #[tokio::test]
    async fn ollama_provider_reads_active_context_size_from_api() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (request_sender, request_receiver) = std::sync::mpsc::channel();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0; 1024];
            let length = stream.read(&mut request).unwrap();
            request_sender
                .send(String::from_utf8_lossy(&request[..length]).into_owned())
                .unwrap();
            let body = r#"{"models":[{"name":"test-model:latest","context_length":16384}]}"#;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )
            .unwrap();
        });

        let provider = OllamaContextWindowProvider::new(format!("http://{address}"));
        assert_eq!(
            provider.context_window_tokens("test-model").await.unwrap(),
            Some(16384)
        );
        assert!(request_receiver.recv().unwrap().starts_with("GET /api/ps "));
        server.join().unwrap();
    }
}
