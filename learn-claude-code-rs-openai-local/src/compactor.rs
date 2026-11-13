use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use anyhow::{Context, Result, bail};
use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
};
use serde::{Deserialize, Serialize};

use crate::{agent::ChatCompletionClient, types::toon_table};

const SUMMARY_NAME: &str = "context_compactor";
const SUMMARY_MARKER: &str = "[Compacted Historical Context Summary]";

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CompactionConfig {
    pub trigger_threshold_ratio: f32,
    pub max_context_tokens: usize,
    pub protected_recent_turns: usize,
    pub compaction_system_prompt: String,
    pub target_summary_tokens: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            trigger_threshold_ratio: 0.8,
            max_context_tokens: 16_384,
            protected_recent_turns: 4,
            compaction_system_prompt: "Summarize the previous conversational turns concisely, retaining key technical decisions, completed code changes, file paths, active hypotheses, and user constraints. Preserve active task state and do not invent facts. Discard redundant back-and-forth.".to_owned(),
            target_summary_tokens: 1_024,
        }
    }
}

impl CompactionConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read compaction config {}", path.display()))?;
        let config: Self = ron::from_str(&contents)
            .with_context(|| format!("Failed to parse RON compaction config {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if !self.trigger_threshold_ratio.is_finite()
            || !(0.0..=1.0).contains(&self.trigger_threshold_ratio)
            || self.trigger_threshold_ratio == 0.0
        {
            bail!("Compaction trigger_threshold_ratio must be finite and in (0, 1]");
        }
        if self.max_context_tokens == 0 {
            bail!("Compaction max_context_tokens must be greater than zero");
        }
        if self.protected_recent_turns == 0 {
            bail!("Compaction protected_recent_turns must be greater than zero");
        }
        if self.compaction_system_prompt.trim().is_empty() {
            bail!("Compaction compaction_system_prompt must not be empty");
        }
        if self.target_summary_tokens == 0 || self.target_summary_tokens > u16::MAX as usize {
            bail!(
                "Compaction target_summary_tokens must be in 1..={}",
                u16::MAX
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct CompactionStateSnapshot {
    pub epoch: u64,
    pub original_tokens: usize,
    pub compacted_tokens: usize,
    pub summary_hash: String,
}

impl CompactionStateSnapshot {
    pub fn render_toon(&self) -> String {
        toon_table(
            "compaction_state",
            &[
                "epoch",
                "original_tokens",
                "compacted_tokens",
                "summary_hash",
            ],
            &[
                self.epoch.to_string(),
                self.original_tokens.to_string(),
                self.compacted_tokens.to_string(),
                self.summary_hash.clone(),
            ],
        )
    }
}

pub(crate) struct ContextCompactor {
    config: CompactionConfig,
    client: Arc<dyn ChatCompletionClient>,
    model: String,
    epoch: AtomicU64,
}

impl ContextCompactor {
    pub(crate) fn new(
        config: CompactionConfig,
        client: Arc<dyn ChatCompletionClient>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            config,
            client,
            model: model.into(),
            epoch: AtomicU64::new(0),
        }
    }

    pub fn estimate_tokens(messages: &[ChatCompletionRequestMessage]) -> Result<usize> {
        messages.iter().try_fold(0usize, |total, message| {
            let serialized =
                serde_json::to_vec(message).context("Failed to estimate chat message tokens")?;
            let message_tokens = serialized.len().div_ceil(4).saturating_add(12);
            Ok(total.saturating_add(message_tokens))
        })
    }

    pub(crate) async fn compact_if_needed(
        &self,
        messages: &mut Vec<ChatCompletionRequestMessage>,
    ) -> Result<Option<CompactionStateSnapshot>> {
        let original_tokens = Self::estimate_tokens(messages)?;
        let threshold = (self.config.max_context_tokens as f64
            * f64::from(self.config.trigger_threshold_ratio))
        .ceil() as usize;
        if original_tokens < threshold {
            return Ok(None);
        }

        let user_turn_starts = messages
            .iter()
            .enumerate()
            .filter_map(|(index, message)| {
                matches!(message, ChatCompletionRequestMessage::User(_)).then_some(index)
            })
            .collect::<Vec<_>>();
        let protected_turns = self.config.protected_recent_turns;
        if user_turn_starts.len() <= protected_turns {
            return Ok(None);
        }
        let history_end = user_turn_starts[user_turn_starts.len() - protected_turns];

        let history_to_summarize = messages[..history_end]
            .iter()
            .filter(|message| {
                !matches!(message, ChatCompletionRequestMessage::System(system)
                    if system.name.as_deref() != Some(SUMMARY_NAME))
            })
            .cloned()
            .collect::<Vec<_>>();
        if history_to_summarize.is_empty() {
            return Ok(None);
        }

        let mut summary_request_messages = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content(self.config.compaction_system_prompt.as_str())
                .build()?
                .into(),
        ];
        summary_request_messages.extend(history_to_summarize);
        summary_request_messages.push(
            ChatCompletionRequestUserMessageArgs::default()
                .content(format!(
                    "Produce a concise TOON-friendly historical summary within {} tokens. Retain essential decisions, paths, constraints, unresolved work, and active hypotheses.",
                    self.config.target_summary_tokens
                ))
                .build()?
                .into(),
        );

        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.model)
            .messages(summary_request_messages)
            .max_tokens(self.config.target_summary_tokens as u16)
            .temperature(0.1_f32)
            .build()?;
        let response = self
            .client
            .complete(request)
            .await
            .context("Context compaction request to local OpenAI-compatible endpoint failed")?;
        let summary = response
            .choices
            .first()
            .and_then(|choice| choice.message.content.as_deref())
            .map(str::trim)
            .filter(|summary| !summary.is_empty())
            .context("Context compaction model returned no summary")?
            .to_owned();

        let mut compacted = messages
            .iter()
            .filter(|message| {
                matches!(message, ChatCompletionRequestMessage::System(system)
                    if system.name.as_deref() != Some(SUMMARY_NAME))
            })
            .cloned()
            .collect::<Vec<_>>();
        compacted.push(
            ChatCompletionRequestSystemMessageArgs::default()
                .content(format!("{SUMMARY_MARKER}:\n{summary}"))
                .name(SUMMARY_NAME)
                .build()?
                .into(),
        );
        compacted.extend(
            messages[history_end..]
                .iter()
                .filter(|message| !matches!(message, ChatCompletionRequestMessage::System(_)))
                .cloned(),
        );
        let compacted_tokens = Self::estimate_tokens(&compacted)?;
        if compacted_tokens >= original_tokens {
            return Ok(None);
        }
        let snapshot = CompactionStateSnapshot {
            epoch: self.epoch.fetch_add(1, Ordering::Relaxed) + 1,
            original_tokens,
            compacted_tokens,
            summary_hash: summary_hash(&summary),
        };
        *messages = compacted;
        Ok(Some(snapshot))
    }
}

fn summary_hash(summary: &str) -> String {
    let hash = summary.bytes().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    });
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::{Arc, Mutex},
    };

    use anyhow::{Context, Result};
    use async_openai::types::chat::{
        ChatCompletionRequestSystemMessageContent, CreateChatCompletionRequest,
        CreateChatCompletionResponse,
    };
    use async_trait::async_trait;
    use serde_json::json;

    use super::{CompactionConfig, CompactionStateSnapshot, ContextCompactor, SUMMARY_NAME};
    use crate::agent::ChatCompletionClient;

    #[derive(Default)]
    struct MockChatClient {
        responses: Mutex<VecDeque<CreateChatCompletionResponse>>,
        requests: Mutex<Vec<CreateChatCompletionRequest>>,
    }

    impl MockChatClient {
        fn new(responses: Vec<CreateChatCompletionResponse>) -> Self {
            Self {
                responses: Mutex::new(responses.into()),
                requests: Mutex::new(Vec::new()),
            }
        }

        fn requests(&self) -> Vec<CreateChatCompletionRequest> {
            self.requests.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl ChatCompletionClient for MockChatClient {
        async fn complete(
            &self,
            request: CreateChatCompletionRequest,
        ) -> Result<CreateChatCompletionResponse> {
            self.requests.lock().unwrap().push(request);
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .context("Mock chat response queue is empty")
        }
    }

    fn response(content: &str) -> CreateChatCompletionResponse {
        serde_json::from_value(json!({
            "id": "compaction-summary",
            "object": "chat.completion",
            "created": 1,
            "model": "local-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }]
        }))
        .unwrap()
    }

    fn message(
        role: &str,
        content: &str,
    ) -> async_openai::types::chat::ChatCompletionRequestMessage {
        serde_json::from_value(json!({"role": role, "content": content})).unwrap()
    }

    fn config() -> CompactionConfig {
        CompactionConfig {
            trigger_threshold_ratio: 0.01,
            max_context_tokens: 100,
            protected_recent_turns: 1,
            compaction_system_prompt: "Summarize historical context.".to_owned(),
            target_summary_tokens: 32,
        }
    }

    #[tokio::test]
    async fn compaction_preserves_system_and_recent_turns_and_rolls_summary_forward() {
        let client = Arc::new(MockChatClient::new(vec![
            response("decided: keep the API stable"),
            response("decided: keep the API stable; next step: test it"),
        ]));
        let compactor = ContextCompactor::new(config(), client.clone(), "local-model");
        let mut messages = vec![
            message("system", "Keep these instructions."),
            message("user", &"old question ".repeat(100)),
            message("assistant", &"old answer ".repeat(100)),
            message("user", "latest question"),
            message("assistant", "latest answer"),
        ];

        let snapshot = compactor
            .compact_if_needed(&mut messages)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(
            messages.len(),
            4,
            "system prompt, summary, and the protected latest user/assistant pair should remain"
        );
        assert!(matches!(
            &messages[0],
            async_openai::types::chat::ChatCompletionRequestMessage::System(system)
                if system.content == ChatCompletionRequestSystemMessageContent::Text(
                    "Keep these instructions.".to_owned()
                )
        ));
        assert!(matches!(
            &messages[1],
            async_openai::types::chat::ChatCompletionRequestMessage::System(system)
                if system.name.as_deref() == Some(SUMMARY_NAME)
                    && matches!(
                        &system.content,
                        ChatCompletionRequestSystemMessageContent::Text(content)
                            if content.contains("decided: keep the API stable")
                    )
        ));
        assert!(matches!(
            &messages[2],
            async_openai::types::chat::ChatCompletionRequestMessage::User(_)
        ));
        let request = &client.requests()[0];
        assert_eq!(request.model, "local-model");
        assert_eq!(
            serde_json::to_value(request).unwrap()["max_tokens"],
            json!(32)
        );
        assert_eq!(snapshot.epoch, 1);
        assert!(snapshot.compacted_tokens < snapshot.original_tokens);

        messages.push(message("user", "follow-up question"));
        messages.push(message("assistant", "follow-up answer"));
        compactor
            .compact_if_needed(&mut messages)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(messages.len(), 4);
        assert_eq!(
            messages
                .iter()
                .filter(|message| matches!(
                    message,
                    async_openai::types::chat::ChatCompletionRequestMessage::System(system)
                        if system.name.as_deref() == Some(SUMMARY_NAME)
                ))
                .count(),
            1
        );
        assert!(client.requests()[1].messages.iter().any(|message| matches!(
            message,
            async_openai::types::chat::ChatCompletionRequestMessage::System(system)
                if system.name.as_deref() == Some(SUMMARY_NAME)
        )));
    }

    #[tokio::test]
    async fn does_not_call_model_when_below_threshold_or_without_older_turns() {
        let client = Arc::new(MockChatClient::default());
        let compactor = ContextCompactor::new(
            CompactionConfig {
                trigger_threshold_ratio: 1.0,
                max_context_tokens: usize::MAX,
                ..config()
            },
            client.clone(),
            "local-model",
        );
        let mut messages = vec![message("user", "one turn"), message("assistant", "answer")];

        assert!(
            compactor
                .compact_if_needed(&mut messages)
                .await
                .unwrap()
                .is_none()
        );
        assert!(client.requests().is_empty());
        assert_eq!(messages.len(), 2);

        let triggered_compactor = ContextCompactor::new(
            CompactionConfig {
                trigger_threshold_ratio: 0.01,
                max_context_tokens: 100,
                ..config()
            },
            client.clone(),
            "local-model",
        );
        assert!(
            triggered_compactor
                .compact_if_needed(&mut messages)
                .await
                .unwrap()
                .is_none()
        );
        assert!(client.requests().is_empty());
    }

    #[test]
    fn compaction_snapshot_uses_toon_table_format() {
        let snapshot = CompactionStateSnapshot {
            epoch: 2,
            original_tokens: 9000,
            compacted_tokens: 3000,
            summary_hash: "abcd1234".to_owned(),
        };
        assert_eq!(
            snapshot.render_toon(),
            "compaction_state[1]{epoch,original_tokens,compacted_tokens,summary_hash}:\n\"2\",\"9000\",\"3000\",\"abcd1234\""
        );
    }

    #[test]
    fn loads_repository_compaction_ron() {
        let config = CompactionConfig::load(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("config/compaction.ron"),
        )
        .unwrap();
        assert_eq!(config.max_context_tokens, 16_384);
    }
}
