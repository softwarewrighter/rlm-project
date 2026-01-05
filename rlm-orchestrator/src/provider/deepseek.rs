//! DeepSeek API provider implementation

use super::{HealthStatus, LlmProvider, LlmRequest, LlmResponse, ProviderError, TokenUsage};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// DeepSeek API provider
pub struct DeepSeekProvider {
    client: Client,
    api_key: String,
    model: String,
}

impl DeepSeekProvider {
    /// Create a new DeepSeek provider
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_model(api_key, "deepseek-chat")
    }

    /// Create with a specific model
    pub fn with_model(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(300))
                .build()
                .expect("Failed to create HTTP client"),
            api_key: api_key.into(),
            model: model.into(),
        }
    }
}

/// OpenAI-compatible chat request
#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    stream: bool,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

/// OpenAI-compatible chat response
#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
    usage: Option<ChatUsage>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Deserialize)]
struct ChatMessageResponse {
    content: String,
}

#[derive(Deserialize)]
struct ChatUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[async_trait]
impl LlmProvider for DeepSeekProvider {
    fn name(&self) -> &str {
        "deepseek"
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, ProviderError> {
        let url = "https://api.deepseek.com/chat/completions";

        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: request.system.clone(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: request.prompt.clone(),
            },
        ];

        let chat_request = ChatRequest {
            model: self.model.clone(),
            messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: false,
        };

        let start = Instant::now();

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&chat_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ProviderError::ProviderError(format!(
                "HTTP {}: {}",
                status, body
            )));
        }

        let chat_response: ChatResponse = response.json().await?;
        let duration_ms = start.elapsed().as_millis() as u64;

        let content = chat_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        let usage = chat_response.usage.map(|u| TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        Ok(LlmResponse {
            content,
            usage,
            duration_ms: Some(duration_ms),
        })
    }

    async fn health_check(&self) -> HealthStatus {
        // DeepSeek doesn't have a dedicated health endpoint
        // We could make a minimal request, but for now just check connectivity
        let url = "https://api.deepseek.com/models";
        let start = Instant::now();

        match self
            .client
            .get(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
        {
            Ok(response) => {
                let latency = start.elapsed().as_millis() as u64;
                if response.status().is_success() {
                    HealthStatus {
                        healthy: true,
                        latency_ms: Some(latency),
                        error: None,
                    }
                } else {
                    HealthStatus {
                        healthy: false,
                        latency_ms: Some(latency),
                        error: Some(format!("HTTP {}", response.status())),
                    }
                }
            }
            Err(e) => HealthStatus {
                healthy: false,
                latency_ms: None,
                error: Some(e.to_string()),
            },
        }
    }
}
