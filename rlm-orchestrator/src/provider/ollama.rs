//! Ollama LLM provider implementation

use super::{HealthStatus, LlmProvider, LlmRequest, LlmResponse, ProviderError, TokenUsage};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Ollama provider for local or remote Ollama servers
pub struct OllamaProvider {
    client: Client,
    base_url: String,
    model: String,
    name: String,
}

impl OllamaProvider {
    /// Create a new Ollama provider
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        let base_url = base_url.into();
        let model = model.into();
        let name = format!("ollama:{}", model);

        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(300))
                .build()
                .expect("Failed to create HTTP client"),
            base_url,
            model,
            name,
        }
    }

    /// Create with a custom name (useful for identifying distributed servers)
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

/// Ollama API request format
#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    system: String,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>,
}

/// Ollama API response format
#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
    #[serde(default)]
    eval_count: Option<u32>,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    total_duration: Option<u64>,
}

/// Ollama tags response (for health check)
#[derive(Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModel>,
}

#[derive(Deserialize)]
struct OllamaModel {
    name: String,
}

#[async_trait]
impl LlmProvider for OllamaProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, ProviderError> {
        let url = format!("{}/api/generate", self.base_url);

        let ollama_request = OllamaRequest {
            model: self.model.clone(),
            prompt: request.prompt.clone(),
            system: request.system.clone(),
            stream: false,
            options: OllamaOptions {
                temperature: request.temperature,
                num_predict: request.max_tokens,
            },
        };

        let start = Instant::now();

        let response = self
            .client
            .post(&url)
            .json(&ollama_request)
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

        let ollama_response: OllamaResponse = response.json().await?;

        // Use Ollama's reported duration (nanoseconds) or fall back to our measurement
        let duration_ms = ollama_response.total_duration
            .map(|ns| ns / 1_000_000)
            .unwrap_or_else(|| start.elapsed().as_millis() as u64);

        let usage = match (ollama_response.prompt_eval_count, ollama_response.eval_count) {
            (Some(prompt), Some(completion)) => Some(TokenUsage {
                prompt_tokens: prompt,
                completion_tokens: completion,
                total_tokens: prompt + completion,
            }),
            _ => None,
        };

        Ok(LlmResponse {
            content: ollama_response.response,
            usage,
            duration_ms: Some(duration_ms),
        })
    }

    async fn health_check(&self) -> HealthStatus {
        let url = format!("{}/api/tags", self.base_url);
        let start = Instant::now();

        match self.client.get(&url).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    let latency = start.elapsed().as_millis() as u64;

                    // Check if our model is available
                    if let Ok(tags) = response.json::<OllamaTagsResponse>().await {
                        let model_available = tags
                            .models
                            .iter()
                            .any(|m| m.name == self.model || m.name.starts_with(&self.model));

                        if model_available {
                            HealthStatus {
                                healthy: true,
                                latency_ms: Some(latency),
                                error: None,
                            }
                        } else {
                            HealthStatus {
                                healthy: false,
                                latency_ms: Some(latency),
                                error: Some(format!("Model {} not found", self.model)),
                            }
                        }
                    } else {
                        HealthStatus {
                            healthy: true,
                            latency_ms: Some(latency),
                            error: None,
                        }
                    }
                } else {
                    HealthStatus {
                        healthy: false,
                        latency_ms: None,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = OllamaProvider::new("http://localhost:11434", "qwen2.5-coder:14b");
        assert_eq!(provider.model(), "qwen2.5-coder:14b");
        assert!(provider.name().contains("ollama"));
    }
}
