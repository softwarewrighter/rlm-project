//! LLM Provider abstraction and implementations

mod deepseek;
mod litellm;
mod ollama;

pub use deepseek::DeepSeekProvider;
pub use litellm::LiteLLMProvider;
pub use ollama::OllamaProvider;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur when interacting with an LLM provider
#[derive(Error, Debug)]
pub enum ProviderError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("JSON serialization/deserialization failed: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Provider returned error: {0}")]
    ProviderError(String),

    #[error("Connection failed: {0}")]
    ConnectionError(String),

    #[error("Timeout waiting for response")]
    Timeout,
}

/// Request to send to an LLM
#[derive(Debug, Clone, Serialize)]
pub struct LlmRequest {
    /// System prompt
    pub system: String,

    /// User message/prompt
    pub prompt: String,

    /// Temperature (0.0 - 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

impl LlmRequest {
    pub fn new(system: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            system: system.into(),
            prompt: prompt.into(),
            temperature: None,
            max_tokens: None,
        }
    }

    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }
}

/// Response from an LLM
#[derive(Debug, Clone, Deserialize)]
pub struct LlmResponse {
    /// The generated text
    pub content: String,

    /// Token usage statistics
    pub usage: Option<TokenUsage>,

    /// Time taken for generation (ms)
    pub duration_ms: Option<u64>,
}

/// Token usage statistics
#[derive(Debug, Clone, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Health status of a provider
#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub healthy: bool,
    pub latency_ms: Option<u64>,
    pub error: Option<String>,
}

/// Trait for LLM providers
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Get the provider name for logging/identification
    fn name(&self) -> &str;

    /// Get the model being used
    fn model(&self) -> &str;

    /// Send a completion request to the LLM
    async fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, ProviderError>;

    /// Check if the provider is healthy
    async fn health_check(&self) -> HealthStatus;
}
