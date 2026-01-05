//! RLM Orchestrator - High-performance Recursive Language Model implementation
//!
//! This crate provides:
//! - Multiple LLM provider backends (Ollama, DeepSeek, OpenAI-compatible)
//! - Connection pooling with load balancing across distributed GPU servers
//! - Structured command execution (pure Rust, no Python dependency)
//! - REST API for external integration

pub mod api;
pub mod commands;
pub mod orchestrator;
pub mod pool;
pub mod provider;
pub mod wasm;

pub use orchestrator::RlmOrchestrator;
pub use pool::LlmPool;
pub use provider::{LlmProvider, LlmRequest, LlmResponse};

/// Configuration for the RLM system
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RlmConfig {
    /// Maximum iterations before giving up
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,

    /// Maximum sub-LM calls per session
    #[serde(default = "default_max_sub_calls")]
    pub max_sub_calls: usize,

    /// Output truncation limit (chars)
    #[serde(default = "default_output_limit")]
    pub output_limit: usize,

    /// Enable bypass for small contexts (skip RLM, send directly to LLM)
    #[serde(default = "default_bypass_enabled")]
    pub bypass_enabled: bool,

    /// Context size threshold for bypass (chars). Contexts smaller than this
    /// will be sent directly to the LLM without RLM iteration.
    /// Default: 4000 chars (~1000 tokens)
    #[serde(default = "default_bypass_threshold")]
    pub bypass_threshold: usize,

    /// Provider configuration
    pub providers: Vec<ProviderConfig>,
}

fn default_max_iterations() -> usize {
    20
}
fn default_max_sub_calls() -> usize {
    50
}
fn default_output_limit() -> usize {
    10000
}
fn default_bypass_enabled() -> bool {
    true
}
fn default_bypass_threshold() -> usize {
    4000
}

/// Configuration for a single LLM provider
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ProviderConfig {
    /// Provider type: "ollama", "deepseek", "openai"
    pub provider_type: String,

    /// Base URL for the provider
    pub base_url: String,

    /// Model name
    pub model: String,

    /// Optional API key
    pub api_key: Option<String>,

    /// Weight for load balancing (higher = more traffic)
    #[serde(default = "default_weight")]
    pub weight: u32,

    /// Role: "root", "sub", or "both"
    #[serde(default = "default_role")]
    pub role: String,
}

fn default_weight() -> u32 {
    1
}
fn default_role() -> String {
    "both".to_string()
}
