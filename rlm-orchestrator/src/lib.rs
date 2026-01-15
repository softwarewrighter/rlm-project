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

pub use orchestrator::{ProgressCallback, ProgressEvent, RlmOrchestrator};
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

    /// WASM execution configuration
    #[serde(default)]
    pub wasm: WasmConfig,
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

/// Configuration for WASM execution features
#[derive(Debug, Clone, serde::Deserialize)]
pub struct WasmConfig {
    /// Enable WASM execution features
    #[serde(default = "default_wasm_enabled")]
    pub enabled: bool,

    /// Enable rust_wasm command (requires rustc with wasm32-unknown-unknown target)
    #[serde(default = "default_rust_wasm_enabled")]
    pub rust_wasm_enabled: bool,

    /// Maximum WASM instructions (fuel limit)
    #[serde(default = "default_fuel_limit")]
    pub fuel_limit: u64,

    /// Maximum WASM memory in bytes
    #[serde(default = "default_memory_limit")]
    pub memory_limit: usize,

    /// Directory for WASM module cache (None = memory-only)
    pub cache_dir: Option<String>,

    /// Maximum memory cache entries
    #[serde(default = "default_cache_size")]
    pub cache_size: usize,

    /// Path to rustc binary (None = auto-detect)
    pub rustc_path: Option<String>,
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rust_wasm_enabled: true,
            fuel_limit: 1_000_000,
            memory_limit: 64 * 1024 * 1024,
            cache_dir: None,
            cache_size: 100,
            rustc_path: None,
        }
    }
}

fn default_wasm_enabled() -> bool {
    true
}
fn default_rust_wasm_enabled() -> bool {
    true
}
fn default_fuel_limit() -> u64 {
    1_000_000
}
fn default_memory_limit() -> usize {
    64 * 1024 * 1024
}
fn default_cache_size() -> usize {
    100
}
