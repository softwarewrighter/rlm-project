//! RLM Orchestrator - High-performance Recursive Language Model implementation
//!
//! This crate provides:
//! - Multiple LLM provider backends (Ollama, DeepSeek, OpenAI-compatible)
//! - Connection pooling with load balancing across distributed GPU servers
//! - Structured command execution (pure Rust, no Python dependency)
//! - REST API for external integration
//!
//! ## 4-Level Capability Architecture
//!
//! The system supports 4 capability levels with increasing power and risk:
//!
//! - **Level 1 (DSL)**: Safe text operations - slice, lines, regex, find, count
//! - **Level 2 (WASM)**: Sandboxed computation - rust_wasm_intent, rust_wasm_mapreduce
//! - **Level 3 (CLI)**: Full Rust capability - rust_cli_intent (process isolation only)
//! - **Level 4 (LLM Delegation)**: Chunk-based LLM analysis - llm_query, llm_delegate_chunks
//!
//! By default, only Levels 1 and 2 are enabled (safe defaults).

pub mod api;
pub mod codegen;
pub mod commands;
pub mod levels;
pub mod orchestrator;
pub mod pool;
pub mod provider;
pub mod wasm;

pub use levels::{Level, LevelRegistry, RiskLevel};
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

    /// Context size threshold for phased processing (chars). Contexts larger than this
    /// trigger a multi-phase approach: assess, strategize, reduce, then analyze.
    /// Default: 500000 chars (~500KB)
    #[serde(default = "default_phased_threshold")]
    pub phased_threshold: usize,

    /// Target size for data after reduction (chars). Iterative reduction continues
    /// until data is below this threshold or max_reduction_passes is reached.
    /// Default: 100000 chars (~100KB, ~25K tokens)
    #[serde(default = "default_target_analysis_size")]
    pub target_analysis_size: usize,

    /// Maximum reduction passes in phased processing. Each pass attempts to
    /// further reduce the data toward target_analysis_size.
    /// Default: 3
    #[serde(default = "default_max_reduction_passes")]
    pub max_reduction_passes: usize,

    /// Level priority order for command selection
    /// Default: ["dsl", "wasm"] (safe levels only)
    #[serde(default = "default_level_priority")]
    pub level_priority: Vec<String>,

    /// Provider configuration
    pub providers: Vec<ProviderConfig>,

    /// Level 1: DSL configuration (text operations)
    #[serde(default)]
    pub dsl: DslConfig,

    /// Level 2: WASM execution configuration (sandboxed computation)
    #[serde(default)]
    pub wasm: WasmConfig,

    /// Level 3: CLI execution configuration (native binaries, process isolation)
    #[serde(default)]
    pub cli: CliConfig,

    /// Level 4: LLM Delegation configuration (chunk-based LLM analysis)
    #[serde(default)]
    pub llm_delegation: LlmDelegationConfig,
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
    false // Off by default for demos; enable in config for production
}
fn default_bypass_threshold() -> usize {
    4000
}
fn default_phased_threshold() -> usize {
    500000 // 500KB - contexts larger than this use phased processing
}
fn default_target_analysis_size() -> usize {
    100000 // 100KB - target size after reduction passes
}
fn default_max_reduction_passes() -> usize {
    3 // Maximum number of reduction iterations
}
fn default_level_priority() -> Vec<String> {
    vec!["dsl".to_string(), "wasm".to_string()]
}

// ============================================================================
// Level 1: DSL Configuration
// ============================================================================

/// Configuration for Level 1: DSL (text operations)
///
/// DSL commands are safe, read-only operations for text extraction and filtering.
/// Enabled by default.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct DslConfig {
    /// Enable DSL commands (slice, lines, regex, find, count, etc.)
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Maximum regex matches to return (prevents memory exhaustion)
    #[serde(default = "default_max_regex_matches")]
    pub max_regex_matches: usize,

    /// Maximum slice size in bytes
    #[serde(default = "default_max_slice_size")]
    pub max_slice_size: usize,

    /// Maximum number of variables
    #[serde(default = "default_max_variables")]
    pub max_variables: usize,

    /// Maximum variable size in bytes
    #[serde(default = "default_max_variable_size")]
    pub max_variable_size: usize,
}

impl Default for DslConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_regex_matches: 10_000,
            max_slice_size: 1024 * 1024, // 1MB
            max_variables: 100,
            max_variable_size: 1024 * 1024, // 1MB
        }
    }
}

fn default_true() -> bool {
    true
}
fn default_max_regex_matches() -> usize {
    10_000
}
fn default_max_slice_size() -> usize {
    1024 * 1024
}
fn default_max_variables() -> usize {
    100
}
fn default_max_variable_size() -> usize {
    1024 * 1024
}

// ============================================================================
// Level 3: CLI Configuration
// ============================================================================

/// Configuration for Level 3: CLI (native binary execution)
///
/// CLI commands compile and execute native Rust binaries with full stdlib access.
/// Process isolation only - no WASM sandbox. Disabled by default.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct CliConfig {
    /// Enable CLI commands (rust_cli_intent)
    #[serde(default)]
    pub enabled: bool,

    /// Sandbox mode: "none" (process only), "docker", "seccomp" (future)
    #[serde(default = "default_sandbox_mode")]
    pub sandbox_mode: String,

    /// Execution timeout in seconds
    #[serde(default = "default_cli_timeout")]
    pub timeout_secs: u64,

    /// Maximum output size in bytes
    #[serde(default = "default_cli_max_output")]
    pub max_output_size: usize,

    /// Directory for binary cache
    pub cache_dir: Option<String>,

    /// Maximum cache size in bytes
    #[serde(default = "default_cli_cache_size")]
    pub max_cache_size: usize,

    /// Code validation settings
    #[serde(default)]
    pub validation: CliValidationConfig,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            enabled: false, // OFF by default
            sandbox_mode: "none".to_string(),
            timeout_secs: 30,
            max_output_size: 10 * 1024 * 1024, // 10MB
            cache_dir: None,
            max_cache_size: 100 * 1024 * 1024, // 100MB
            validation: CliValidationConfig::default(),
        }
    }
}

/// Code validation configuration for CLI execution
#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct CliValidationConfig {
    /// Allow filesystem read operations
    #[serde(default)]
    pub allow_filesystem_read: bool,

    /// Allow filesystem write operations
    #[serde(default)]
    pub allow_filesystem_write: bool,

    /// Allow network operations
    #[serde(default)]
    pub allow_network: bool,

    /// Allow process spawning
    #[serde(default)]
    pub allow_process_spawn: bool,

    /// Allow unsafe code
    #[serde(default)]
    pub allow_unsafe: bool,
}

fn default_sandbox_mode() -> String {
    "none".to_string()
}
fn default_cli_timeout() -> u64 {
    30
}
fn default_cli_max_output() -> usize {
    10 * 1024 * 1024
}
fn default_cli_cache_size() -> usize {
    100 * 1024 * 1024
}

// ============================================================================
// Level 4: LLM Delegation Configuration
// ============================================================================

/// Configuration for Level 4: LLM Delegation (chunk-based LLM analysis)
///
/// Delegates chunks of text to specialized LLMs for fuzzy/semantic analysis.
/// Non-deterministic. Disabled by default.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlmDelegationConfig {
    /// Enable LLM delegation commands (llm_query, llm_delegate)
    #[serde(default)]
    pub enabled: bool,

    /// Chunk size in bytes for splitting large content
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,

    /// Overlap between chunks in bytes
    #[serde(default = "default_chunk_overlap")]
    pub overlap: usize,

    /// Maximum chunks per query
    #[serde(default = "default_max_chunks")]
    pub max_chunks: usize,

    /// Privacy mode: "local" (use local LLM), "cloud", "hybrid"
    #[serde(default = "default_privacy_mode")]
    pub privacy_mode: String,

    /// Maximum concurrent delegation calls
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,

    /// Rate limit per minute
    #[serde(default = "default_rate_limit")]
    pub rate_limit_per_minute: usize,

    /// Maximum recursion depth for nested llm_delegate calls
    /// Default: 3 (root -> nested1 -> nested2 -> nested3, but nested3 cannot delegate further)
    #[serde(default = "default_max_recursion_depth")]
    pub max_recursion_depth: usize,

    /// Default capability levels for nested RLM instances
    /// Note: llm_delegate is automatically excluded from nested instances to prevent infinite recursion
    #[serde(default = "default_nested_levels")]
    pub nested_levels: Vec<String>,

    /// Coordinator mode: Base LLM only uses llm_delegate/llm_query, sub-LLMs do actual data work
    /// When enabled, the base LLM acts as a pure task decomposer/coordinator, while nested
    /// RLM instances use the nested_levels (dsl, cli, etc.) to process data.
    /// Default: false (base LLM has access to all configured levels)
    #[serde(default)]
    pub coordinator_mode: bool,

    /// Dedicated provider for delegation (optional, falls back to root provider)
    pub provider: Option<DelegationProviderConfig>,
}

impl Default for LlmDelegationConfig {
    fn default() -> Self {
        Self {
            enabled: false, // OFF by default
            chunk_size: 4096,
            overlap: 256,
            max_chunks: 100,
            privacy_mode: "local".to_string(),
            max_concurrent: 5,
            rate_limit_per_minute: 60,
            max_recursion_depth: 3,
            nested_levels: vec!["dsl".to_string(), "wasm".to_string()],
            coordinator_mode: false,
            provider: None,
        }
    }
}

/// Provider configuration for LLM delegation
#[derive(Debug, Clone, serde::Deserialize)]
pub struct DelegationProviderConfig {
    /// Provider type: "litellm", "ollama", "openai"
    #[serde(rename = "type")]
    pub provider_type: String,

    /// Provider URL
    pub url: String,

    /// Model name (use fast/cheap models for delegation)
    pub model: String,

    /// Environment variable containing API key
    pub api_key_env: Option<String>,
}

fn default_chunk_size() -> usize {
    4096
}
fn default_chunk_overlap() -> usize {
    256
}
fn default_max_chunks() -> usize {
    100
}
fn default_privacy_mode() -> String {
    "local".to_string()
}
fn default_max_concurrent() -> usize {
    5
}
fn default_rate_limit() -> usize {
    60
}
fn default_max_recursion_depth() -> usize {
    3
}
fn default_nested_levels() -> Vec<String> {
    vec!["dsl".to_string(), "wasm".to_string()]
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

    /// Code generation provider type: "ollama" or "litellm"
    #[serde(default = "default_codegen_provider")]
    pub codegen_provider: String,

    /// Code generation LLM URL
    /// For Ollama: "http://localhost:11434"
    /// For LiteLLM: "http://localhost:4000"
    pub codegen_url: Option<String>,

    /// Code generation LLM model
    /// For Ollama: "qwen2.5-coder:14b"
    /// For LiteLLM: "deepseek/deepseek-coder"
    #[serde(default = "default_codegen_model")]
    pub codegen_model: String,
}

fn default_codegen_provider() -> String {
    "litellm".to_string()
}

fn default_codegen_model() -> String {
    "deepseek/deepseek-coder".to_string()
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
            codegen_provider: default_codegen_provider(),
            codegen_url: None,
            codegen_model: default_codegen_model(),
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
