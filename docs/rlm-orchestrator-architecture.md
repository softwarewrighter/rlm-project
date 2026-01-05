# RLM Orchestrator Architecture

A Rust-based Recursive Language Model orchestrator designed for distributed GPU infrastructure.

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            RLM ORCHESTRATOR                                   │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                          API Layer (Tower/Axum)                          │ │
│  │  REST: /query, /status, /contexts    gRPC: RlmService                   │ │
│  └────────────────────────────────┬────────────────────────────────────────┘ │
│                                   │                                          │
│  ┌────────────────────────────────▼────────────────────────────────────────┐ │
│  │                         Session Manager                                  │ │
│  │  • Session lifecycle    • History tracking    • Timeout handling        │ │
│  └────────────────────────────────┬────────────────────────────────────────┘ │
│                                   │                                          │
│  ┌──────────────┬─────────────────┼─────────────────┬──────────────────────┐ │
│  │              │                 │                 │                      │ │
│  ▼              ▼                 ▼                 ▼                      │ │
│ ┌────────┐ ┌─────────┐    ┌─────────────┐    ┌──────────┐                 │ │
│ │Context │ │  REPL   │    │   LLM Pool  │    │ Executor │                 │ │
│ │ Store  │ │ Manager │    │  (Routing)  │    │  Queue   │                 │ │
│ └────────┘ └─────────┘    └──────┬──────┘    └──────────┘                 │ │
│                                  │                                         │ │
│                    ┌─────────────┼─────────────┐                          │ │
│                    │             │             │                          │ │
│                    ▼             ▼             ▼                          │ │
│              ┌──────────┐ ┌──────────┐ ┌──────────┐                       │ │
│              │  Ollama  │ │ DeepSeek │ │  Claude  │  ... more providers   │ │
│              │ Provider │ │ Provider │ │ Provider │                       │ │
│              └──────────┘ └──────────┘ └──────────┘                       │ │
└──────────────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
             ┌──────────┐  ┌──────────┐  ┌──────────┐
             │ Server 1 │  │ Server 2 │  │ Server 3 │
             │  M40s    │  │   RTX    │  │  P100s   │
             │ Ollama   │  │ Ollama   │  │ Ollama   │
             └──────────┘  └──────────┘  └──────────┘
```

## Project Structure

```
rlm-orchestrator/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── config.rs              # Configuration management
│   ├── error.rs               # Error types
│   │
│   ├── api/
│   │   ├── mod.rs
│   │   ├── rest.rs            # Axum REST handlers
│   │   ├── grpc.rs            # tonic gRPC service
│   │   └── types.rs           # Request/Response types
│   │
│   ├── core/
│   │   ├── mod.rs
│   │   ├── orchestrator.rs    # Main RLM logic
│   │   ├── session.rs         # Session management
│   │   ├── context.rs         # Context store
│   │   └── history.rs         # Execution history
│   │
│   ├── repl/
│   │   ├── mod.rs
│   │   ├── python.rs          # PyO3 Python REPL
│   │   ├── sandbox.rs         # Execution sandboxing
│   │   └── functions.rs       # Injected functions (llm_query, etc.)
│   │
│   ├── providers/
│   │   ├── mod.rs
│   │   ├── traits.rs          # LlmProvider trait
│   │   ├── ollama.rs          # Ollama provider
│   │   ├── deepseek.rs        # DeepSeek API provider
│   │   ├── claude.rs          # Claude API provider
│   │   ├── openai.rs          # OpenAI-compatible provider
│   │   └── llama_cpp.rs       # Direct llama.cpp integration
│   │
│   ├── routing/
│   │   ├── mod.rs
│   │   ├── pool.rs            # LLM pool management
│   │   ├── balancer.rs        # Load balancing strategies
│   │   └── health.rs          # Health checking
│   │
│   └── telemetry/
│       ├── mod.rs
│       ├── metrics.rs         # Prometheus metrics
│       └── tracing.rs         # Distributed tracing
│
├── proto/
│   └── rlm.proto              # gRPC service definition
│
├── examples/
│   ├── simple_query.rs
│   ├── batch_processing.rs
│   └── emacs_integration.rs
│
└── tests/
    ├── integration/
    └── providers/
```

## Core Types

### Cargo.toml

```toml
[package]
name = "rlm-orchestrator"
version = "0.1.0"
edition = "2021"

[dependencies]
# Async runtime
tokio = { version = "1", features = ["full"] }

# Web framework
axum = "0.7"
tower = "0.4"
tower-http = { version = "0.5", features = ["cors", "trace"] }

# gRPC
tonic = "0.11"
prost = "0.12"

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# HTTP client
reqwest = { version = "0.11", features = ["json", "rustls-tls"] }

# Python integration
pyo3 = { version = "0.20", features = ["auto-initialize"] }

# Configuration
config = "0.14"
dotenvy = "0.15"

# Error handling
thiserror = "1"
anyhow = "1"

# Logging/Tracing
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Metrics
prometheus = "0.13"

# Utilities
uuid = { version = "1", features = ["v4"] }
chrono = { version = "0.4", features = ["serde"] }
dashmap = "5"  # Concurrent HashMap
parking_lot = "0.12"
regex = "1"

# Async utilities
futures = "0.3"
async-trait = "0.1"

[build-dependencies]
tonic-build = "0.11"
```

### Configuration (config.rs)

```rust
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub server: ServerConfig,
    pub providers: ProvidersConfig,
    pub repl: ReplConfig,
    pub limits: LimitsConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub grpc_port: u16,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ProvidersConfig {
    pub ollama: Vec<OllamaServerConfig>,
    pub deepseek: Option<DeepSeekConfig>,
    pub claude: Option<ClaudeConfig>,
    pub openai_compatible: Vec<OpenAICompatibleConfig>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct OllamaServerConfig {
    pub name: String,
    pub host: String,
    pub port: u16,
    pub models: Vec<String>,
    pub priority: u8,           // For load balancing
    pub max_concurrent: usize,
    pub gpu_memory_gb: f32,     // For smart routing
}

#[derive(Debug, Deserialize, Clone)]
pub struct DeepSeekConfig {
    pub api_key_env: String,    // Environment variable name
    pub base_url: String,
    pub models: DeepSeekModels,
}

#[derive(Debug, Deserialize, Clone)]
pub struct DeepSeekModels {
    pub root: String,           // e.g., "deepseek-chat"
    pub sub: String,            // e.g., "deepseek-chat" or cheaper model
}

#[derive(Debug, Deserialize, Clone)]
pub struct ClaudeConfig {
    pub api_key_env: String,
    pub models: ClaudeModels,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ClaudeModels {
    pub root: String,           // e.g., "claude-sonnet-4-20250514"
    pub sub: String,            // e.g., "claude-haiku-4-5-20251001"
}

#[derive(Debug, Deserialize, Clone)]
pub struct OpenAICompatibleConfig {
    pub name: String,
    pub base_url: String,
    pub api_key_env: Option<String>,
    pub models: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ReplConfig {
    pub python_path: Option<PathBuf>,
    pub timeout_seconds: u64,
    pub max_output_chars: usize,
    pub sandbox_enabled: bool,
}

#[derive(Debug, Deserialize, Clone)]
pub struct LimitsConfig {
    pub max_iterations: usize,
    pub max_context_chars: usize,
    pub max_sub_calls_per_iteration: usize,
    pub session_timeout_minutes: u64,
}

impl Config {
    pub fn load() -> anyhow::Result<Self> {
        let config = config::Config::builder()
            .add_source(config::File::with_name("config/default"))
            .add_source(config::File::with_name("config/local").required(false))
            .add_source(config::Environment::with_prefix("RLM"))
            .build()?;
        
        Ok(config.try_deserialize()?)
    }
}
```

### Example config file (config/default.toml)

```toml
[server]
host = "0.0.0.0"
port = 8080
grpc_port = 50051

[providers]
[[providers.ollama]]
name = "m40-server"
host = "192.168.1.10"
port = 11434
models = ["qwen2.5-coder:32b", "llama3.3:70b"]
priority = 1
max_concurrent = 2
gpu_memory_gb = 24.0

[[providers.ollama]]
name = "rtx-server"
host = "192.168.1.11"
port = 11434
models = ["llama3.3:70b", "deepseek-coder:33b"]
priority = 2
max_concurrent = 1
gpu_memory_gb = 16.0

[[providers.ollama]]
name = "p100-server"
host = "192.168.1.12"
port = 11434
models = ["qwen2.5-coder:14b", "llama3.2:3b"]
priority = 3
max_concurrent = 4
gpu_memory_gb = 32.0  # 2x P100

[providers.deepseek]
api_key_env = "DEEPSEEK_API_KEY"
base_url = "https://api.deepseek.com"
models = { root = "deepseek-chat", sub = "deepseek-chat" }

[providers.claude]
api_key_env = "ANTHROPIC_API_KEY"
models = { root = "claude-sonnet-4-20250514", sub = "claude-haiku-4-5-20251001" }

[repl]
timeout_seconds = 300
max_output_chars = 50000
sandbox_enabled = true

[limits]
max_iterations = 30
max_context_chars = 100_000_000  # 100MB
max_sub_calls_per_iteration = 50
session_timeout_minutes = 60
```

### LLM Provider Trait (providers/traits.rs)

```rust
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequest {
    pub prompt: String,
    pub system: Option<String>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    pub content: String,
    pub tokens_used: TokenUsage,
    pub model: String,
    pub provider: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct ProviderHealth {
    pub available: bool,
    pub latency_ms: Option<u64>,
    pub queue_depth: usize,
    pub error_rate: f32,
}

#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Unique identifier for this provider instance
    fn id(&self) -> &str;
    
    /// Human-readable name
    fn name(&self) -> &str;
    
    /// Available models on this provider
    fn models(&self) -> &[String];
    
    /// Approximate context window size in characters
    fn context_limit(&self) -> usize;
    
    /// Query the LLM
    async fn query(&self, request: LlmRequest) -> anyhow::Result<LlmResponse>;
    
    /// Check provider health
    async fn health_check(&self) -> ProviderHealth;
    
    /// Priority for load balancing (lower = preferred)
    fn priority(&self) -> u8;
    
    /// Current load (0.0 - 1.0)
    fn load(&self) -> f32;
}
```

### Ollama Provider (providers/ollama.rs)

```rust
use super::traits::*;
use crate::config::OllamaServerConfig;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct OllamaProvider {
    config: OllamaServerConfig,
    client: Client,
    base_url: String,
    active_requests: AtomicUsize,
    total_requests: AtomicUsize,
    total_errors: AtomicUsize,
}

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    system: Option<String>,
    stream: bool,
    options: Option<OllamaOptions>,
}

#[derive(Serialize)]
struct OllamaOptions {
    num_predict: Option<usize>,
    temperature: Option<f32>,
    stop: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
    done: bool,
    total_duration: Option<u64>,
    prompt_eval_count: Option<usize>,
    eval_count: Option<usize>,
}

impl OllamaProvider {
    pub fn new(config: OllamaServerConfig) -> Self {
        let base_url = format!("http://{}:{}", config.host, config.port);
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .expect("Failed to create HTTP client");
        
        Self {
            config,
            client,
            base_url,
            active_requests: AtomicUsize::new(0),
            total_requests: AtomicUsize::new(0),
            total_errors: AtomicUsize::new(0),
        }
    }
    
    fn select_model(&self, preferred: Option<&str>) -> &str {
        if let Some(model) = preferred {
            if self.config.models.contains(&model.to_string()) {
                return model;
            }
        }
        // Default to first available model
        &self.config.models[0]
    }
}

#[async_trait]
impl LlmProvider for OllamaProvider {
    fn id(&self) -> &str {
        &self.config.name
    }
    
    fn name(&self) -> &str {
        &self.config.name
    }
    
    fn models(&self) -> &[String] {
        &self.config.models
    }
    
    fn context_limit(&self) -> usize {
        // Most Ollama models: ~128K tokens ≈ 500K chars
        500_000
    }
    
    async fn query(&self, request: LlmRequest) -> anyhow::Result<LlmResponse> {
        self.active_requests.fetch_add(1, Ordering::SeqCst);
        self.total_requests.fetch_add(1, Ordering::SeqCst);
        
        let model = self.select_model(None);
        
        let ollama_request = OllamaRequest {
            model: model.to_string(),
            prompt: request.prompt,
            system: request.system,
            stream: false,
            options: Some(OllamaOptions {
                num_predict: request.max_tokens,
                temperature: request.temperature,
                stop: request.stop_sequences,
            }),
        };
        
        let result = self.client
            .post(format!("{}/api/generate", self.base_url))
            .json(&ollama_request)
            .send()
            .await;
        
        self.active_requests.fetch_sub(1, Ordering::SeqCst);
        
        match result {
            Ok(response) => {
                let ollama_response: OllamaResponse = response.json().await?;
                Ok(LlmResponse {
                    content: ollama_response.response,
                    tokens_used: TokenUsage {
                        prompt_tokens: ollama_response.prompt_eval_count.unwrap_or(0),
                        completion_tokens: ollama_response.eval_count.unwrap_or(0),
                        total_tokens: ollama_response.prompt_eval_count.unwrap_or(0)
                            + ollama_response.eval_count.unwrap_or(0),
                    },
                    model: model.to_string(),
                    provider: self.config.name.clone(),
                })
            }
            Err(e) => {
                self.total_errors.fetch_add(1, Ordering::SeqCst);
                Err(e.into())
            }
        }
    }
    
    async fn health_check(&self) -> ProviderHealth {
        let start = Instant::now();
        
        let result = self.client
            .get(format!("{}/api/tags", self.base_url))
            .timeout(Duration::from_secs(5))
            .send()
            .await;
        
        let latency = start.elapsed().as_millis() as u64;
        let total = self.total_requests.load(Ordering::SeqCst);
        let errors = self.total_errors.load(Ordering::SeqCst);
        let error_rate = if total > 0 { errors as f32 / total as f32 } else { 0.0 };
        
        ProviderHealth {
            available: result.is_ok(),
            latency_ms: Some(latency),
            queue_depth: self.active_requests.load(Ordering::SeqCst),
            error_rate,
        }
    }
    
    fn priority(&self) -> u8 {
        self.config.priority
    }
    
    fn load(&self) -> f32 {
        let active = self.active_requests.load(Ordering::SeqCst) as f32;
        let max = self.config.max_concurrent as f32;
        (active / max).min(1.0)
    }
}
```

### DeepSeek Provider (providers/deepseek.rs)

```rust
use super::traits::*;
use crate::config::DeepSeekConfig;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

pub struct DeepSeekProvider {
    config: DeepSeekConfig,
    client: Client,
    api_key: String,
    active_requests: AtomicUsize,
}

#[derive(Serialize)]
struct DeepSeekRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    stop: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct DeepSeekResponse {
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Deserialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

impl DeepSeekProvider {
    pub fn new(config: DeepSeekConfig) -> anyhow::Result<Self> {
        let api_key = env::var(&config.api_key_env)
            .map_err(|_| anyhow::anyhow!("Missing env var: {}", config.api_key_env))?;
        
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()?;
        
        Ok(Self {
            config,
            client,
            api_key,
            active_requests: AtomicUsize::new(0),
        })
    }
}

#[async_trait]
impl LlmProvider for DeepSeekProvider {
    fn id(&self) -> &str {
        "deepseek"
    }
    
    fn name(&self) -> &str {
        "DeepSeek API"
    }
    
    fn models(&self) -> &[String] {
        // Return both root and sub models
        &[]  // Would need to store these properly
    }
    
    fn context_limit(&self) -> usize {
        // DeepSeek supports 64K context
        250_000
    }
    
    async fn query(&self, request: LlmRequest) -> anyhow::Result<LlmResponse> {
        self.active_requests.fetch_add(1, Ordering::SeqCst);
        
        let mut messages = Vec::new();
        if let Some(system) = request.system {
            messages.push(Message {
                role: "system".to_string(),
                content: system,
            });
        }
        messages.push(Message {
            role: "user".to_string(),
            content: request.prompt,
        });
        
        let ds_request = DeepSeekRequest {
            model: self.config.models.root.clone(),
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            stop: request.stop_sequences,
        };
        
        let response = self.client
            .post(format!("{}/chat/completions", self.config.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&ds_request)
            .send()
            .await?;
        
        self.active_requests.fetch_sub(1, Ordering::SeqCst);
        
        let ds_response: DeepSeekResponse = response.json().await?;
        
        Ok(LlmResponse {
            content: ds_response.choices[0].message.content.clone(),
            tokens_used: TokenUsage {
                prompt_tokens: ds_response.usage.prompt_tokens,
                completion_tokens: ds_response.usage.completion_tokens,
                total_tokens: ds_response.usage.total_tokens,
            },
            model: self.config.models.root.clone(),
            provider: "deepseek".to_string(),
        })
    }
    
    async fn health_check(&self) -> ProviderHealth {
        // Simple health check - try to list models
        let result = self.client
            .get(format!("{}/models", self.config.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .timeout(Duration::from_secs(5))
            .send()
            .await;
        
        ProviderHealth {
            available: result.is_ok(),
            latency_ms: None,
            queue_depth: self.active_requests.load(Ordering::SeqCst),
            error_rate: 0.0,
        }
    }
    
    fn priority(&self) -> u8 {
        10  // Cloud APIs are lower priority than local
    }
    
    fn load(&self) -> f32 {
        // API has no real load limit
        0.0
    }
}
```

### LLM Pool with Load Balancing (routing/pool.rs)

```rust
use crate::providers::traits::{LlmProvider, LlmRequest, LlmResponse, ProviderHealth};
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct LlmPool {
    providers: Vec<Arc<dyn LlmProvider>>,
    health_cache: DashMap<String, ProviderHealth>,
    strategy: LoadBalanceStrategy,
}

#[derive(Debug, Clone, Copy)]
pub enum LoadBalanceStrategy {
    RoundRobin,
    LeastLoaded,
    Priority,
    Adaptive,  // Combines load + priority + health
}

impl LlmPool {
    pub fn new(strategy: LoadBalanceStrategy) -> Self {
        Self {
            providers: Vec::new(),
            health_cache: DashMap::new(),
            strategy,
        }
    }
    
    pub fn add_provider(&mut self, provider: Arc<dyn LlmProvider>) {
        self.providers.push(provider);
    }
    
    pub async fn query(&self, request: LlmRequest) -> anyhow::Result<LlmResponse> {
        let provider = self.select_provider().await?;
        provider.query(request).await
    }
    
    pub async fn query_with_model(
        &self,
        request: LlmRequest,
        model: &str,
    ) -> anyhow::Result<LlmResponse> {
        // Find provider that has this model
        for provider in &self.providers {
            if provider.models().iter().any(|m| m == model) {
                let health = self.health_cache.get(provider.id());
                if health.map(|h| h.available).unwrap_or(true) {
                    return provider.query(request).await;
                }
            }
        }
        anyhow::bail!("No available provider with model: {}", model)
    }
    
    async fn select_provider(&self) -> anyhow::Result<Arc<dyn LlmProvider>> {
        match self.strategy {
            LoadBalanceStrategy::Priority => self.select_by_priority().await,
            LoadBalanceStrategy::LeastLoaded => self.select_least_loaded().await,
            LoadBalanceStrategy::Adaptive => self.select_adaptive().await,
            LoadBalanceStrategy::RoundRobin => self.select_round_robin().await,
        }
    }
    
    async fn select_by_priority(&self) -> anyhow::Result<Arc<dyn LlmProvider>> {
        let mut candidates: Vec<_> = self.providers.iter()
            .filter(|p| {
                self.health_cache.get(p.id())
                    .map(|h| h.available)
                    .unwrap_or(true)
            })
            .collect();
        
        candidates.sort_by_key(|p| p.priority());
        
        candidates.first()
            .cloned()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("No available providers"))
    }
    
    async fn select_least_loaded(&self) -> anyhow::Result<Arc<dyn LlmProvider>> {
        self.providers.iter()
            .filter(|p| {
                self.health_cache.get(p.id())
                    .map(|h| h.available)
                    .unwrap_or(true)
            })
            .min_by(|a, b| a.load().partial_cmp(&b.load()).unwrap())
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("No available providers"))
    }
    
    async fn select_adaptive(&self) -> anyhow::Result<Arc<dyn LlmProvider>> {
        // Score = priority * 0.3 + load * 0.5 + error_rate * 0.2
        let mut best: Option<(Arc<dyn LlmProvider>, f32)> = None;
        
        for provider in &self.providers {
            let health = self.health_cache.get(provider.id());
            
            if !health.as_ref().map(|h| h.available).unwrap_or(true) {
                continue;
            }
            
            let error_rate = health.as_ref().map(|h| h.error_rate).unwrap_or(0.0);
            let score = (provider.priority() as f32 * 0.3)
                + (provider.load() * 0.5)
                + (error_rate * 0.2);
            
            match &best {
                None => best = Some((provider.clone(), score)),
                Some((_, best_score)) if score < *best_score => {
                    best = Some((provider.clone(), score))
                }
                _ => {}
            }
        }
        
        best.map(|(p, _)| p)
            .ok_or_else(|| anyhow::anyhow!("No available providers"))
    }
    
    async fn select_round_robin(&self) -> anyhow::Result<Arc<dyn LlmProvider>> {
        // Simple implementation - in production, use AtomicUsize counter
        self.providers.first()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("No providers configured"))
    }
    
    pub async fn refresh_health(&self) {
        for provider in &self.providers {
            let health = provider.health_check().await;
            self.health_cache.insert(provider.id().to_string(), health);
        }
    }
}
```

### Core Orchestrator (core/orchestrator.rs)

```rust
use crate::config::Config;
use crate::providers::traits::{LlmProvider, LlmRequest};
use crate::repl::PythonRepl;
use crate::routing::pool::LlmPool;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlmQuery {
    pub query: String,
    pub context: String,
    pub context_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlmResult {
    pub answer: String,
    pub iterations: usize,
    pub history: Vec<IterationRecord>,
    pub total_tokens: usize,
    pub total_sub_calls: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationRecord {
    pub step: usize,
    pub code: Option<String>,
    pub output: String,
    pub sub_calls: usize,
    pub tokens_used: usize,
}

pub struct RlmOrchestrator {
    config: Config,
    llm_pool: Arc<LlmPool>,
    code_regex: Regex,
    final_regex: Regex,
    final_var_regex: Regex,
}

impl RlmOrchestrator {
    pub fn new(config: Config, llm_pool: Arc<LlmPool>) -> Self {
        Self {
            config,
            llm_pool,
            code_regex: Regex::new(r"```repl\n([\s\S]*?)```").unwrap(),
            final_regex: Regex::new(r"FINAL\(([\s\S]*?)\)").unwrap(),
            final_var_regex: Regex::new(r"FINAL_VAR\((\w+)\)").unwrap(),
        }
    }
    
    pub async fn process(&self, query: RlmQuery) -> anyhow::Result<RlmResult> {
        let mut repl = PythonRepl::new(self.llm_pool.clone())?;
        let mut history = Vec::new();
        let mut total_tokens = 0;
        let mut total_sub_calls = 0;
        
        // Load context into REPL
        repl.set_variable("context", &query.context)?;
        
        // Build system prompt
        let system_prompt = self.build_system_prompt(&query);
        
        for iteration in 0..self.config.limits.max_iterations {
            // Build conversation history for this iteration
            let prompt = self.build_iteration_prompt(&history, &query.query);
            
            // Query root LLM
            let request = LlmRequest {
                prompt,
                system: Some(system_prompt.clone()),
                max_tokens: Some(4096),
                temperature: Some(0.7),
                stop_sequences: None,
            };
            
            let response = self.llm_pool.query(request).await?;
            total_tokens += response.tokens_used.total_tokens;
            
            // Check for final answer
            if let Some(answer) = self.extract_final(&response.content, &repl) {
                return Ok(RlmResult {
                    answer,
                    iterations: iteration + 1,
                    history,
                    total_tokens,
                    total_sub_calls,
                });
            }
            
            // Extract and execute code
            let (output, sub_calls) = if let Some(code) = self.extract_code(&response.content) {
                let (output, calls) = repl.execute_with_tracking(&code).await?;
                total_sub_calls += calls;
                (output, calls)
            } else {
                ("No code block found in response".to_string(), 0)
            };
            
            history.push(IterationRecord {
                step: iteration + 1,
                code: self.extract_code(&response.content),
                output: self.truncate_output(&output),
                sub_calls,
                tokens_used: response.tokens_used.total_tokens,
            });
        }
        
        anyhow::bail!("Max iterations ({}) reached without final answer", 
                      self.config.limits.max_iterations)
    }
    
    fn build_system_prompt(&self, query: &RlmQuery) -> String {
        let context_type = query.context_type.as_deref().unwrap_or("text");
        let context_len = query.context.len();
        
        format!(r#"You are an RLM agent tasked with answering queries over large contexts.

Your context is a {context_type} with {context_len} total characters.

The REPL environment provides:
1. `context` - the full input (may be huge, use programmatic access)
2. `llm_query(prompt)` - recursive sub-LM call for semantic analysis
3. Standard Python libraries

Strategy:
1. Probe context structure (print samples, lengths)
2. Filter/chunk based on content type
3. Use llm_query() for semantic analysis of chunks
4. Aggregate results in variables
5. Return FINAL(answer) or FINAL_VAR(variable_name) when ready

Write Python code in ```repl blocks. You will see truncated outputs.
For long outputs, store in variables and query sub-LMs to analyze them."#)
    }
    
    fn build_iteration_prompt(&self, history: &[IterationRecord], query: &str) -> String {
        let mut prompt = format!("Query: {}\n\n", query);
        
        if !history.is_empty() {
            prompt.push_str("Previous iterations:\n");
            for record in history.iter().rev().take(5) {
                prompt.push_str(&format!(
                    "[Step {}]\nCode:\n{}\nOutput:\n{}\n\n",
                    record.step,
                    record.code.as_deref().unwrap_or("(none)"),
                    record.output
                ));
            }
        }
        
        prompt.push_str("What's your next step? Write code or provide FINAL(answer).");
        prompt
    }
    
    fn extract_code(&self, response: &str) -> Option<String> {
        self.code_regex.captures(response)
            .map(|caps| caps.get(1).unwrap().as_str().to_string())
    }
    
    fn extract_final(&self, response: &str, repl: &PythonRepl) -> Option<String> {
        // Check for FINAL(answer)
        if let Some(caps) = self.final_regex.captures(response) {
            return Some(caps.get(1).unwrap().as_str().to_string());
        }
        
        // Check for FINAL_VAR(variable_name)
        if let Some(caps) = self.final_var_regex.captures(response) {
            let var_name = caps.get(1).unwrap().as_str();
            return repl.get_variable(var_name).ok();
        }
        
        None
    }
    
    fn truncate_output(&self, output: &str) -> String {
        let max = self.config.repl.max_output_chars;
        if output.len() <= max {
            output.to_string()
        } else {
            let half = max / 2;
            format!(
                "{}...\n[truncated {} chars]\n...{}",
                &output[..half],
                output.len() - max,
                &output[output.len() - half..]
            )
        }
    }
}
```

### Python REPL with Sub-LM Calls (repl/python.rs)

```rust
use crate::routing::pool::LlmPool;
use crate::providers::traits::LlmRequest;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::Mutex;

pub struct PythonRepl {
    llm_pool: Arc<LlmPool>,
    globals: Py<PyDict>,
    sub_call_count: Arc<AtomicUsize>,
}

impl PythonRepl {
    pub fn new(llm_pool: Arc<LlmPool>) -> anyhow::Result<Self> {
        Python::with_gil(|py| {
            let globals = PyDict::new(py);
            
            // Import standard libraries
            let builtins = PyModule::import(py, "builtins")?;
            globals.set_item("__builtins__", builtins)?;
            
            // Import useful modules
            let re = PyModule::import(py, "re")?;
            let json = PyModule::import(py, "json")?;
            let collections = PyModule::import(py, "collections")?;
            
            globals.set_item("re", re)?;
            globals.set_item("json", json)?;
            globals.set_item("collections", collections)?;
            
            Ok(Self {
                llm_pool,
                globals: globals.into(),
                sub_call_count: Arc::new(AtomicUsize::new(0)),
            })
        })
    }
    
    pub fn set_variable(&mut self, name: &str, value: &str) -> anyhow::Result<()> {
        Python::with_gil(|py| {
            let globals = self.globals.as_ref(py);
            globals.set_item(name, value)?;
            Ok(())
        })
    }
    
    pub fn get_variable(&self, name: &str) -> anyhow::Result<String> {
        Python::with_gil(|py| {
            let globals = self.globals.as_ref(py);
            let value = globals.get_item(name)?
                .ok_or_else(|| anyhow::anyhow!("Variable not found: {}", name))?;
            Ok(value.to_string())
        })
    }
    
    pub async fn execute_with_tracking(&mut self, code: &str) -> anyhow::Result<(String, usize)> {
        self.sub_call_count.store(0, Ordering::SeqCst);
        
        // Inject llm_query function
        let llm_pool = self.llm_pool.clone();
        let sub_call_count = self.sub_call_count.clone();
        
        let output = Python::with_gil(|py| -> anyhow::Result<String> {
            let globals = self.globals.as_ref(py);
            
            // Create llm_query function
            let llm_query = PyCFunction::new_closure(
                py,
                Some("llm_query"),
                Some("Query a sub-LLM with the given prompt"),
                move |args: &PyTuple, _kwargs: Option<&PyDict>| -> PyResult<String> {
                    let prompt: String = args.get_item(0)?.extract()?;
                    
                    // We need to block here since we're in sync Python context
                    // In production, you'd use pyo3-asyncio or similar
                    let rt = tokio::runtime::Handle::current();
                    let pool = llm_pool.clone();
                    let count = sub_call_count.clone();
                    
                    let result = rt.block_on(async move {
                        count.fetch_add(1, Ordering::SeqCst);
                        let request = LlmRequest {
                            prompt,
                            system: None,
                            max_tokens: Some(4096),
                            temperature: Some(0.3),
                            stop_sequences: None,
                        };
                        pool.query(request).await
                    });
                    
                    result.map(|r| r.content)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                            format!("LLM query failed: {}", e)
                        ))
                }
            )?;
            
            globals.set_item("llm_query", llm_query)?;
            
            // Capture stdout
            let sys = PyModule::import(py, "sys")?;
            let io = PyModule::import(py, "io")?;
            let string_io = io.getattr("StringIO")?.call0()?;
            let old_stdout = sys.getattr("stdout")?;
            sys.setattr("stdout", string_io)?;
            
            // Execute code
            let result = py.run(code, Some(globals), None);
            
            // Restore stdout and get output
            sys.setattr("stdout", old_stdout)?;
            let output: String = string_io.call_method0("getvalue")?.extract()?;
            
            match result {
                Ok(_) => Ok(output),
                Err(e) => Ok(format!("{}\nError: {}", output, e)),
            }
        })?;
        
        let sub_calls = self.sub_call_count.load(Ordering::SeqCst);
        Ok((output, sub_calls))
    }
}
```

### REST API (api/rest.rs)

```rust
use crate::core::orchestrator::{RlmOrchestrator, RlmQuery, RlmResult};
use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub orchestrator: Arc<RlmOrchestrator>,
}

#[derive(Debug, Deserialize)]
pub struct QueryRequest {
    pub query: String,
    pub context: String,
    #[serde(default)]
    pub context_type: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct QueryResponse {
    pub success: bool,
    pub result: Option<RlmResult>,
    pub error: Option<String>,
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/query", post(process_query))
        .route("/providers", get(list_providers))
        .with_state(state)
}

async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

async fn process_query(
    State(state): State<AppState>,
    Json(request): Json<QueryRequest>,
) -> impl IntoResponse {
    let query = RlmQuery {
        query: request.query,
        context: request.context,
        context_type: request.context_type,
    };
    
    match state.orchestrator.process(query).await {
        Ok(result) => (
            StatusCode::OK,
            Json(QueryResponse {
                success: true,
                result: Some(result),
                error: None,
            }),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(QueryResponse {
                success: false,
                result: None,
                error: Some(e.to_string()),
            }),
        ),
    }
}

async fn list_providers(State(state): State<AppState>) -> impl IntoResponse {
    // Would return list of configured providers and their health
    Json(serde_json::json!({
        "providers": []
    }))
}
```

### Main Entry Point (main.rs)

```rust
mod api;
mod config;
mod core;
mod error;
mod providers;
mod repl;
mod routing;
mod telemetry;

use crate::api::rest::{create_router, AppState};
use crate::config::Config;
use crate::core::orchestrator::RlmOrchestrator;
use crate::providers::{
    ollama::OllamaProvider,
    deepseek::DeepSeekProvider,
};
use crate::routing::pool::{LlmPool, LoadBalanceStrategy};
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();
    
    // Load configuration
    let config = Config::load()?;
    tracing::info!("Loaded configuration");
    
    // Initialize LLM pool
    let mut pool = LlmPool::new(LoadBalanceStrategy::Adaptive);
    
    // Add Ollama providers
    for ollama_config in &config.providers.ollama {
        let provider = OllamaProvider::new(ollama_config.clone());
        pool.add_provider(Arc::new(provider));
        tracing::info!("Added Ollama provider: {}", ollama_config.name);
    }
    
    // Add DeepSeek provider if configured
    if let Some(ds_config) = &config.providers.deepseek {
        match DeepSeekProvider::new(ds_config.clone()) {
            Ok(provider) => {
                pool.add_provider(Arc::new(provider));
                tracing::info!("Added DeepSeek provider");
            }
            Err(e) => {
                tracing::warn!("Failed to initialize DeepSeek provider: {}", e);
            }
        }
    }
    
    let pool = Arc::new(pool);
    
    // Create orchestrator
    let orchestrator = Arc::new(RlmOrchestrator::new(config.clone(), pool.clone()));
    
    // Start health check background task
    let health_pool = pool.clone();
    tokio::spawn(async move {
        loop {
            health_pool.refresh_health().await;
            tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;
        }
    });
    
    // Create API router
    let app = create_router(AppState { orchestrator });
    
    // Start server
    let addr = format!("{}:{}", config.server.host, config.server.port);
    let listener = TcpListener::bind(&addr).await?;
    tracing::info!("RLM Orchestrator listening on {}", addr);
    
    axum::serve(listener, app).await?;
    
    Ok(())
}
```

## CLI Client

For convenience, here's a simple CLI client:

```rust
// examples/cli.rs
use clap::Parser;
use reqwest::Client;
use serde_json::json;
use std::fs;

#[derive(Parser)]
#[command(name = "rlm")]
#[command(about = "RLM Orchestrator CLI")]
struct Cli {
    /// Query to process
    #[arg(short, long)]
    query: String,
    
    /// Context file path
    #[arg(short, long)]
    context_file: String,
    
    /// Server URL
    #[arg(short, long, default_value = "http://localhost:8080")]
    server: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    
    let context = fs::read_to_string(&cli.context_file)?;
    
    let client = Client::new();
    let response = client
        .post(format!("{}/query", cli.server))
        .json(&json!({
            "query": cli.query,
            "context": context,
        }))
        .send()
        .await?;
    
    let result: serde_json::Value = response.json().await?;
    
    if result["success"].as_bool().unwrap_or(false) {
        println!("Answer: {}", result["result"]["answer"]);
        println!("Iterations: {}", result["result"]["iterations"]);
        println!("Total tokens: {}", result["result"]["total_tokens"]);
        println!("Sub-calls: {}", result["result"]["total_sub_calls"]);
    } else {
        eprintln!("Error: {}", result["error"]);
    }
    
    Ok(())
}
```

## Emacs Integration (Elisp)

```elisp
;;; rlm.el --- RLM Orchestrator integration for Emacs -*- lexical-binding: t -*-

(require 'json)
(require 'url)

(defgroup rlm nil
  "RLM Orchestrator integration."
  :group 'tools)

(defcustom rlm-server-url "http://localhost:8080"
  "URL of the RLM orchestrator server."
  :type 'string
  :group 'rlm)

(defun rlm--request (endpoint method &optional data)
  "Make a request to the RLM server."
  (let* ((url-request-method method)
         (url-request-extra-headers
          '(("Content-Type" . "application/json")))
         (url-request-data
          (when data (encode-coding-string (json-encode data) 'utf-8)))
         (buffer (url-retrieve-synchronously
                  (concat rlm-server-url endpoint) t t 120)))
    (when buffer
      (with-current-buffer buffer
        (goto-char url-http-end-of-headers)
        (json-read)))))

(defun rlm-query (query context)
  "Query the RLM orchestrator with QUERY over CONTEXT."
  (interactive
   (list (read-string "Query: ")
         (if (use-region-p)
             (buffer-substring-no-properties (region-beginning) (region-end))
           (buffer-string))))
  (let ((result (rlm--request "/query" "POST"
                              `((query . ,query)
                                (context . ,context)))))
    (if (eq (alist-get 'success result) t)
        (let ((answer (alist-get 'answer (alist-get 'result result)))
              (iterations (alist-get 'iterations (alist-get 'result result)))
              (tokens (alist-get 'total_tokens (alist-get 'result result))))
          (message "Answer (%d iterations, %d tokens): %s"
                   iterations tokens answer)
          answer)
      (error "RLM query failed: %s" (alist-get 'error result)))))

(defun rlm-query-buffer ()
  "Query the RLM with the entire buffer as context."
  (interactive)
  (let ((query (read-string "Query about this buffer: ")))
    (rlm-query query (buffer-string))))

(defun rlm-query-region (start end)
  "Query the RLM with the selected region as context."
  (interactive "r")
  (let ((query (read-string "Query about selection: ")))
    (rlm-query query (buffer-substring-no-properties start end))))

(provide 'rlm)
;;; rlm.el ends here
```

## Usage Examples

### Basic Query

```bash
# Using CLI
rlm --query "What is the main thesis of this document?" \
    --context-file large_document.txt

# Using curl
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find all mentions of machine learning",
    "context": "'"$(cat huge_corpus.txt)"'"
  }'
```

### From Emacs

```elisp
;; Query about current buffer
M-x rlm-query-buffer RET
"What are the main functions in this code?" RET

;; Query about selection
;; Select region first, then:
M-x rlm-query-region RET
"Summarize this section" RET
```

## Performance Considerations

1. **Connection Pooling**: The reqwest client reuses connections
2. **Async Execution**: All LLM calls are async, enabling parallelism
3. **Health Caching**: Provider health is cached to avoid constant checks
4. **Output Truncation**: Large REPL outputs are truncated to save tokens
5. **Adaptive Routing**: Load balancing considers multiple factors

## Future Enhancements

- [ ] Async sub-LM calls from Python (pyo3-asyncio)
- [ ] Deeper recursion (sub-RLM instead of just sub-LM)
- [ ] Persistent session storage (Redis/SQLite)
- [ ] Web UI for monitoring
- [ ] gRPC streaming for real-time progress
- [ ] Model-specific optimizations (different prompts per model)
