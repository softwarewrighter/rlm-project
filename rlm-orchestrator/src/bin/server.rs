//! RLM Server binary

use anyhow::{Context, Result};
use rlm::api::{create_router, ApiState};
use rlm::orchestrator::RlmOrchestrator;
use rlm::pool::{LlmPool, LoadBalanceStrategy, ProviderRole};
use rlm::provider::{DeepSeekProvider, LiteLLMProvider, OllamaProvider};
use rlm::RlmConfig;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::DEBUG)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("Starting RLM Server v{}", env!("CARGO_PKG_VERSION"));

    // Load config from file
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "config.toml".to_string());

    let config_contents = std::fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read config file: {}", config_path))?;

    let config: RlmConfig = toml::from_str(&config_contents)
        .with_context(|| format!("Failed to parse config file: {}", config_path))?;

    info!(
        config_path = config_path,
        providers = config.providers.len(),
        "Loaded configuration"
    );

    // Get DeepSeek API key from environment
    let deepseek_key = std::env::var("DEEPSEEK_API_KEY").ok();

    // Create the pool
    let mut pool = LlmPool::new(LoadBalanceStrategy::RoundRobin);
    let mut provider_count = 0;
    let mut root_provider_name = String::from("unknown");

    for provider_config in &config.providers {
        let role = ProviderRole::from(provider_config.role.as_str());

        match provider_config.provider_type.as_str() {
            "ollama" => {
                let provider =
                    OllamaProvider::new(&provider_config.base_url, &provider_config.model);
                info!(
                    provider = "ollama",
                    model = provider_config.model,
                    base_url = provider_config.base_url,
                    role = ?role,
                    "Added Ollama provider"
                );
                // Track root provider name
                if role == ProviderRole::Root || role == ProviderRole::Both {
                    root_provider_name = format!("ollama:{}", provider_config.model);
                }
                pool.add_provider(Arc::new(provider), provider_config.weight, role);
                provider_count += 1;
            }
            "deepseek" => {
                let api_key = provider_config
                    .api_key
                    .clone()
                    .or_else(|| deepseek_key.clone());

                if let Some(key) = api_key {
                    let provider = DeepSeekProvider::with_model(&key, &provider_config.model);
                    info!(
                        provider = "deepseek",
                        model = provider_config.model,
                        role = ?role,
                        "Added DeepSeek provider"
                    );
                    // Track root provider name
                    if role == ProviderRole::Root || role == ProviderRole::Both {
                        root_provider_name = format!("deepseek:{}", provider_config.model);
                    }
                    pool.add_provider(Arc::new(provider), provider_config.weight, role);
                    provider_count += 1;
                } else {
                    warn!(
                        "DeepSeek provider configured but no API key found (set DEEPSEEK_API_KEY)"
                    );
                }
            }
            "litellm" => {
                let api_key = provider_config
                    .api_key
                    .clone()
                    .or_else(|| std::env::var("LITELLM_API_KEY").ok())
                    .or_else(|| std::env::var("LITELLM_MASTER_KEY").ok());

                if let Some(key) = api_key {
                    let provider = LiteLLMProvider::with_base_url(
                        &provider_config.base_url,
                        &key,
                        &provider_config.model,
                    );
                    info!(
                        provider = "litellm",
                        model = provider_config.model,
                        base_url = provider_config.base_url,
                        role = ?role,
                        "Added LiteLLM provider"
                    );
                    // Track root provider name
                    if role == ProviderRole::Root || role == ProviderRole::Both {
                        root_provider_name = format!("litellm:{}", provider_config.model);
                    }
                    pool.add_provider(Arc::new(provider), provider_config.weight, role);
                    provider_count += 1;
                } else {
                    warn!("LiteLLM provider configured but no API key found (set LITELLM_API_KEY or LITELLM_MASTER_KEY)");
                }
            }
            _ => {
                warn!(
                    provider = provider_config.provider_type,
                    "Unknown provider type, skipping"
                );
            }
        }
    }

    if provider_count == 0 {
        anyhow::bail!("No providers configured! Check your config.toml and environment variables.");
    }

    let pool = Arc::new(pool);

    // Start health check task
    pool.clone().start_health_check_task(30);

    // Create orchestrator
    let wasm_enabled = config.wasm.enabled;
    let rust_wasm_enabled = config.wasm.rust_wasm_enabled;
    let orchestrator = Arc::new(RlmOrchestrator::new(config, pool));

    // Create API state
    let state = Arc::new(ApiState {
        orchestrator,
        wasm_enabled,
        rust_wasm_enabled,
        root_provider_name,
    });

    // Create router
    let app = create_router(state);

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    info!("Listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
