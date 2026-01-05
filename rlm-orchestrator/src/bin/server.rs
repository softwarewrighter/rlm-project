//! RLM Server binary

use anyhow::Result;
use rlm::api::{create_router, ApiState};
use rlm::orchestrator::RlmOrchestrator;
use rlm::pool::{LlmPool, LoadBalanceStrategy, ProviderRole};
use rlm::provider::OllamaProvider;
use rlm::{ProviderConfig, RlmConfig};
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("Starting RLM Server v{}", env!("CARGO_PKG_VERSION"));

    // TODO: Load config from file
    let config = RlmConfig {
        max_iterations: 20,
        max_sub_calls: 50,
        output_limit: 10000,
        providers: vec![ProviderConfig {
            provider_type: "ollama".to_string(),
            base_url: "http://localhost:11434".to_string(),
            model: "qwen2.5-coder:14b".to_string(),
            api_key: None,
            weight: 1,
            role: "both".to_string(),
        }],
    };

    // Create the pool
    let mut pool = LlmPool::new(LoadBalanceStrategy::RoundRobin);

    for provider_config in &config.providers {
        match provider_config.provider_type.as_str() {
            "ollama" => {
                let provider = OllamaProvider::new(
                    &provider_config.base_url,
                    &provider_config.model,
                );
                pool.add_provider(
                    Arc::new(provider),
                    provider_config.weight,
                    ProviderRole::from(provider_config.role.as_str()),
                );
            }
            _ => {
                tracing::warn!(
                    provider = provider_config.provider_type,
                    "Unknown provider type, skipping"
                );
            }
        }
    }

    let pool = Arc::new(pool);

    // Start health check task
    pool.clone().start_health_check_task(30);

    // Create orchestrator
    let orchestrator = Arc::new(RlmOrchestrator::new(config, pool));

    // Create API state
    let state = Arc::new(ApiState { orchestrator });

    // Create router
    let app = create_router(state);

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    info!("Listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
