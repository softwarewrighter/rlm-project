//! LLM connection pool with load balancing

use crate::provider::{LlmProvider, LlmRequest, LlmResponse, ProviderError};
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::interval;
use tracing::{debug, warn};

/// Statistics for a single provider
#[derive(Debug, Default)]
pub struct ProviderStats {
    /// Total requests made
    pub requests: AtomicU64,
    /// Successful requests
    pub successes: AtomicU64,
    /// Failed requests
    pub failures: AtomicU64,
    /// Total latency in ms
    pub total_latency_ms: AtomicU64,
    /// Last health check result
    pub healthy: std::sync::atomic::AtomicBool,
    /// Last health check latency
    pub health_latency_ms: AtomicU64,
}

/// A pooled provider with metadata
struct PooledProvider {
    provider: Arc<dyn LlmProvider>,
    weight: u32,
    role: ProviderRole,
    stats: Arc<ProviderStats>,
}

/// Role of a provider in the RLM system
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProviderRole {
    /// Only used for root LLM calls
    Root,
    /// Only used for sub-LM calls
    Sub,
    /// Can be used for both
    Both,
}

impl From<&str> for ProviderRole {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "root" => ProviderRole::Root,
            "sub" => ProviderRole::Sub,
            _ => ProviderRole::Both,
        }
    }
}

/// Load balancing strategy
#[derive(Debug, Clone, Copy)]
pub enum LoadBalanceStrategy {
    /// Round-robin across healthy providers
    RoundRobin,
    /// Weighted random selection
    WeightedRandom,
    /// Select provider with lowest latency
    LeastLatency,
}

/// LLM connection pool with load balancing
pub struct LlmPool {
    providers: Vec<PooledProvider>,
    strategy: LoadBalanceStrategy,
    round_robin_idx: AtomicUsize,
    stats_map: DashMap<String, Arc<ProviderStats>>,
}

impl LlmPool {
    /// Create a new empty pool
    pub fn new(strategy: LoadBalanceStrategy) -> Self {
        Self {
            providers: Vec::new(),
            strategy,
            round_robin_idx: AtomicUsize::new(0),
            stats_map: DashMap::new(),
        }
    }

    /// Add a provider to the pool
    pub fn add_provider(
        &mut self,
        provider: Arc<dyn LlmProvider>,
        weight: u32,
        role: ProviderRole,
    ) {
        let stats = Arc::new(ProviderStats::default());
        stats.healthy.store(true, Ordering::Relaxed);

        self.stats_map
            .insert(provider.name().to_string(), Arc::clone(&stats));

        self.providers.push(PooledProvider {
            provider,
            weight,
            role,
            stats,
        });
    }

    /// Get the number of providers
    pub fn len(&self) -> usize {
        self.providers.len()
    }

    /// Check if the pool is empty
    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }

    /// Get a provider for root LLM calls
    pub fn get_root_provider(&self) -> Option<Arc<dyn LlmProvider>> {
        self.get_provider_for_role(|role| role == ProviderRole::Root || role == ProviderRole::Both)
    }

    /// Get a provider for sub-LM calls
    pub fn get_sub_provider(&self) -> Option<Arc<dyn LlmProvider>> {
        self.get_provider_for_role(|role| role == ProviderRole::Sub || role == ProviderRole::Both)
    }

    fn get_provider_for_role<F>(&self, role_filter: F) -> Option<Arc<dyn LlmProvider>>
    where
        F: Fn(ProviderRole) -> bool,
    {
        let eligible: Vec<_> = self
            .providers
            .iter()
            .filter(|p| role_filter(p.role) && p.stats.healthy.load(Ordering::Relaxed))
            .collect();

        if eligible.is_empty() {
            // Fall back to any provider if none healthy
            return self
                .providers
                .iter()
                .find(|p| role_filter(p.role))
                .map(|p| Arc::clone(&p.provider));
        }

        match self.strategy {
            LoadBalanceStrategy::RoundRobin => {
                let idx = self.round_robin_idx.fetch_add(1, Ordering::Relaxed) % eligible.len();
                Some(Arc::clone(&eligible[idx].provider))
            }
            LoadBalanceStrategy::WeightedRandom => {
                let total_weight: u32 = eligible.iter().map(|p| p.weight).sum();
                let mut random = (std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
                    % total_weight as u128) as u32;

                for p in &eligible {
                    if random < p.weight {
                        return Some(Arc::clone(&p.provider));
                    }
                    random -= p.weight;
                }
                eligible.first().map(|p| Arc::clone(&p.provider))
            }
            LoadBalanceStrategy::LeastLatency => eligible
                .iter()
                .min_by_key(|p| p.stats.health_latency_ms.load(Ordering::Relaxed))
                .map(|p| Arc::clone(&p.provider)),
        }
    }

    /// Send a completion request using load balancing
    pub async fn complete(
        &self,
        request: &LlmRequest,
        for_sub_call: bool,
    ) -> Result<LlmResponse, ProviderError> {
        let provider = if for_sub_call {
            self.get_sub_provider()
        } else {
            self.get_root_provider()
        }
        .ok_or_else(|| ProviderError::ConnectionError("No providers available".to_string()))?;

        let stats = self.stats_map.get(provider.name()).map(|s| Arc::clone(&s));

        if let Some(ref stats) = stats {
            stats.requests.fetch_add(1, Ordering::Relaxed);
        }

        let start = Instant::now();
        let result = provider.complete(request).await;
        let latency = start.elapsed().as_millis() as u64;

        if let Some(stats) = stats {
            match &result {
                Ok(_) => {
                    stats.successes.fetch_add(1, Ordering::Relaxed);
                    stats.total_latency_ms.fetch_add(latency, Ordering::Relaxed);
                }
                Err(_) => {
                    stats.failures.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        result
    }

    /// Run health checks on all providers
    pub async fn health_check_all(&self) {
        for pooled in &self.providers {
            let status = pooled.provider.health_check().await;
            pooled
                .stats
                .healthy
                .store(status.healthy, Ordering::Relaxed);

            if let Some(latency) = status.latency_ms {
                pooled
                    .stats
                    .health_latency_ms
                    .store(latency, Ordering::Relaxed);
            }

            if status.healthy {
                debug!(provider = %pooled.provider.name(), latency_ms = ?status.latency_ms, "Health check passed");
            } else {
                warn!(provider = %pooled.provider.name(), error = ?status.error, "Health check failed");
            }
        }
    }

    /// Start background health check task
    pub fn start_health_check_task(self: Arc<Self>, interval_secs: u64) {
        tokio::spawn(async move {
            let mut ticker = interval(Duration::from_secs(interval_secs));
            loop {
                ticker.tick().await;
                self.health_check_all().await;
            }
        });
    }

    /// Get stats for a provider by name
    pub fn get_stats(&self, name: &str) -> Option<Arc<ProviderStats>> {
        self.stats_map.get(name).map(|s| Arc::clone(&s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_role_parsing() {
        assert_eq!(ProviderRole::from("root"), ProviderRole::Root);
        assert_eq!(ProviderRole::from("sub"), ProviderRole::Sub);
        assert_eq!(ProviderRole::from("both"), ProviderRole::Both);
        assert_eq!(ProviderRole::from("anything"), ProviderRole::Both);
    }
}
