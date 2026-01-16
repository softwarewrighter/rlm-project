//! Level 4: LLM Delegation - Chunk-based LLM analysis
//!
//! LLM delegation commands send chunks of text to specialized LLMs for
//! fuzzy/semantic analysis. Non-deterministic. Disabled by default.

use super::{Level, RiskLevel};
use crate::LlmDelegationConfig;

/// Level 4: LLM Delegation (chunk-based LLM analysis)
pub struct LlmDelegationLevel {
    enabled: bool,
    privacy_mode: String,
}

impl LlmDelegationLevel {
    pub fn new(config: &LlmDelegationConfig) -> Self {
        Self {
            enabled: config.enabled,
            privacy_mode: config.privacy_mode.clone(),
        }
    }

    /// Get the privacy mode
    pub fn privacy_mode(&self) -> &str {
        &self.privacy_mode
    }
}

impl Level for LlmDelegationLevel {
    fn id(&self) -> &'static str {
        "llm_delegation"
    }

    fn name(&self) -> &'static str {
        "LLM Delegation"
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn supported_commands(&self) -> &[&'static str] {
        &[
            "llm_query",
            "llm_delegate_chunks",
        ]
    }

    fn risk_level(&self) -> RiskLevel {
        RiskLevel::High
    }

    fn description(&self) -> &'static str {
        "semantic analysis via specialized LLMs"
    }
}
