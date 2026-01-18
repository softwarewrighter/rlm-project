//! Level 3: CLI - Native binary execution
//!
//! CLI commands compile and execute native Rust binaries with full stdlib access.
//! Process isolation only - no WASM sandbox. Disabled by default for safety.

use super::{Level, RiskLevel};
use crate::CliConfig;

/// Level 3: CLI (Native binary execution)
pub struct CliLevel {
    enabled: bool,
    sandbox_mode: String,
}

impl CliLevel {
    pub fn new(config: &CliConfig) -> Self {
        Self {
            enabled: config.enabled,
            sandbox_mode: config.sandbox_mode.clone(),
        }
    }

    /// Get the sandbox mode
    pub fn sandbox_mode(&self) -> &str {
        &self.sandbox_mode
    }
}

impl Level for CliLevel {
    fn id(&self) -> &'static str {
        "cli"
    }

    fn name(&self) -> &'static str {
        "Rust CLI"
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn supported_commands(&self) -> &[&'static str] {
        &["rust_cli_intent"]
    }

    fn risk_level(&self) -> RiskLevel {
        RiskLevel::Medium
    }

    fn description(&self) -> &'static str {
        "full Rust stdlib, process isolation only"
    }
}
