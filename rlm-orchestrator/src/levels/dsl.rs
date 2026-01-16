//! Level 1: DSL - Safe text operations
//!
//! DSL commands provide safe, read-only text manipulation operations.
//! These are the safest operations and are enabled by default.

use super::{Level, RiskLevel};
use crate::DslConfig;

/// Level 1: DSL (Domain Specific Language for text operations)
pub struct DslLevel {
    enabled: bool,
}

impl DslLevel {
    pub fn new(config: &DslConfig) -> Self {
        Self {
            enabled: config.enabled,
        }
    }
}

impl Level for DslLevel {
    fn id(&self) -> &'static str {
        "dsl"
    }

    fn name(&self) -> &'static str {
        "DSL"
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn supported_commands(&self) -> &[&'static str] {
        &[
            // Text extraction
            "slice",
            "lines",
            "regex",
            "find",
            "count",
            "split",
            "len",
            // Variables
            "set",
            "get",
            "print",
            // Control flow
            "final",
            "final_var",
        ]
    }

    fn risk_level(&self) -> RiskLevel {
        RiskLevel::VeryLow
    }

    fn description(&self) -> &'static str {
        "safe text extraction and filtering"
    }
}
