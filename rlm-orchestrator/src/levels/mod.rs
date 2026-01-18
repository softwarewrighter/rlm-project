//! Capability Levels for RLM Orchestrator
//!
//! The RLM system supports 4 capability levels with increasing power and risk:
//!
//! | Level | Name | Default | Risk | Description |
//! |-------|------|---------|------|-------------|
//! | 1 | DSL | ON | Very Low | Safe text operations |
//! | 2 | WASM | ON | Low | Sandboxed computation |
//! | 3 | CLI | OFF | Medium | Native binaries, process isolation |
//! | 4 | LLM Delegation | OFF | Variable | Chunk-based LLM analysis |

mod cli;
mod dsl;
mod llm;
mod wasm;

pub use cli::CliLevel;
pub use dsl::DslLevel;
pub use llm::LlmDelegationLevel;
pub use wasm::WasmLevel;

use crate::{CliConfig, DslConfig, LlmDelegationConfig, WasmConfig};

/// Risk level for a capability level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    /// Very low risk - read-only text operations
    VeryLow,
    /// Low risk - sandboxed code execution
    Low,
    /// Medium risk - process-isolated native code
    Medium,
    /// High risk - external LLM calls, non-deterministic
    High,
}

impl std::fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RiskLevel::VeryLow => write!(f, "very low"),
            RiskLevel::Low => write!(f, "low"),
            RiskLevel::Medium => write!(f, "medium"),
            RiskLevel::High => write!(f, "high"),
        }
    }
}

/// Trait for capability levels
pub trait Level: Send + Sync {
    /// Level identifier (e.g., "dsl", "wasm", "cli", "llm_delegation")
    fn id(&self) -> &'static str;

    /// Human-readable name (e.g., "DSL", "WASM MapReduce")
    fn name(&self) -> &'static str;

    /// Check if this level is enabled
    fn is_enabled(&self) -> bool;

    /// Get commands supported by this level
    fn supported_commands(&self) -> &[&'static str];

    /// Check if this level can handle a specific command
    fn can_handle(&self, cmd: &str) -> bool {
        self.is_enabled() && self.supported_commands().contains(&cmd)
    }

    /// Get risk level
    fn risk_level(&self) -> RiskLevel;

    /// Get a description of this level
    fn description(&self) -> &'static str;
}

/// Registry of capability levels
pub struct LevelRegistry {
    levels: Vec<Box<dyn Level>>,
    priority: Vec<String>,
}

impl LevelRegistry {
    /// Create a new level registry from configuration
    pub fn new(
        dsl_config: &DslConfig,
        wasm_config: &WasmConfig,
        cli_config: &CliConfig,
        llm_config: &LlmDelegationConfig,
        priority: Vec<String>,
    ) -> Self {
        let levels: Vec<Box<dyn Level>> = vec![
            Box::new(DslLevel::new(dsl_config)),
            Box::new(WasmLevel::new(wasm_config)),
            Box::new(CliLevel::new(cli_config)),
            Box::new(LlmDelegationLevel::new(llm_config)),
        ];

        Self { levels, priority }
    }

    /// Check if a level is enabled by ID
    pub fn is_level_enabled(&self, level_id: &str) -> bool {
        self.levels
            .iter()
            .find(|l| l.id() == level_id)
            .map(|l| l.is_enabled())
            .unwrap_or(false)
    }

    /// Find the best level to handle a command based on priority
    pub fn find_handler(&self, cmd: &str) -> Option<&dyn Level> {
        // Check levels in priority order
        for level_id in &self.priority {
            if let Some(level) = self.levels.iter().find(|l| l.id() == level_id)
                && level.can_handle(cmd)
            {
                return Some(level.as_ref());
            }
        }

        // Fallback: check all levels
        for level in &self.levels {
            if level.can_handle(cmd) {
                return Some(level.as_ref());
            }
        }

        None
    }

    /// Get all enabled levels
    pub fn enabled_levels(&self) -> Vec<&dyn Level> {
        self.levels
            .iter()
            .filter(|l| l.is_enabled())
            .map(|l| l.as_ref())
            .collect()
    }

    /// Get all levels
    pub fn all_levels(&self) -> Vec<&dyn Level> {
        self.levels.iter().map(|l| l.as_ref()).collect()
    }

    /// Get the priority order
    pub fn priority(&self) -> &[String] {
        &self.priority
    }

    /// Get level by ID
    pub fn get_level(&self, level_id: &str) -> Option<&dyn Level> {
        self.levels
            .iter()
            .find(|l| l.id() == level_id)
            .map(|l| l.as_ref())
    }

    /// Generate system prompt section describing available levels
    pub fn generate_prompt_section(&self) -> String {
        let mut lines = vec!["Available capability levels (in priority order):".to_string()];

        for level_id in &self.priority {
            if let Some(level) = self.get_level(level_id) {
                let status = if level.is_enabled() {
                    "ENABLED"
                } else {
                    "DISABLED"
                };
                let commands = level.supported_commands().join(", ");
                lines.push(format!(
                    "- Level {} ({}): {} - {} [{}]",
                    match level.id() {
                        "dsl" => "1",
                        "wasm" => "2",
                        "cli" => "3",
                        "llm_delegation" => "4",
                        _ => "?",
                    },
                    level.name(),
                    commands,
                    level.description(),
                    status
                ));
            }
        }

        // Add any levels not in priority
        for level in &self.levels {
            if !self.priority.contains(&level.id().to_string()) {
                let status = if level.is_enabled() {
                    "ENABLED"
                } else {
                    "DISABLED"
                };
                let commands = level.supported_commands().join(", ");
                lines.push(format!(
                    "- Level {} ({}): {} - {} [{}]",
                    match level.id() {
                        "dsl" => "1",
                        "wasm" => "2",
                        "cli" => "3",
                        "llm_delegation" => "4",
                        _ => "?",
                    },
                    level.name(),
                    commands,
                    level.description(),
                    status
                ));
            }
        }

        lines.push(String::new());
        lines.push("Use the lowest level that can accomplish the task.".to_string());

        lines.join("\n")
    }
}

impl std::fmt::Debug for LevelRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LevelRegistry")
            .field("priority", &self.priority)
            .field(
                "enabled_levels",
                &self
                    .enabled_levels()
                    .iter()
                    .map(|l| l.id())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}
