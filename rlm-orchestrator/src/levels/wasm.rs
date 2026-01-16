//! Level 2: WASM - Sandboxed computation
//!
//! WASM commands provide sandboxed code execution with fuel and memory limits.
//! Enabled by default as part of the safe default configuration.

use super::{Level, RiskLevel};
use crate::WasmConfig;

/// Level 2: WASM (WebAssembly sandboxed execution)
pub struct WasmLevel {
    enabled: bool,
    rust_wasm_enabled: bool,
}

impl WasmLevel {
    pub fn new(config: &WasmConfig) -> Self {
        Self {
            enabled: config.enabled,
            rust_wasm_enabled: config.rust_wasm_enabled,
        }
    }

    /// Check if Rust-to-WASM compilation is enabled
    pub fn is_rust_wasm_enabled(&self) -> bool {
        self.enabled && self.rust_wasm_enabled
    }
}

impl Level for WasmLevel {
    fn id(&self) -> &'static str {
        "wasm"
    }

    fn name(&self) -> &'static str {
        "WASM MapReduce"
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn supported_commands(&self) -> &[&'static str] {
        &[
            // Pre-compiled WASM
            "wasm",
            "wasm_wat",
            "wasm_template",
            // Rust-to-WASM (requires rust_wasm_enabled)
            "rust_wasm",
            "rust_wasm_intent",
            "rust_wasm_reduce_intent",
            "rust_wasm_mapreduce",
        ]
    }

    fn risk_level(&self) -> RiskLevel {
        RiskLevel::Low
    }

    fn description(&self) -> &'static str {
        "sandboxed computation with fuel/memory limits"
    }
}
