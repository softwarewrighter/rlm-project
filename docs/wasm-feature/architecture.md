# RLM Dynamic WASM Analysis - Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RLM Orchestrator                                │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────────────┐ │
│  │  Root LLM   │───▶│   Command    │───▶│         WASM Subsystem          │ │
│  │  (Ollama)   │    │   Executor   │    │  ┌───────────┐  ┌────────────┐  │ │
│  └─────────────┘    └──────────────┘    │  │  Compiler │  │  Runtime   │  │ │
│        │                   │            │  │  Service  │  │ (wasmtime) │  │ │
│        │                   │            │  └─────┬─────┘  └─────┬──────┘  │ │
│        ▼                   ▼            │        │              │         │ │
│  ┌─────────────┐    ┌──────────────┐    │        ▼              ▼         │ │
│  │   System    │    │   Variable   │    │  ┌───────────────────────────┐  │ │
│  │   Prompt    │    │    Store     │    │  │     Module Cache (LRU)    │  │ │
│  └─────────────┘    └──────────────┘    │  └───────────────────────────┘  │ │
│                                         └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Command Executor (Existing)

The existing `CommandExecutor` in `src/commands/mod.rs` handles JSON commands. It already supports:
- `wasm`: Execute pre-compiled WASM modules
- `wasm_wat`: Compile and execute WAT code

**Extension Point**: Add new `rust_wasm` command that accepts Rust source code.

### 2. WASM Subsystem (Enhanced)

```
┌─────────────────────────────────────────────────────────────────┐
│                       WASM Subsystem                             │
│                                                                  │
│  ┌──────────────────┐      ┌──────────────────────────────────┐ │
│  │  RustCompiler    │      │        WasmExecutor              │ │
│  │                  │      │  (existing - enhanced)           │ │
│  │  - Template gen  │      │                                  │ │
│  │  - rustc invoke  │─────▶│  - Fuel limiting                 │ │
│  │  - Error capture │      │  - Memory limiting               │ │
│  │  - WASM output   │      │  - String I/O                    │ │
│  └────────┬─────────┘      └──────────────────────────────────┘ │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐      ┌──────────────────────────────────┐ │
│  │  ModuleCache     │      │        WasmLibrary               │ │
│  │                  │      │  (existing - extended)           │ │
│  │  - Hash-based    │      │                                  │ │
│  │  - LRU eviction  │      │  - Pre-compiled modules          │ │
│  │  - Disk persist  │      │  - Dynamic module registration   │ │
│  └──────────────────┘      └──────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Rust Compiler Service

Responsible for converting Rust source to WASM bytecode.

```rust
pub struct RustCompiler {
    /// Path to rustc binary
    rustc_path: PathBuf,
    /// Temporary directory for compilation
    temp_dir: PathBuf,
    /// Sysroot for wasm32-unknown-unknown target
    wasm_sysroot: Option<PathBuf>,
}

impl RustCompiler {
    /// Compile Rust source to WASM
    pub fn compile(&self, source: &str) -> Result<Vec<u8>, CompileError>;

    /// Check if wasm32 target is available
    pub fn check_target(&self) -> bool;
}
```

**Compilation Flow**:
1. Wrap user code in template (see Design doc)
2. Write to temp file
3. Invoke `rustc --target wasm32-unknown-unknown`
4. Read WASM output or capture errors
5. Clean up temp files

### 4. Module Cache

```rust
pub struct ModuleCache {
    /// In-memory cache: hash -> compiled WASM
    memory: LruCache<String, Vec<u8>>,
    /// Disk cache directory
    cache_dir: PathBuf,
    /// Maximum memory cache size
    max_memory_items: usize,
}

impl ModuleCache {
    /// Get cached module by source hash
    pub fn get(&self, source: &str) -> Option<Vec<u8>>;

    /// Store compiled module
    pub fn put(&mut self, source: &str, wasm: Vec<u8>);

    /// Compute cache key from source
    fn hash_source(source: &str) -> String;
}
```

**Cache Strategy**:
- L1: In-memory LRU cache (fast, ~100 modules)
- L2: Disk cache (slower, ~1000 modules)
- Key: MD5 hash of source code
- Eviction: LRU when limits exceeded

## Data Flow

### Happy Path: Cache Hit

```
LLM                    CommandExecutor           Cache              Runtime
 │                           │                     │                   │
 │  rust_wasm command        │                     │                   │
 │──────────────────────────▶│                     │                   │
 │                           │  lookup(hash)       │                   │
 │                           │────────────────────▶│                   │
 │                           │  wasm bytes         │                   │
 │                           │◀────────────────────│                   │
 │                           │                     │   execute(wasm)   │
 │                           │─────────────────────────────────────────▶│
 │                           │                     │   result string   │
 │                           │◀─────────────────────────────────────────│
 │  output                   │                     │                   │
 │◀──────────────────────────│                     │                   │
```

### Cache Miss with Successful Compilation

```
LLM              CommandExecutor        Compiler            Cache           Runtime
 │                     │                   │                  │                │
 │  rust_wasm          │                   │                  │                │
 │────────────────────▶│                   │                  │                │
 │                     │  lookup(hash)     │                  │                │
 │                     │─────────────────────────────────────▶│                │
 │                     │  miss             │                  │                │
 │                     │◀─────────────────────────────────────│                │
 │                     │  compile(source)  │                  │                │
 │                     │──────────────────▶│                  │                │
 │                     │  wasm bytes       │                  │                │
 │                     │◀──────────────────│                  │                │
 │                     │  store(hash,wasm) │                  │                │
 │                     │─────────────────────────────────────▶│                │
 │                     │                   │  execute(wasm)   │                │
 │                     │───────────────────────────────────────────────────────▶│
 │                     │                   │  result          │                │
 │                     │◀───────────────────────────────────────────────────────│
 │  output             │                   │                  │                │
 │◀────────────────────│                   │                  │                │
```

### Compilation Error with Retry

```
LLM              CommandExecutor        Compiler
 │                     │                   │
 │  rust_wasm (bad)    │                   │
 │────────────────────▶│                   │
 │                     │  compile(source)  │
 │                     │──────────────────▶│
 │                     │  CompileError     │
 │                     │◀──────────────────│
 │  error message      │                   │
 │◀────────────────────│                   │
 │                     │                   │
 │  rust_wasm (fixed)  │                   │
 │────────────────────▶│                   │
 │                     │  compile(source)  │
 │                     │──────────────────▶│
 │                     │  wasm bytes       │
 │                     │◀──────────────────│
 │  ... continues ...  │                   │
```

## Security Architecture

### Sandboxing Layers

```
┌────────────────────────────────────────────────────────────┐
│                    Host System                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  RLM Process                          │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │              wasmtime Runtime                   │  │  │
│  │  │  ┌──────────────────────────────────────────┐  │  │  │
│  │  │  │           WASM Module                     │  │  │  │
│  │  │  │                                           │  │  │  │
│  │  │  │  - Linear memory (isolated)               │  │  │  │
│  │  │  │  - No filesystem                          │  │  │  │
│  │  │  │  - No network                             │  │  │  │
│  │  │  │  - No syscalls                            │  │  │  │
│  │  │  │  - Fuel-limited execution                 │  │  │  │
│  │  │  └──────────────────────────────────────────┘  │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

### Resource Limits

| Resource | Limit | Enforcement |
|----------|-------|-------------|
| CPU | 10M instructions | wasmtime fuel |
| Memory | 64MB | wasmtime memory limit |
| Time | 5 seconds | Timeout wrapper |
| Stack | 1MB | WASM default |

### Compilation Security

- Temporary files in isolated directory
- Compiler runs with minimal permissions
- Output validated as valid WASM before execution
- Source code sanitized (no file includes)

## Integration Points

### 1. System Prompt Enhancement

The system prompt must be updated to include:
- `rust_wasm` command syntax
- Rust code requirements (function signature)
- Available standard library features
- Example use cases

### 2. Config Extension

```toml
[wasm]
fuel_limit = 10_000_000
memory_limit_mb = 64
timeout_ms = 5000
cache_size = 100
cache_dir = "~/.rlm/wasm-cache"

[wasm.compiler]
rustc_path = "rustc"
target = "wasm32-unknown-unknown"
```

### 3. CLI Flags

```
--wasm-enabled        Enable dynamic WASM compilation (default: true)
--wasm-cache-dir      Cache directory for compiled modules
--wasm-fuel-limit     Maximum WASM instructions
```

## Dependencies

### Required
- `wasmtime` 27.x (already included)
- `rustc` with `wasm32-unknown-unknown` target
- `md5` for cache hashing (already included)

### Optional
- `wasm-opt` for optimization (future)
- `wasm-strip` for size reduction (future)

## Future Considerations

### Phase 2: Enhanced Runtime
- Import common crates (`regex`, `serde_json`) as WASM modules
- Provide host functions for logging/debugging
- Support for multiple function signatures

### Phase 3: Optimization
- Pre-warm compiler process
- Parallel compilation for batch requests
- WASM optimization passes

### Phase 4: Advanced Features
- Streaming input/output for very large documents
- Multiple analysis passes with shared state
- Custom allocator for better memory efficiency
