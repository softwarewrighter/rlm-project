# RLM Dynamic WASM Analysis - Implementation Plan

## Overview

This plan breaks down the WASM feature implementation into phases, with clear deliverables and dependencies.

## Prerequisites

Before starting, ensure the development environment has:

- [ ] Rust toolchain installed (`rustup`)
- [ ] WASM target installed: `rustup target add wasm32-unknown-unknown`
- [ ] `wasm-opt` (optional): `brew install binaryen` or equivalent

## Phase 1: Rust Compiler Service

**Goal**: Create a service that compiles Rust source code to WASM bytecode.

### Tasks

#### 1.1 Create Compiler Module Structure
- [ ] Create `src/wasm/compiler.rs`
- [ ] Define `CompileError` enum
- [ ] Define `RustCompiler` struct

#### 1.2 Implement Compiler Detection
- [ ] Find `rustc` in PATH
- [ ] Check fallback locations
- [ ] Verify `wasm32-unknown-unknown` target is installed
- [ ] Return helpful error if not available

#### 1.3 Create Code Template
- [ ] Create `src/wasm/templates/` directory
- [ ] Write `wasm_template.rs` with:
  - Memory allocation functions
  - Result storage globals
  - Entry point wrapper
  - Panic handler
- [ ] Add `{user_code}` placeholder

#### 1.4 Implement Compilation
- [ ] Generate full source by inserting user code into template
- [ ] Write to temp file
- [ ] Invoke `rustc` with correct flags
- [ ] Capture stdout/stderr
- [ ] Read WASM output on success
- [ ] Parse and clean error messages on failure
- [ ] Clean up temp files

#### 1.5 Add Source Validation
- [ ] Check for required `analyze` function signature
- [ ] Block forbidden patterns (`std::fs`, `include!`, etc.)
- [ ] Validate basic Rust syntax (optional: use `syn` crate)

#### 1.6 Unit Tests
- [ ] Test successful compilation
- [ ] Test compilation errors
- [ ] Test validation failures
- [ ] Test missing compiler handling

**Deliverable**: `RustCompiler::compile(&str) -> Result<Vec<u8>, CompileError>`

---

## Phase 2: Module Cache

**Goal**: Cache compiled WASM modules to avoid redundant compilation.

### Tasks

#### 2.1 Create Cache Module
- [ ] Create `src/wasm/cache.rs`
- [ ] Add `lru` crate to dependencies
- [ ] Define `ModuleCache` struct

#### 2.2 Implement Memory Cache
- [ ] Use `LruCache<String, Vec<u8>>`
- [ ] Implement `get(source) -> Option<Vec<u8>>`
- [ ] Implement `put(source, wasm)`
- [ ] Use MD5 hash as cache key

#### 2.3 Implement Disk Cache
- [ ] Create cache directory on init
- [ ] Write WASM files with hash as filename
- [ ] Read from disk on memory cache miss
- [ ] Promote disk hits to memory cache

#### 2.4 Add Cache Management
- [ ] Implement `clear()` method
- [ ] Implement `stats()` method
- [ ] Add LRU eviction for memory cache
- [ ] Add size-based cleanup for disk cache

#### 2.5 Unit Tests
- [ ] Test cache hit/miss
- [ ] Test disk persistence
- [ ] Test LRU eviction
- [ ] Test cache clearing

**Deliverable**: `ModuleCache` with two-tier caching

---

## Phase 3: Command Executor Integration

**Goal**: Add `rust_wasm` command to the command executor.

### Tasks

#### 3.1 Define Command Variant
- [ ] Add `RustWasm` to `Command` enum
- [ ] Add `code: String` field
- [ ] Add `on: Option<String>` field
- [ ] Add `store: Option<String>` field

#### 3.2 Add Dependencies to Executor
- [ ] Add `rust_compiler: RustCompiler` field
- [ ] Add `wasm_cache: ModuleCache` field
- [ ] Initialize in `CommandExecutor::new()`
- [ ] Make fields optional (graceful degradation if compiler unavailable)

#### 3.3 Implement Execution Logic
- [ ] Check cache for compiled module
- [ ] Compile on cache miss
- [ ] Store in cache
- [ ] Execute with `WasmExecutor`
- [ ] Store result in variable
- [ ] Return output message

#### 3.4 Error Handling
- [ ] Add `WasmCompileError` to `CommandError`
- [ ] Format compilation errors for LLM
- [ ] Handle runtime errors gracefully

#### 3.5 Integration Tests
- [ ] Test successful execution
- [ ] Test compilation error handling
- [ ] Test runtime error handling
- [ ] Test cache integration

**Deliverable**: Working `rust_wasm` command

---

## Phase 4: System Prompt and Examples

**Goal**: Update the system prompt to teach the LLM how to use `rust_wasm`.

### Tasks

#### 4.1 Update System Prompt
- [ ] Add `rust_wasm` command documentation
- [ ] Explain function signature requirements
- [ ] List available standard library features
- [ ] List restrictions (no fs, net, etc.)

#### 4.2 Add Examples
- [ ] Word frequency example
- [ ] Numeric aggregation example
- [ ] Pattern extraction example
- [ ] JSON parsing example (if serde available)

#### 4.3 Add Usage Guidelines
- [ ] When to use `rust_wasm` vs built-in commands
- [ ] Common patterns and idioms
- [ ] Error recovery tips

#### 4.4 Test with Real LLM
- [ ] Test with various models (llama3.2, qwen3, etc.)
- [ ] Verify LLM can generate valid code
- [ ] Test error recovery flow
- [ ] Collect failure cases for prompt improvement

**Deliverable**: Updated system prompt with WASM documentation

---

## Phase 5: Configuration and CLI

**Goal**: Make WASM feature configurable.

### Tasks

#### 5.1 Add Config Options
- [ ] Add `[wasm]` section to `config.toml`
- [ ] Add `enabled` flag
- [ ] Add `fuel_limit` option
- [ ] Add `memory_limit` option
- [ ] Add `cache_dir` option
- [ ] Add `compiler.rustc_path` option

#### 5.2 Update CLI
- [ ] Add `--wasm-enabled` flag
- [ ] Add `--wasm-cache-dir` flag
- [ ] Add `--wasm-fuel-limit` flag
- [ ] Show WASM status in `--dry-run` output

#### 5.3 Update Server API
- [ ] Include WASM capability in `/health` endpoint
- [ ] Add cache stats to `/stats` endpoint
- [ ] Add optional compile-only endpoint for debugging

#### 5.4 Documentation
- [ ] Update `docs/commands.md` with `rust_wasm`
- [ ] Add WASM configuration to README
- [ ] Add troubleshooting guide

**Deliverable**: Fully configurable WASM feature

---

## Phase 6: Testing and Hardening

**Goal**: Ensure reliability and security.

### Tasks

#### 6.1 Security Testing
- [ ] Test sandbox escape attempts
- [ ] Test resource exhaustion
- [ ] Test malicious code patterns
- [ ] Fuzz test compiler with random inputs

#### 6.2 Performance Testing
- [ ] Benchmark compilation time
- [ ] Benchmark execution time vs built-in commands
- [ ] Measure cache hit rate in realistic scenarios
- [ ] Profile memory usage

#### 6.3 Integration Testing
- [ ] End-to-end tests with various queries
- [ ] Test with multiple LLM models
- [ ] Test error recovery paths
- [ ] Test graceful degradation when compiler unavailable

#### 6.4 Documentation Review
- [ ] Code comments and docstrings
- [ ] Architecture diagram accuracy
- [ ] Example code correctness
- [ ] Config documentation completeness

**Deliverable**: Production-ready WASM feature

---

## Task Checklist Summary

### Phase 1: Compiler Service
```
[ ] 1.1 Module structure
[ ] 1.2 Compiler detection
[ ] 1.3 Code template
[ ] 1.4 Compilation logic
[ ] 1.5 Source validation
[ ] 1.6 Unit tests
```

### Phase 2: Cache
```
[ ] 2.1 Cache module
[ ] 2.2 Memory cache
[ ] 2.3 Disk cache
[ ] 2.4 Cache management
[ ] 2.5 Unit tests
```

### Phase 3: Command Integration
```
[ ] 3.1 Command variant
[ ] 3.2 Executor dependencies
[ ] 3.3 Execution logic
[ ] 3.4 Error handling
[ ] 3.5 Integration tests
```

### Phase 4: System Prompt
```
[ ] 4.1 Update prompt
[ ] 4.2 Add examples
[ ] 4.3 Usage guidelines
[ ] 4.4 LLM testing
```

### Phase 5: Configuration
```
[ ] 5.1 Config options
[ ] 5.2 CLI flags
[ ] 5.3 Server API
[ ] 5.4 Documentation
```

### Phase 6: Hardening
```
[ ] 6.1 Security testing
[ ] 6.2 Performance testing
[ ] 6.3 Integration testing
[ ] 6.4 Documentation review
```

---

## Dependencies Graph

```
Phase 1 ─────────────────────────────────────────────────┐
(Compiler)                                               │
    │                                                    │
    ▼                                                    │
Phase 2 ──────────────────────────────────┐              │
(Cache)                                   │              │
    │                                     │              │
    ▼                                     ▼              │
Phase 3 ◀─────────────────────────────────┘              │
(Command)                                                │
    │                                                    │
    ├────────────────────────────────────────────────────┘
    │
    ▼
Phase 4 ──────────────────────────────────┐
(Prompt)                                  │
    │                                     │
    ▼                                     │
Phase 5 ──────────────────────────────────┤
(Config)                                  │
    │                                     │
    ▼                                     │
Phase 6 ◀─────────────────────────────────┘
(Testing)
```

---

## Risk Mitigation

| Risk | Mitigation | Contingency |
|------|------------|-------------|
| rustc not available on target system | Document as prerequisite | Disable feature gracefully |
| Compilation too slow | Aggressive caching | Pre-compile common patterns |
| LLM generates invalid code | Good examples, retry mechanism | Fallback to built-in commands |
| WASM sandbox vulnerability | Use wasmtime (proven runtime) | Regular security updates |

---

## Success Criteria

1. **Functional**: `rust_wasm` command executes successfully
2. **Cached**: Second execution of same code is instant (<10ms)
3. **Safe**: No sandbox escapes possible
4. **Documented**: LLM can use feature without additional prompting
5. **Reliable**: < 1% failure rate after LLM retry

---

## Getting Started

To begin implementation:

```bash
# Ensure WASM target is available
rustup target add wasm32-unknown-unknown

# Create branch
git checkout -b feature/rust-wasm

# Start with Phase 1
mkdir -p rlm-orchestrator/src/wasm/templates
touch rlm-orchestrator/src/wasm/compiler.rs
```

First task: Implement `RustCompiler::find_rustc()` to detect the Rust compiler.
