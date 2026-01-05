# WASM Dynamic Code Execution - Design Document

## Goal

Enable the root LLM to generate and execute arbitrary analysis code in a sandboxed WebAssembly environment, going beyond the fixed command set.

## Research Summary

### WASM Runtimes for Rust

| Runtime | Pros | Cons |
|---------|------|------|
| **wasmtime** | Fast, secure, Bytecode Alliance backed, good Rust API | API changes frequently |
| **wasmer** | Multiple backends, WASIX extensions, can serialize native code | Slightly slower |
| **wasmedge** | Good for edge/IoT | Less Rust-focused |

**Recommendation:** Use **wasmtime** for its security focus and performance.

### How LLMs Generate Executable Code

Several approaches exist:

1. **Generate Python → Pyodide (WASM)** - Good for browser, complex setup
2. **Generate WAT directly** - Low-level, error-prone for LLMs
3. **Generate Rust → compile to WASM** - Powerful but slow compile
4. **Generate simple DSL → interpret** - Fast, limited
5. **Call pre-compiled WASM modules** - Hybrid approach

## Proposed Architecture

### Phase 1: Hybrid Command + Code (Recommended Start)

Add a new `code` command that executes a simple expression language:

```json
{
  "op": "code",
  "lang": "expr",
  "code": "context.lines().filter(|l| l.contains('ERROR')).count()",
  "store": "error_count"
}
```

The expression language supports:
- String methods: `lines()`, `split()`, `contains()`, `trim()`
- Iterators: `filter()`, `map()`, `count()`, `collect()`
- Basic logic: `if/else`, `&&`, `||`
- No I/O, no loops (use iterator methods instead)

**Implementation:** Parse with a simple Rust parser, execute directly (no WASM yet).

### Phase 2: WASM Execution

For complex analysis, allow full WASM modules:

```json
{
  "op": "wasm",
  "module": "<base64-encoded WASM binary>",
  "function": "analyze",
  "args": ["${context}"],
  "store": "result"
}
```

**How the LLM generates WASM:**

Option A: **LLM generates Rust source → we compile**
```json
{
  "op": "compile_wasm",
  "source": "#[no_mangle] pub fn analyze(ctx: &str) -> String { ... }",
  "store": "module"
}
```
Then:
```json
{
  "op": "wasm",
  "module": "${module}",
  "function": "analyze",
  "args": ["${context}"]
}
```

Option B: **Pre-compiled analysis modules**
We provide a library of WASM modules:
- `text_analysis.wasm` - grep, count, transform
- `json_analysis.wasm` - parse, query, transform
- `code_analysis.wasm` - AST parsing, symbol extraction

LLM calls them:
```json
{
  "op": "wasm",
  "module": "text_analysis",
  "function": "count_pattern",
  "args": ["${context}", "ERROR"]
}
```

### Phase 3: Full Dynamic WASM

LLM generates WAT (WebAssembly Text) or uses a compile service:

```json
{
  "op": "wasm_from_wat",
  "wat": "(module (func (export \"analyze\") ...))",
  "store": "module"
}
```

## Security Model

### wasmtime Sandboxing
- **Memory isolation**: WASM runs in linear memory, cannot access host
- **No I/O by default**: No filesystem, network, or system calls
- **Fuel limits**: Cap instruction count to prevent infinite loops
- **Memory limits**: Cap memory allocation

### Configuration
```toml
[wasm]
enabled = true
fuel_limit = 1_000_000      # Max instructions
memory_limit_mb = 64        # Max memory
timeout_ms = 5000           # Max wall-clock time
```

## Implementation Plan

### Step 1: Add wasmtime dependency
```toml
[dependencies]
wasmtime = "18"
```

### Step 2: Create WASM executor module
```rust
// src/wasm/mod.rs
pub struct WasmExecutor {
    engine: wasmtime::Engine,
    fuel_limit: u64,
}

impl WasmExecutor {
    pub fn execute(&self, module: &[u8], func: &str, args: Vec<String>)
        -> Result<String, WasmError>;
}
```

### Step 3: Add `code` command (Phase 1)
Simple expression evaluator for immediate utility.

### Step 4: Add `wasm` command (Phase 2)
Execute pre-compiled or provided WASM modules.

### Step 5: Add `compile_wasm` command (Phase 3)
Compile Rust/WAT to WASM on-demand.

## Example: Complex Analysis with WASM

**Query:** "Categorize all errors by type and count each"

**Without WASM (current):**
```
Iteration 1: {"op": "regex", "pattern": "ERROR: (\\w+)", "store": "errors"}
Iteration 2: {"op": "llm_query", "prompt": "Categorize these: ${errors}"}
... (multiple iterations, sub-LLM calls)
```

**With WASM:**
```
Iteration 1: {
  "op": "code",
  "lang": "expr",
  "code": "context.lines().filter(|l| l.contains('ERROR')).map(|l| l.split(':').nth(1).trim()).group_by(|t| t).map(|(k,v)| format!(\"{}: {}\", k, v.len())).join('\\n')",
  "store": "categorized"
}
Iteration 2: {"op": "final", "answer": "${categorized}"}
```

**Result:** Fewer iterations, no sub-LLM calls, deterministic.

## Trade-offs

| Approach | Flexibility | Safety | Speed | LLM Difficulty |
|----------|-------------|--------|-------|----------------|
| JSON commands only | Low | High | Fast | Easy |
| Expression DSL | Medium | High | Fast | Medium |
| Pre-compiled WASM | Medium | High | Fast | Easy |
| LLM-generated WASM | High | High | Slower (compile) | Hard |

## Recommendation

Start with **Phase 1 (Expression DSL)** as it provides:
- Immediate value (fewer iterations for common patterns)
- No compile overhead
- Safe (we control the parser)
- Good LLM compatibility (looks like Rust)

Then evolve to **Phase 2 (Pre-compiled WASM)** for specialized analysis.

**Phase 3** (LLM-generates-WASM) is powerful but requires:
- Robust compilation pipeline
- Good LLM prompting for WASM-friendly code
- Consider only for advanced use cases

## References

- [NVIDIA: Sandboxing Agentic AI with WebAssembly](https://developer.nvidia.com/blog/sandboxing-agentic-ai-workflows-with-webassembly/)
- [wasmtime Documentation](https://docs.wasmtime.dev/)
- [Comparing WASM Runtimes](https://blog.logrocket.com/webassembly-runtimes-compared/)
- [llm-wasm-sandbox](https://pypi.org/project/llm-wasm-sandbox/)
