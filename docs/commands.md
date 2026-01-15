# RLM Commands Reference

The RLM orchestrator executes JSON commands issued by the root LLM. Each command performs an operation on the context or stored variables.

## Command Format

All commands are JSON objects with an `op` field:

```json
{"op": "command_name", ...parameters}
```

## Available Commands

### slice

Extract a character range from the context or a variable.

```json
{
  "op": "slice",
  "start": 0,
  "end": 1000,
  "on": "context",
  "store": "first_chunk"
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | int | required | Start character index |
| `end` | int | required | End character index |
| `on` | string | "context" | Source variable name |
| `store` | string | null | Variable to store result |

### lines

Extract a line range from the context or a variable.

```json
{
  "op": "lines",
  "start": 0,
  "end": 50,
  "on": "context",
  "store": "first_lines"
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | int | required | Start line number (0-indexed) |
| `end` | int | required | End line number (exclusive) |
| `on` | string | "context" | Source variable name |
| `store` | string | null | Variable to store result |

### find

Find all lines containing a text string (case-insensitive).

```json
{
  "op": "find",
  "text": "error",
  "on": "context",
  "store": "error_lines"
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to search for (case-insensitive) |
| `on` | string | "context" | Source variable name |
| `store` | string | null | Variable to store matching lines |

**Output:** Matching lines with line numbers, e.g.:
```
L42: ERROR: Connection failed
L157: error handling complete
```

**Fuzzy Search:** When searching with multiple words, `find` uses OR-based fuzzy matching:
- Words are split and each word (3+ chars) is searched independently
- Lines containing ANY search word are returned
- Results are ranked by number of matching words (most relevant first)
- Example: `"Prince Andrei secret vault"` finds lines containing "prince", "andrei", "secret", or "vault"

**Note:** Search is case-insensitive. A preview of the first 5 matches is always shown in the command output.

### regex

Search using a regular expression pattern (case-insensitive by default).

```json
{
  "op": "regex",
  "pattern": "def \\w+\\(",
  "on": "context",
  "store": "function_defs"
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pattern` | string | required | Regex pattern (case-insensitive) |
| `on` | string | "context" | Source variable name |
| `store` | string | null | Variable to store matching lines |

**Output:** Matching lines with line numbers (same format as `find`):
```
L42: def calculate_total(items):
L157: def process_order(order):
```

**Note:** Regex matching is case-insensitive by default. Both `find` and `regex` show a preview of the first 5 matches in the command output.

### count

Count lines, characters, or matches in a variable.

```json
{
  "op": "count",
  "what": "lines",
  "on": "context",
  "store": "line_count"
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `what` | string | required | "lines", "chars", or "matches" |
| `on` | string | "context" | Source variable name |
| `store` | string | null | Variable to store count |

**"matches" mode:** Counts newline-separated entries (works with `find`/`regex` output).

### llm_query

Delegate a question to a sub-LLM for semantic analysis.

```json
{
  "op": "llm_query",
  "prompt": "Summarize the key points: ${chunk}",
  "on": "chunk",
  "store": "summary"
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Prompt for the sub-LLM |
| `on` | string | null | Variable to include as context |
| `store` | string | null | Variable to store response |

**Variable expansion:** Use `${varname}` in the prompt to insert variable values.

### final

Return the final answer and end the loop.

```json
{
  "op": "final",
  "answer": "There are 3 ERROR lines in the log."
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `answer` | string | required | The final answer to return |

## Variables

Commands can store results in variables for later use:

```json
{"op": "find", "text": "ERROR", "store": "errors"}
{"op": "count", "what": "matches", "on": "errors", "store": "count"}
```

### Built-in Variables

- `context` - The original input context (always available)

### Variable Expansion

Use `${varname}` to insert variable values:

```json
{
  "op": "llm_query",
  "prompt": "Analyze these errors: ${errors}"
}
```

## Command Chaining

The root LLM issues one command per iteration. Results accumulate in variables, building up understanding:

```
Iteration 1: find "ERROR" → store in errors
Iteration 2: count matches on errors → know there are 3
Iteration 3: lines 0-10 on context → see header
Iteration 4: final → return answer
```

### wasm

Execute a pre-compiled WASM module from the library.

```json
{
  "op": "wasm",
  "module": "line_counter",
  "function": "analyze",
  "on": "context",
  "store": "line_count"
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | string | required | Name of pre-compiled module |
| `function` | string | "analyze" | Function to call |
| `on` | string | "context" | Variable to pass as input |
| `store` | string | null | Variable to store result |

**Available modules:**
- `line_counter` - Counts lines in text

### wasm_wat

Compile and execute WebAssembly Text format code.

```json
{
  "op": "wasm_wat",
  "wat": "(module ...)",
  "function": "analyze",
  "store": "result"
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wat` | string | required | WAT source code |
| `function` | string | "analyze" | Function to call |
| `on` | string | "context" | Variable to pass as input |
| `store` | string | null | Variable to store result |

**WASM Interface Requirements:**
Modules must export:
- `memory` - Linear memory
- `alloc(size: i32) -> i32` - Allocate memory for input
- `<function>(ptr: i32, len: i32) -> i32` - Analysis function
- `get_result_ptr() -> i32` - Get result string pointer
- `get_result_len() -> i32` - Get result string length

### rust_wasm

Compile and execute custom Rust analysis code. The Rust code is compiled to WASM and executed in a sandbox.

```json
{
  "op": "rust_wasm",
  "code": "pub fn analyze(input: &str) -> String { input.lines().count().to_string() }",
  "on": "context",
  "store": "line_count"
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `code` | string | required | Rust source code with `analyze` function |
| `on` | string | "context" | Variable to pass as input |
| `store` | string | null | Variable to store result |

**Function Signature (REQUIRED):**

```rust
pub fn analyze(input: &str) -> String {
    // Your analysis code
    // Return result as String
}
```

**Available in Rust code:**
- Core Rust: iterators, pattern matching, string operations, formatting
- Collections: `HashMap`, `HashSet`, `BTreeMap`, `BTreeSet`, `VecDeque`
- Standard library types and traits

**NOT Available (blocked for security):**
- File I/O (`std::fs`)
- Network (`std::net`)
- Process spawning (`std::process`)
- Environment access (`std::env`)
- External crates (`extern crate`)
- File inclusion macros (`include!`, `include_str!`, `include_bytes!`)

**Examples:**

Word frequency analysis:
```json
{
  "op": "rust_wasm",
  "code": "pub fn analyze(input: &str) -> String { let mut counts: HashMap<&str, usize> = HashMap::new(); for word in input.split_whitespace() { *counts.entry(word).or_insert(0) += 1; } let mut pairs: Vec<_> = counts.into_iter().collect(); pairs.sort_by(|a, b| b.1.cmp(&a.1)); pairs.iter().take(10).map(|(w, c)| format!(\"{}: {}\", w, c)).collect::<Vec<_>>().join(\", \") }",
  "store": "top_words"
}
```

Sum numbers in text:
```json
{
  "op": "rust_wasm",
  "code": "pub fn analyze(input: &str) -> String { let sum: i64 = input.split_whitespace().filter_map(|s| s.parse::<i64>().ok()).sum(); sum.to_string() }",
  "on": "data",
  "store": "total"
}
```

Custom pattern counting:
```json
{
  "op": "rust_wasm",
  "code": "pub fn analyze(input: &str) -> String { input.lines().filter(|l| l.contains(\"ERROR\") && l.contains(\"timeout\")).count().to_string() }",
  "store": "timeout_errors"
}
```

**When to use rust_wasm:**
- Complex aggregations that can't be done with `count`
- Frequency analysis and statistics
- Custom filtering with multiple conditions
- Numeric computations across data
- Multi-step transformations

**When NOT to use rust_wasm:**
- Simple text search → use `find` or `regex`
- Basic counting → use `count`
- Line/character extraction → use `lines` or `slice`

**Performance Notes:**
- First execution requires compilation (~1-2 seconds)
- Subsequent executions of identical code use cached WASM (~instant)
- WASM execution is sandboxed with fuel limits

## Limits

Configured in `config.toml`:

- `max_iterations` - Maximum command iterations (default: 20)
- `max_sub_calls` - Maximum llm_query calls (default: 50)
- `output_limit` - Maximum output characters per command (default: 10000)

### WASM Limits

- `fuel_limit` - Maximum WASM instructions (default: 1,000,000)
- `memory_limit` - Maximum WASM memory (default: 64MB)
