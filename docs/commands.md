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

Find all occurrences of a text string.

```json
{
  "op": "find",
  "text": "ERROR",
  "on": "context",
  "store": "error_positions"
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to search for |
| `on` | string | "context" | Source variable name |
| `store` | string | null | Variable to store positions |

**Output:** Newline-separated character positions of matches.

### regex

Search using a regular expression pattern.

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
| `pattern` | string | required | Regex pattern |
| `on` | string | "context" | Source variable name |
| `store` | string | null | Variable to store matches |

**Output:** Newline-separated matched text.

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

## Limits

Configured in `config.toml`:

- `max_iterations` - Maximum command iterations (default: 20)
- `max_sub_calls` - Maximum llm_query calls (default: 50)
- `output_limit` - Maximum output characters per command (default: 10000)
