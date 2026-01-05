# How RLM Works

RLM operates as a **loop** where the AI issues commands, gets results, and iterates until it has enough information to answer.

## The Iteration Loop

```
┌─────────────┐
│   Query +   │
│   Context   │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│   Root LLM  │◄────│  Command    │
│  (thinks)   │────►│  Executor   │
└─────────────┘     └─────────────┘
       │                   │
       │ (loop until done) │
       │                   ▼
       │            ┌─────────────┐
       │            │  Sub-LLM    │
       │            │  (helpers)  │
       │            └─────────────┘
       ▼
┌─────────────┐
│   Final     │
│   Answer    │
└─────────────┘
```

## Example Walkthrough

**Query:** "How many ERROR lines are in this log?"

**Context:** (500 lines of log data)

### Iteration 1
**AI thinks:** "I need to find ERROR lines first"
```json
{"op": "find", "text": "ERROR", "store": "errors"}
```
**Result:** "Found 3 occurrences" (positions stored in `errors`)

### Iteration 2
**AI thinks:** "I found them, let me count"
```json
{"op": "count", "what": "matches", "on": "errors", "store": "error_count"}
```
**Result:** "Counted 3 lines"

### Iteration 3
**AI thinks:** "I have the answer!"
```json
{"op": "final", "answer": "There are 3 ERROR lines in the log."}
```

## Available Commands

| Command | What It Does | Example |
|---------|--------------|---------|
| `slice` | Get character range | `{"op": "slice", "start": 0, "end": 1000}` |
| `lines` | Get line range | `{"op": "lines", "start": 0, "end": 50}` |
| `find` | Find text occurrences | `{"op": "find", "text": "ERROR"}` |
| `regex` | Regex pattern search | `{"op": "regex", "pattern": "def \\w+"}` |
| `count` | Count lines/chars/matches | `{"op": "count", "what": "lines"}` |
| `llm_query` | Ask a helper AI | `{"op": "llm_query", "prompt": "Summarize this"}` |
| `final` | Return the answer | `{"op": "final", "answer": "..."}` |

## The Sub-LLM Pattern

For semantic analysis, the root LLM can delegate to helper LLMs:

```json
{
  "op": "llm_query",
  "prompt": "Extract all function names from this code: ${chunk}",
  "on": "code_section"
}
```

This is powerful because:
- Helper LLMs can run on cheaper/faster hardware
- Multiple helpers can work in parallel
- Root LLM orchestrates, helpers do grunt work

## Variables and State

Commands can store results in variables:

```json
{"op": "find", "text": "ERROR", "store": "error_positions"}
```

Later commands can reference them:

```json
{"op": "count", "what": "matches", "on": "error_positions"}
```

This lets the AI build up understanding incrementally.

## Next Steps

- [Getting Started](getting-started.md) - Try it yourself
- [Commands Reference](../commands.md) - Full command documentation
