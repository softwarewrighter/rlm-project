# RLM Visualizer

The RLM Visualizer is an interactive web interface for running and understanding RLM queries.

## Accessing the Visualizer

Start the RLM server and navigate to:

```
http://localhost:8080/visualize
```

## Interface Overview

```
┌────────────────────────────────────────────────────────────┐
│  RLM Orchestrator Visualizer                               │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Query: [________________________] [Run]                   │
│                                                            │
│  Context:                                                  │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ (paste your text here)                               │ │
│  │                                                      │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
├────────────────────────────────────────────────────────────┤
│  Stats: Iterations: 5 | Sub-calls: 2 | Context: 1.2KB     │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Flow:  [Query] ──► [Step 1] ──► [Step 2] ──► [Answer]    │
│                                                            │
├────────────────────────────────────────────────────────────┤
│  Timeline:                                                 │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐            │
│  │  1   │ │  2   │ │  3   │ │  4   │ │  5   │            │
│  │ find │ │count │ │lines │ │query │ │final │            │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘            │
│                                                            │
├────────────────────────────────────────────────────────────┤
│  Details:                                                  │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Command: {"op": "find", "text": "ERROR"}             │ │
│  │ Output: Found 3 occurrences                          │ │
│  │ Variables: errors = "15\n64\n210"                    │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## Components

### Query Input
Enter your question about the context. Be specific:
- "How many ERROR lines are there?"
- "Find all function definitions starting with 'handle_'"
- "Summarize the key points in this document"

### Context Area
Paste the text you want to analyze. This can be:
- Log files
- Source code
- Documents
- Any text content

The context size is shown in the stats bar.

### Stats Bar
Shows real-time metrics:
- **Iterations** - How many command loops executed
- **Sub-calls** - Number of `llm_query` delegations to helper LLMs
- **Context** - Size of the input context

### Flow Diagram
A visual representation of the query pipeline:
```
[Query] ──► [Step 1] ──► [Step 2] ──► ... ──► [Answer]
```

Each box represents a significant step in the reasoning process.

### Timeline
Clickable iteration history. Each tile shows:
- Iteration number
- Primary operation (find, count, regex, llm_query, final)

Click any tile to see its details.

### Detail Panel
When you click a timeline tile, this shows:
- **Raw Command** - The JSON the LLM issued
- **Output** - What the executor returned
- **Variables** - State after this iteration
- **Errors** - Any issues encountered

## Usage Examples

### Example 1: Log Analysis

**Query:** "Count error types in this log"

**Context:**
```
2024-01-01 10:00:00 ERROR connection refused
2024-01-01 10:00:01 INFO server starting
2024-01-01 10:00:02 ERROR timeout
2024-01-01 10:00:03 ERROR connection refused
2024-01-01 10:00:04 WARN low memory
```

**Result:** The visualizer shows the AI:
1. Finding all ERROR lines
2. Using llm_query to categorize each error
3. Aggregating results into a summary

### Example 2: Code Analysis

**Query:** "List all functions that take more than 2 parameters"

**Context:** (paste source code)

**Result:** The AI:
1. Uses regex to find function definitions
2. Filters by parameter count
3. Returns a formatted list

## Tips

### Performance
- **DeepSeek root LLM** is recommended for speed
- **Local Ollama** works but is slower
- Large contexts take more iterations

### Debugging
- Watch the timeline to see where the AI gets stuck
- Check the detail panel for error messages
- If iterations max out, try a more specific query

### Best Practices
- Start with simple queries to understand the flow
- Build up to complex multi-step analysis
- Use the detail panel to learn what commands work well

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+Enter` | Run query |
| `Escape` | Clear selection |

## API Alternative

For programmatic access, use the `/debug` endpoint:

```bash
curl -X POST http://localhost:8080/debug \
  -H "Content-Type: application/json" \
  -d '{"query": "...", "context": "..."}'
```

Returns full iteration history in JSON format.
