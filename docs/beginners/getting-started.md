# Getting Started with RLM

This guide will have you running RLM queries in minutes.

## Prerequisites

- An LLM provider (DeepSeek API key or Ollama installed locally)
- Rust toolchain (for the server) or Python 3.8+ (for the CLI)

## Option 1: Rust Server (Recommended)

### Step 1: Configure Providers

Edit `rlm-orchestrator/config.toml`:

```toml
max_iterations = 20
max_sub_calls = 50

# Root LLM - the orchestrator (needs to be smart)
[[providers]]
provider_type = "deepseek"
model = "deepseek-chat"
role = "root"

# Sub LLM - for llm_query calls (can be smaller/local)
[[providers]]
provider_type = "ollama"
base_url = "http://localhost:11434"
model = "qwen2.5-coder:14b"
role = "sub"
```

### Step 2: Set Your API Key

```bash
export DEEPSEEK_API_KEY="your-key-here"
```

### Step 3: Start the Server

```bash
cd rlm-orchestrator
cargo run --bin rlm-server
```

You should see:
```
RLM Orchestrator starting...
  Root LLM: deepseek-chat (deepseek)
  Sub LLMs: qwen2.5-coder:14b (ollama)
Listening on 0.0.0.0:8080
```

### Step 4: Try a Query

**Using curl:**
```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How many ERROR lines?",
    "context": "Line 1: OK\nLine 2: ERROR connection failed\nLine 3: OK\nLine 4: ERROR timeout\nLine 5: OK"
  }'
```

**Using the visualizer:**
Open http://localhost:8080/visualize in your browser for an interactive UI.

## Option 2: Python CLI

### Step 1: Install Dependencies

```bash
pip install httpx rich typer pydantic
```

### Step 2: Run a Query

```bash
# With Ollama (local)
python src/rlm.py \
  --query "Find all function definitions" \
  --context-file ./your-code.py

# With DeepSeek (API)
export DEEPSEEK_API_KEY="your-key"
python src/rlm.py \
  --query "Summarize the main points" \
  --context-file ./document.txt \
  --provider deepseek
```

## Understanding the Output

When you run a query, you'll see the AI iterating:

```
Iteration 1: {"op": "find", "text": "ERROR", "store": "errors"}
  → Found 2 occurrences

Iteration 2: {"op": "count", "what": "matches", "on": "errors"}
  → Counted 2 lines

Iteration 3: {"op": "final", "answer": "There are 2 ERROR lines"}

Final Answer: There are 2 ERROR lines
```

## Visualizer Guide

The web visualizer at `/visualize` provides:

1. **Input Area** - Enter your query and context
2. **Run Button** - Executes the RLM loop
3. **Stats Bar** - Shows iterations, sub-calls, context size
4. **Flow Diagram** - Visual Query → Steps → Answer
5. **Timeline** - Clickable iteration history
6. **Detail Panel** - See commands and outputs for each step

## Tips for Good Queries

**Be specific:**
- "Count the number of functions that return a string" (specific)
- "What's in this code?" (too vague)

**Reference the context:**
- "Find all ERROR lines in the log" (references context)
- "Tell me about errors" (vague)

**Break down complex questions:**
- First ask for structure, then dig into specifics

## Troubleshooting

**"Connection refused" on Ollama:**
- Ensure Ollama is running: `ollama serve`
- Check the base_url in config.toml

**Slow responses:**
- Local models are slower than APIs
- Consider using DeepSeek for root LLM

**"Max iterations reached":**
- The AI is stuck in a loop
- Try a more specific query
- Check if the context contains what you're looking for

## Next Steps

- [Visualizer Guide](../visualizer.md) - Master the web UI
- [Commands Reference](../commands.md) - All available commands
- [Configuration](../config.md) - Advanced config options
