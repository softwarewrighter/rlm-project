# Recursive Language Models (RLM) - Explain Like I'm 5

## The Cookie Jar Problem

Imagine you have a **really, really big cookie jar** - so big you can't see all the cookies at once. You want to find all the chocolate chip cookies.

**Normal way (regular LLM):** Try to dump ALL the cookies on the table at once. But your table is too small! Cookies fall off, you lose track, and you miss some chocolate chips.

**Smart way (RLM):**
1. Look at a handful of cookies at a time
2. Ask your friend to check each handful: "Any chocolate chips here?"
3. Keep track of what your friend finds
4. When done, add up all the chocolate chips!

That's RLM! Instead of forcing everything into the AI's brain at once (where it gets confused), we let the AI **look at pieces** and **ask helper AIs** about each piece.

---

## The Three Magic Powers of RLM

### 1. The Context Box
Instead of eating all the text, the AI puts it in a box and looks at it piece by piece.

```
┌─────────────────────────────────────┐
│  CONTEXT BOX                        │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   │
│  │Doc 1│ │Doc 2│ │Doc 3│ │ ... │   │
│  └─────┘ └─────┘ └─────┘ └─────┘   │
│  "I can peek at any piece I want!"  │
└─────────────────────────────────────┘
```

### 2. The Command Toolkit
The AI issues commands to search, slice, and analyze the context:

```json
{"op": "find", "text": "ERROR", "store": "errors"}
{"op": "count", "what": "matches", "on": "errors"}
{"op": "final", "answer": "Found 3 errors"}
```

### 3. The Helper AI Phone
The main AI can call helper AIs to analyze each piece:

```
Main AI: "Hey helper, what's in this chunk?"
Helper AI: "It talks about dragons and a magic sword!"
Main AI: "Thanks! *writes that down* Now checking next chunk..."
```

---

## How Our Implementation Works

### Current: Structured Commands (v0.1)

The AI issues JSON commands that the orchestrator executes:

```
┌─────────────┐
│   Query +   │
│   Context   │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│  Root LLM   │────►│   Command   │
│  (thinks)   │◄────│   Executor  │
└─────────────┘     └─────────────┘
       │                   │
       │  (iterate)        ▼
       │            ┌─────────────┐
       │            │  Sub-LLM    │
       │            │  (helpers)  │
       │            └─────────────┘
       ▼
┌─────────────┐
│   Answer    │
└─────────────┘
```

**Available commands:**
- `slice` / `lines` - Extract portions of text
- `find` / `regex` - Search for patterns
- `count` - Get statistics
- `llm_query` - Ask a helper AI
- `final` - Return the answer

**Limitations:**
- Fixed command set (no loops, conditionals)
- LLM must use multiple iterations for complex logic
- No arbitrary computation

### Roadmap: WASM Dynamic Code (v0.2)

The AI will generate WebAssembly modules for complex analysis:

```
┌─────────────┐
│  Root LLM   │
│  generates  │
│    WASM     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│         WASM Sandbox                │
│                                     │
│  • Loops and conditionals           │
│  • Custom analysis functions        │
│  • Memory-safe execution            │
│  • No filesystem/network access     │
│                                     │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│   Results   │
└─────────────┘
```

**Why WASM?**
- **Safe**: Sandboxed execution, no escape to host system
- **Fast**: Near-native performance
- **Portable**: Runs anywhere without Python/Node
- **Composable**: AI can write real programs, not just commands

---

## Real Example: Finding Needles in Haystacks

**Task:** Count ERROR lines in a 10,000 line log file.

### Without RLM (Regular AI):
```
AI: *tries to read 10,000 lines*
AI: *context window exceeded*
AI: "I can only see the last 2000 lines..."
```

### With RLM (Current Implementation):
```
Iteration 1: {"op": "find", "text": "ERROR", "store": "errors"}
             → Found 47 occurrences

Iteration 2: {"op": "count", "what": "matches", "on": "errors"}
             → Counted 47 lines

Iteration 3: {"op": "final", "answer": "There are 47 ERROR lines"}
```

### With RLM + WASM (Future):
```
AI generates WASM module:

fn analyze(context: &str) -> String {
    let errors: Vec<&str> = context
        .lines()
        .filter(|l| l.contains("ERROR"))
        .collect();

    let by_type = errors
        .iter()
        .map(|l| extract_error_type(l))
        .fold(HashMap::new(), |mut m, t| {
            *m.entry(t).or_insert(0) += 1;
            m
        });

    format!("Found {} errors: {:?}", errors.len(), by_type)
}

→ "Found 47 errors: {timeout: 23, connection: 15, auth: 9}"
```

---

## Architecture Overview

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│   Client    │────▶│  RLM Server     │────▶│  Root LLM   │
│  /visualize │     │  (Rust/Axum)    │     │  (DeepSeek) │
└─────────────┘     └────────┬────────┘     └─────────────┘
                             │
                    ┌────────▼────────┐
                    │ Command Executor │
                    │  + WASM Runtime  │  ◄── Future
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Ollama   │  │ Ollama   │  │ DeepSeek │
        │ (local)  │  │ (remote) │  │  (API)   │
        └──────────┘  └──────────┘  └──────────┘
              Sub-LM Pool (for llm_query)
```

---

## The Secret Sauce: Why RLM Works

### Traditional LLM: "Context Rot"
```
Input size:    [============================] 10M tokens
Model window:  [======]                        200K tokens
Result:        "I... forgot... what was the question?"
```

### RLM: "Divide and Conquer"
```
Input size:    [============================] 10M tokens
                  ↓ chunk ↓ chunk ↓ chunk
Sub-queries:   [==] → answer1
               [==] → answer2
               [==] → answer3
                  ↓ combine
Final:         Accurate answer from all pieces!
```

The paper shows RLM achieves:
- **91%** accuracy on tasks where base models score **0%** (context too large)
- **58%** F1 on OOLONG-Pairs (vs 0.04% for GPT-5 base)
- Handles **10M+ tokens** effectively

---

## Getting Started

### Quick Start (5 minutes)
```bash
cd rlm-orchestrator
export DEEPSEEK_API_KEY="your-key"
cargo run --bin rlm-server

# Open visualizer
open http://localhost:8080/visualize
```

### Try a Query
```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How many ERROR lines?",
    "context": "OK\nERROR timeout\nOK\nERROR connection\nOK"
  }'
```

---

## Measuring Success

We know RLM is working when:

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Accuracy** | >80% on NIAH | Test with known-answer datasets |
| **Scale** | 100K+ tokens | Process contexts larger than model window |
| **Efficiency** | <10 iterations avg | Track iterations per query |
| **Speed** | <30s per query | Benchmark response times |

---

## TL;DR

1. **RLM = AI + tools to explore large contexts piece by piece**
2. **Current:** Structured JSON commands (safe, simple)
3. **Future:** WASM code generation (powerful, still safe)
4. **Why it works:** Avoids "context rot" by dividing and conquering
5. **Key insight:** The context is a database the AI queries, not input it memorizes

---

## References

- [Recursive Language Models Paper](https://arxiv.org/html/2512.24601v1) - MIT CSAIL
- [Project README](../README.md) - Quick start guide
- [Architecture Docs](architecture.md) - Technical details
