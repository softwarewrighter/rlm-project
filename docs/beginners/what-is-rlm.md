# What is RLM?

**Recursive Language Models (RLM)** is an inference strategy that lets AI process documents **far larger than its context window** by working through them piece by piece.

## The Problem

Regular LLMs have a "context window" - the maximum amount of text they can consider at once. Even the largest models max out around 200K-2M tokens. But real-world tasks often involve:

- Entire codebases (millions of lines)
- Document collections (thousands of files)
- Log files (gigabytes of data)

When you force too much into the context window, the AI "forgets" earlier content and produces inaccurate answers.

## The Solution

RLM solves this by giving the AI **tools to explore** instead of forcing everything into its brain:

```
┌─────────────────────────────────────────┐
│             CONTEXT BOX                 │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
│  │Part1│ │Part2│ │Part3│ │ ... │       │
│  └─────┘ └─────┘ └─────┘ └─────┘       │
│                                         │
│  AI: "Let me peek at the parts I need"  │
└─────────────────────────────────────────┘
```

### Current Implementation (v0.1)
The AI issues structured JSON commands:
- **Slice/Lines** - Look at specific portions of the text
- **Find/Regex** - Search for text patterns
- **Count** - Get statistics about the content
- **LLM Query** - Ask helper AIs to analyze chunks

### Roadmap (v0.2)
The AI will generate **WebAssembly (WASM)** modules for complex analysis - enabling loops, conditionals, and custom logic while remaining sandboxed and safe.

## A Simple Analogy

**Regular AI:** Tries to read an entire library at once, brain explodes.

**RLM AI:**
1. Checks the card catalog
2. Finds relevant sections
3. Asks assistants to summarize each book
4. Combines results into a coherent answer

## Why It Matters

The paper ["Recursive Language Models"](https://arxiv.org/html/2512.24601v1) shows RLM can:

- Process inputs **2 orders of magnitude** beyond normal limits
- Achieve **91%** accuracy on tasks where base models score **0%** (couldn't fit context)
- Handle **10+ million tokens** effectively

## Next Steps

- [How It Works](how-it-works.md) - See the iteration loop in action
- [Getting Started](getting-started.md) - Run your first RLM query
