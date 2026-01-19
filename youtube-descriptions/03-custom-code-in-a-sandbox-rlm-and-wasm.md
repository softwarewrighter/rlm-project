# Custom Code in a Sandbox? RLM and WASM

**Video URL:** https://www.youtube.com/watch?v=jMo5AaMRUkM
**Duration:** 7:45

---

What happens when built-in DSL commands aren't enough? Let the LLM write custom code. This video explores RLM's two-level architecture: L1 DSL for simple operations and L2 WASM for sandboxed Rust code execution. Includes live log analysis demos showing both levels in action.

Timestamps:
0:00 - Title
0:05 - Hook
0:11 - Problem Statement
0:26 - Big Idea: Expand Workspace
0:55 - Level 1: DSL Overview
1:17 - Level 2: WASM Introduction
1:51 - Architecture Diagram
2:17 - Visualizer Demo: Count Lines
3:05 - Visualizer Demo: Find Secret
4:00 - Log Analysis Demo Setup
4:25 - Log Demo: Top Error Types
5:17 - Log Demo: Busiest Hour
6:04 - Log Demo: Errors Per Day
6:53 - Recap and CTA

⚠️ AUDIO NOTE: I am aware of the audio problems; please turn on captions to help clarify what is being said.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MORE IN THIS SERIES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
▶ Recursive Language Model implemented, evaluated, explained
   https://www.youtube.com/watch?v=5DhaTPuyhys

▶ Can AI Find a Secret Hidden in War and Peace?
   https://www.youtube.com/watch?v=d5gaL4iOdLA

▶ Why I Let an LLM Compile Native Binaries
   https://www.youtube.com/watch?v=oN6XyZdEHqY

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BACKGROUND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This is my Rust implementation of RLM, not the Python implementation by the paper's authors. I built this from scratch using Rust, DeepSeek via LiteLLM, and vibe coding with Claude.

CAPABILITY LEVELS (Roadmap):
• L1 (DSL): Built-in commands for text operations (find, regex, count, filter, extract)
• L2 (WASM): LLM generates Rust code → compiled to WebAssembly sandbox
• L3 (CLI): LLM generates Rust code → compiled to native binary for large datasets
• L4 (LLM): Recursive delegation - LLM delegates chunks to sub-LLMs for semantic analysis

LINKS:
📄 Paper: https://arxiv.org/abs/2512.24601
💻 Code: https://github.com/softwarewrighter/rlm-project

#RLM #RecursiveLanguageModel #Rust #VibeCoding #Deepseek #LiteLLM #WASM #WebAssembly #LLM #AI #MachineLearning #CodeGeneration #DataAnalysis #LogAnalysis #Sandbox #Programming #SoftwareEngineering #DevTools #OpenSource #AITools #TechDemo
