# Why I Let an LLM Compile Native Binaries

**Video URL:** https://www.youtube.com/watch?v=oN6XyZdEHqY
**Duration:** 5:51

---

My RLM implementation "Level 3" breaks free from the WASM sandbox. When your data outgrows WebAssembly limits, native Rust CLI tools process thousands of lines without timeouts.

This video demonstrates four CLI demos in action: error ranking, unique IP analysis, response time percentiles, and word frequency analysis.

Timestamps:
0:00 - Title
0:05 - Hook
0:19 - RLM Paper Overview
0:35 - Level 3 Introduction
0:55 - VHS Demo: Error Ranking
1:28 - Percentiles Intro
1:47 - VHS Demo: Percentiles
2:25 - Visualizer Introduction
2:44 - CLI Demo 1: Error Ranking
3:17 - CLI Demo 2: Unique IPs
3:55 - CLI Demo 3: Percentiles
4:35 - CLI Demo 4: Word Frequency
5:16 - Call to Action

⚠️ AUDIO NOTE: I am aware of the audio problems; please turn on captions to help clarify what is being said.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MORE IN THIS SERIES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
▶ Recursive Language Model implemented, evaluated, explained
   https://www.youtube.com/watch?v=5DhaTPuyhys

▶ Can AI Find a Secret Hidden in War and Peace?
   https://www.youtube.com/watch?v=d5gaL4iOdLA

▶ Custom Code in a Sandbox? RLM and WASM
   https://www.youtube.com/watch?v=jMo5AaMRUkM

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

#RLM #RecursiveLanguageModel #Rust #VibeCoding #Deepseek #LiteLLM #CLI #NativeCode #LLM #AI #MachineLearning #CodeGeneration #DataAnalysis #WASM #WebAssembly #Programming #SoftwareEngineering #DevTools #OpenSource #AITools #TechDemo
