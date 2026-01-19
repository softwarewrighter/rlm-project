# Recursive Language Model implemented, evaluated, explained

**Video URL:** https://www.youtube.com/watch?v=5DhaTPuyhys
**Duration:** 3:35

---

How do you process data larger than an LLM's context window? Instead of expanding context, expand the workspace. This video explains the Recursive Language Model (RLM) technique using a "cookie jar" analogy and shows benchmark results: 87-91% accuracy on 9 SCROLLS tests using only ~3000 tokens per iteration.

⚠️ AUDIO NOTE: I am aware of the audio problems; please turn on captions to help clarify what is being said.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MORE IN THIS SERIES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
▶ Can AI Find a Secret Hidden in War and Peace?
   https://www.youtube.com/watch?v=d5gaL4iOdLA

▶ Custom Code in a Sandbox? RLM and WASM
   https://www.youtube.com/watch?v=jMo5AaMRUkM

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

#RLM #RecursiveLanguageModel #Rust #VibeCoding #Deepseek #LiteLLM #LLM #AI #MachineLearning #SCROLLS #Benchmark #ContextWindow #NLP #Programming #SoftwareEngineering #OpenSource #AITools #TechDemo
