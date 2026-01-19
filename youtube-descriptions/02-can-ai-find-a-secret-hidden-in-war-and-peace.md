# Can AI Find a Secret Hidden in War and Peace?

**Video URL:** https://www.youtube.com/watch?v=d5gaL4iOdLA
**Duration:** 1:39

---

I hid a secret message in the full text of War and Peace (3.2MB, 580,000+ words) and challenged my RLM implementation to find it. The LLM located the secret in just 2 iterations using only ~3000 tokens total—without ever seeing the full novel in context.

⚠️ AUDIO NOTE: I am aware of the audio problems; please turn on captions to help clarify what is being said.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MORE IN THIS SERIES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
▶ Recursive Language Model implemented, evaluated, explained
   https://www.youtube.com/watch?v=5DhaTPuyhys

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

#RLM #RecursiveLanguageModel #Rust #VibeCoding #Deepseek #LiteLLM #LLM #AI #MachineLearning #WarAndPeace #Tolstoy #Demo #ContextWindow #NLP #Programming #SoftwareEngineering #OpenSource #AITools #TechDemo
