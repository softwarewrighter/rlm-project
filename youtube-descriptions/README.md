# YouTube Video Descriptions

Standardized descriptions for the RLM (Recursive Language Model) video series.

## Playlist

**RLM Implementation Series:** https://www.youtube.com/playlist?list=PLKjvVAEaR4itAgRBOJGi-B2CCY2-Wvgem

## Videos

| # | Title | Duration | File |
|---|-------|----------|------|
| 1 | Recursive Language Model implemented, evaluated, explained | 3:35 | [01-recursive-language-model-implemented-evaluated-explained.md](01-recursive-language-model-implemented-evaluated-explained.md) |
| 2 | Can AI Find a Secret Hidden in War and Peace? | 1:39 | [02-can-ai-find-a-secret-hidden-in-war-and-peace.md](02-can-ai-find-a-secret-hidden-in-war-and-peace.md) |
| 3 | Custom Code in a Sandbox? RLM and WASM | 7:45 | [03-custom-code-in-a-sandbox-rlm-and-wasm.md](03-custom-code-in-a-sandbox-rlm-and-wasm.md) |
| 4 | Why I Let an LLM Compile Native Binaries | 5:51 | [04-why-i-let-an-llm-compile-native-binaries.md](04-why-i-let-an-llm-compile-native-binaries.md) |

## Standard Elements

Each description includes:

1. **Video-specific intro** - Unique content at the top describing that video
2. **Timestamps** - Where applicable
3. **Audio disclaimer** - Captions recommendation
4. **Series links** - Links to all other videos in the series
5. **Background section** - Common info repeated in all videos:
   - Rust implementation clarification (not the Python version by paper authors)
   - Capability levels roadmap (L1-L4)
   - Paper and repo links

## Required Hashtags

All videos include these core hashtags:
- `#RLM`
- `#RecursiveLanguageModel`
- `#Rust`
- `#VibeCoding`
- `#Deepseek`
- `#LiteLLM`

Plus video-specific hashtags relevant to each topic.

## Links

- **Paper:** https://arxiv.org/abs/2512.24601
- **Code:** https://github.com/softwarewrighter/rlm-project

## Capability Levels

| Level | Name | Description |
|-------|------|-------------|
| L1 | DSL | Built-in commands for text operations (find, regex, count, filter, extract) |
| L2 | WASM | LLM generates Rust code → compiled to WebAssembly sandbox |
| L3 | CLI | LLM generates Rust code → compiled to native binary for large datasets |
| L4 | LLM | Recursive delegation - LLM delegates chunks to sub-LLMs for semantic analysis |
