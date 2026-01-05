# Recursive Language Models (RLM) - Explain Like I'm 5

## The Cookie Jar Problem 🍪

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

### 1. 📦 The Context Box
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

### 2. 💻 The Code Superpower
The AI can write Python code to search, filter, and organize the text - like having a robot assistant!

```python
# AI writes this code itself!
for doc in documents:
    if "chocolate" in doc:
        interesting_docs.append(doc)
```

### 3. 🤖 The Helper AI Phone
The main AI can call helper AIs to analyze each piece. Like having friends who can each read one chapter of a huge book!

```
Main AI: "Hey helper, what's in chapter 5?"
Helper AI: "It talks about dragons and a magic sword!"
Main AI: "Thanks! *writes that down* Now checking chapter 6..."
```

---

## Real Example: Finding Needles in Haystacks

**Task:** Find who won the beauty pageant in a 10-million-word document collection.

### Without RLM (Regular AI):
```
AI: *tries to read 10 million words*
AI: *brain melts* 
AI: "Uhh... I think maybe... Susan? No wait... I forgot..."
```

### With RLM:
```
Step 1: AI writes code to list all documents
        "Okay, I have 1000 documents totaling 10M words"

Step 2: AI searches for keywords
        documents = grep("beauty pageant", all_docs)
        "Found 5 documents mentioning beauty pageant!"

Step 3: AI asks helper to check each one
        for doc in documents:
            answer = helper_ai("Who won the pageant?", doc)
            results.append(answer)
        
Step 4: AI combines results
        "Based on 3 matching answers: Maria Dalmacio won!"
```

---

## Mike's Implementation Options

Here are the ways you could build RLM with your setup:

### Option A: Custom Rust Orchestrator 🦀

**What it is:** A Rust program that coordinates everything - loads documents, runs a Python REPL, and calls your LLMs.

```
┌─────────────────────────────────────────────────────────┐
│                 RUST RLM ORCHESTRATOR                    │
│                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │ Context  │    │  Python  │    │    LLM Pool      │  │
│  │  Store   │◄──►│   REPL   │◄──►│                  │  │
│  │(HashMap) │    │  (PyO3)  │    │ Ollama (M40s)    │  │
│  └──────────┘    └──────────┘    │ Ollama (RTX)     │  │
│                                   │ Ollama (P100s)   │  │
│                                   │ DeepSeek API     │  │
│                                   │ Claude API       │  │
│                                   └──────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

| Pros | Cons |
|------|------|
| ✅ Full control over everything | ❌ Most development work |
| ✅ Optimal for your GPU cluster | ❌ Need to maintain it yourself |
| ✅ Can load-balance across servers | ❌ ~2-4 weeks to build properly |
| ✅ Native performance | ❌ Python REPL integration adds complexity |
| ✅ Your preferred language! | |

**Best for:** Production use, processing huge documents regularly, when you need to squeeze every bit of performance from your hardware.

---

### Option B: OpenCode + DeepSeek API 🔷

**What it is:** Use Z.ai's opencode CLI with DeepSeek as the backend, wrapped with RLM capabilities.

```
┌─────────────────────────────────────────────────────────┐
│                    OPENCODE + WRAPPER                    │
│                                                          │
│   User Query                                             │
│       │                                                  │
│       ▼                                                  │
│   ┌─────────────────┐                                   │
│   │  rlm-wrapper.sh │  ◄── Injects RLM system prompt    │
│   └────────┬────────┘                                   │
│            │                                             │
│            ▼                                             │
│   ┌─────────────────┐    ┌──────────────────┐          │
│   │    opencode     │───►│   DeepSeek API   │          │
│   │  (code executor)│    │  (deepseek-chat) │          │
│   └─────────────────┘    └──────────────────┘          │
│            │                                             │
│            ▼                                             │
│   ┌─────────────────┐                                   │
│   │  Python REPL    │  ◄── llm_query() calls back to   │
│   │  + llm_query()  │      DeepSeek or local Ollama    │
│   └─────────────────┘                                   │
└─────────────────────────────────────────────────────────┘
```

| Pros | Cons |
|------|------|
| ✅ Fast to set up (~1 day) | ❌ DeepSeek API costs money |
| ✅ DeepSeek is very capable | ❌ Less control than custom solution |
| ✅ OpenCode handles code execution | ❌ Depends on external API availability |
| ✅ Can fall back to local Ollama | ❌ OpenCode still maturing |
| ✅ GLM-4 support too | |

**Best for:** Quick experiments, when you want good results fast, hybrid cloud/local setup.

---

### Option C: Pure Ollama + Python Script 🦙

**What it is:** A standalone Python script that implements RLM using only your local Ollama servers.

```
┌─────────────────────────────────────────────────────────┐
│                  PYTHON RLM SCRIPT                       │
│                                                          │
│   ┌─────────────────────────────────────────────────┐   │
│   │                  rlm.py                          │   │
│   │                                                  │   │
│   │  context_store = {"context": big_document}      │   │
│   │                                                  │   │
│   │  while not done:                                 │   │
│   │      code = ask_root_llm("what next?")          │   │
│   │      output = exec(code)  # runs in REPL        │   │
│   │      if "FINAL" in output:                       │   │
│   │          done = True                             │   │
│   └─────────────────────────────────────────────────┘   │
│                          │                               │
│                          ▼                               │
│   ┌─────────────────────────────────────────────────┐   │
│   │              YOUR OLLAMA SERVERS                 │   │
│   │                                                  │   │
│   │  Server 1 (M40 24GB)     Server 2 (RTX)         │   │
│   │  └─ qwen2.5-coder:32b    └─ llama3.3:70b        │   │
│   │                                                  │   │
│   │  Server 3 (P100s)                                │   │
│   │  └─ deepseek-coder:33b                           │   │
│   └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

| Pros | Cons |
|------|------|
| ✅ Completely free (your hardware) | ❌ Slower than cloud APIs |
| ✅ Works offline | ❌ Limited by your GPU VRAM |
| ✅ Simple to understand & modify | ❌ No fancy load balancing built-in |
| ✅ Great for learning/experimenting | ❌ Python, not Rust 😉 |
| ✅ $0.45/kWh + solar = very cheap | |

**Best for:** Privacy-sensitive work, learning how RLM works, when internet is unreliable, cost optimization.

---

### Option D: Claude Code CLI + MCP Server 🔌

**What it is:** Extend Claude Code with custom MCP tools that provide RLM capabilities.

```
┌─────────────────────────────────────────────────────────┐
│                CLAUDE CODE + MCP RLM                     │
│                                                          │
│   ┌─────────────────┐                                   │
│   │   Claude Code   │                                   │
│   │   CLI / Web     │                                   │
│   └────────┬────────┘                                   │
│            │ MCP Protocol                                │
│            ▼                                             │
│   ┌─────────────────────────────────────────────────┐   │
│   │            MCP RLM Server                        │   │
│   │                                                  │   │
│   │  Tools:                                          │   │
│   │  • load_context(name, content)                  │   │
│   │  • peek_context(name, start, end)               │   │
│   │  • context_info(name)                           │   │
│   │  • llm_subquery(prompt, provider, model)        │   │
│   │  • execute_code(code)                           │   │
│   └─────────────────────────────────────────────────┘   │
│                          │                               │
│            ┌─────────────┴─────────────┐                │
│            ▼                           ▼                │
│   ┌─────────────────┐       ┌─────────────────┐        │
│   │  Ollama Servers │       │   Claude API    │        │
│   │  (sub-queries)  │       │  (sub-queries)  │        │
│   └─────────────────┘       └─────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

| Pros | Cons |
|------|------|
| ✅ Integrates with existing Claude workflow | ❌ Depends on Claude Code availability |
| ✅ MCP is extensible standard | ❌ Two AI layers (Claude + sub-LLM) |
| ✅ Can use Claude's strong reasoning | ❌ Costs money (Claude API) |
| ✅ Easy to add more tools later | ❌ MCP server needs to stay running |
| ✅ Hybrid local/cloud naturally | |

**Best for:** When you're already using Claude Code, want best-of-both-worlds, professional work.

---

### Option E: Hybrid Rust + Emacs Integration 🚀

**What it is:** Rust daemon with elisp bindings for Emacs integration.

```
┌─────────────────────────────────────────────────────────┐
│                EMACS + RUST RLM DAEMON                   │
│                                                          │
│   ┌─────────────────────────────────────────────────┐   │
│   │                    EMACS                         │   │
│   │                                                  │   │
│   │  (rlm-query "Find all TODO items"               │   │
│   │             (buffer-string))                     │   │
│   │                                                  │   │
│   │  ;; Communicates via JSON-RPC or socket         │   │
│   └─────────────────────────────────────────────────┘   │
│                          │                               │
│                          ▼                               │
│   ┌─────────────────────────────────────────────────┐   │
│   │              RUST RLM DAEMON                     │   │
│   │                                                  │   │
│   │  • Runs as background service                   │   │
│   │  • Manages context across sessions              │   │
│   │  • Load balances across GPU servers             │   │
│   │  • Caches frequent queries                      │   │
│   └─────────────────────────────────────────────────┘   │
│                          │                               │
│            ┌─────────────┴─────────────┐                │
│            ▼                           ▼                │
│      Local Ollama              Cloud APIs               │
│      (your GPUs)           (fallback/overflow)          │
└─────────────────────────────────────────────────────────┘
```

| Pros | Cons |
|------|------|
| ✅ Native Emacs integration | ❌ Most complex to build |
| ✅ Persistent daemon = fast startup | ❌ Need elisp + Rust expertise |
| ✅ Perfect for your workflow | ❌ 4-6 weeks development |
| ✅ Can integrate with org-mode, magit | ❌ Niche (just for you!) |
| ✅ Full Rust performance | |

**Best for:** Ultimate integration with your Emacs workflow, long-term investment.

---

## Quick Comparison Chart

| Option | Setup Time | Cost | Performance | Control | Your Match |
|--------|------------|------|-------------|---------|------------|
| A. Rust Custom | 2-4 weeks | $0* | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| B. OpenCode+DeepSeek | 1 day | $$ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| C. Python+Ollama | 2-3 days | $0* | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| D. Claude+MCP | 1-2 days | $$$ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| E. Emacs+Rust | 4-6 weeks | $0* | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

*$0 = just electricity for your servers

---

## The Secret Sauce: Why RLM Works

### Traditional LLM: "Context Rot" 🤢
```
Input size:    [============================] 10M tokens
Model window:  [======]                        272K tokens
Result:        🤯 "I... forgot... what was the question?"
```

### RLM: "Divide and Conquer" 💪
```
Input size:    [============================] 10M tokens
                  ↓ chunk ↓ chunk ↓ chunk
Sub-queries:   [==] → answer1
               [==] → answer2  
               [==] → answer3
                  ↓ combine
Final:         🎯 Accurate answer from all pieces!
```

The paper shows RLM achieves:
- **91.33%** accuracy on BrowseComp+ (vs 0% for base model that couldn't fit context!)
- **58%** F1 on OOLONG-Pairs (vs 0.04% for GPT-5 base)
- Handles **10M+ tokens** effectively

---

## Getting Started: Recommended Path

Based on your setup (Arch Linux, distributed GPUs, Rust preference, Emacs user):

### Week 1: Quick Win
Start with **Option C (Python + Ollama)** to understand how RLM works.

### Week 2-3: Production Path
Build **Option A (Rust Orchestrator)** with your learnings.

### Week 4+: Integration
Add **Option E (Emacs bindings)** for daily workflow integration.

### Parallel Track
Set up **Option D (MCP Server)** for when you're using Claude Code anyway.

---

## One More Analogy: The Library Research Assistant 📚

**You:** "Find everything about quantum computing in this library."

**Regular AI (tries to read entire library):**
*head explodes* 📚💥🤯

**RLM AI:**
1. "Let me check the card catalog first..." *(probes structure)*
2. "Physics section, rows 12-15 look relevant..." *(filters)*
3. "Hey assistant, summarize book 12A" *(sub-query)*
4. "Hey assistant, summarize book 12B" *(sub-query)*
5. "Hey assistant, summarize book 12C" *(sub-query)*
6. "Combining all summaries... here's your answer!" *(aggregate)*

**Result:** Accurate, comprehensive, and didn't need to read every cookbook in the library! 🎉

---

## TL;DR

1. **RLM = Let AI peek at big data in pieces + call helper AIs**
2. **Your best options:** Rust orchestrator (production) or Python+Ollama (learning)
3. **Why it works:** Avoids "context rot" by dividing and conquering
4. **Key insight:** The prompt is a variable, not input - the AI manipulates it with code

Now go build something cool! 🚀
