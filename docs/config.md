# RLM Configuration

Configuration is stored in `rlm-orchestrator/config.toml`.

## Example Configuration

```toml
# Execution limits
max_iterations = 20
max_sub_calls = 50
output_limit = 10000

# Root LLM (orchestrates the RLM loop)
[[providers]]
provider_type = "deepseek"
base_url = "https://api.deepseek.com"
model = "deepseek-chat"
role = "root"
weight = 1

# Sub LLM (handles llm_query calls)
[[providers]]
provider_type = "ollama"
base_url = "http://localhost:11434"
model = "qwen2.5-coder:14b"
role = "sub"
weight = 1

# Additional sub LLM (load balanced)
[[providers]]
provider_type = "ollama"
base_url = "http://big72.local:11434"
model = "mistral:7b"
role = "sub"
weight = 1
```

## Global Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `max_iterations` | int | 20 | Maximum command iterations before giving up |
| `max_sub_calls` | int | 50 | Maximum llm_query delegations |
| `output_limit` | int | 10000 | Truncate command output beyond this |

## Provider Configuration

Each `[[providers]]` block defines an LLM endpoint.

### Common Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `provider_type` | string | yes | "deepseek" or "ollama" |
| `model` | string | yes | Model name/ID |
| `role` | string | yes | "root" or "sub" |
| `weight` | int | no | Load balancing weight (default: 1) |

### DeepSeek Provider

```toml
[[providers]]
provider_type = "deepseek"
base_url = "https://api.deepseek.com"
model = "deepseek-chat"
role = "root"
```

**Environment variable:** `DEEPSEEK_API_KEY`

### Ollama Provider

```toml
[[providers]]
provider_type = "ollama"
base_url = "http://localhost:11434"
model = "qwen2.5-coder:14b"
role = "sub"
```

For remote Ollama servers:
```toml
base_url = "http://big72.local:11434"
```

## Roles

### root

The orchestrating LLM that:
- Receives the user query
- Issues JSON commands
- Decides when to call sub-LLMs
- Produces the final answer

**Recommendations:**
- Use a smart, capable model (deepseek-chat, GPT-4, Claude)
- Only one root provider should be configured

### sub

Helper LLMs that handle `llm_query` calls:
- Analyze chunks of text
- Summarize content
- Answer specific questions about context

**Recommendations:**
- Can be smaller/faster models
- Multiple sub providers enable load balancing
- Local Ollama works well for cost savings

## Load Balancing

When multiple sub providers are configured, requests are distributed by weight:

```toml
[[providers]]
provider_type = "ollama"
base_url = "http://localhost:11434"
model = "qwen2.5-coder:14b"
role = "sub"
weight = 2  # Gets 2x the requests

[[providers]]
provider_type = "ollama"
base_url = "http://big72.local:11434"
model = "mistral:7b"
role = "sub"
weight = 1  # Gets 1x the requests
```

## Model Recommendations

### For Root LLM

| Model | Provider | Notes |
|-------|----------|-------|
| deepseek-chat | DeepSeek | Fast, capable, recommended |
| gpt-4 | OpenAI | Excellent but expensive |
| qwen2.5-coder:32b | Ollama | Good local option, needs 24GB+ VRAM |

### For Sub LLMs

| Model | Provider | Notes |
|-------|----------|-------|
| qwen2.5-coder:14b | Ollama | Good balance of speed/quality |
| mistral:7b | Ollama | Very fast, simpler tasks |
| llama3.2 | Ollama | General purpose |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DEEPSEEK_API_KEY` | API key for DeepSeek provider |
| `OPENAI_API_KEY` | API key for OpenAI provider (if added) |
