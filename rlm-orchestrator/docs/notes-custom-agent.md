# Using Z.ai GLM Coding Plan in Custom Agents

This document explains how to integrate Z.ai's GLM Coding Plan (quota-based subscription) into custom AI coding agents, bypassing the per-token metered API.

## Background

Z.ai offers two billing models for GLM models:

| Plan | Endpoint | Billing |
|------|----------|---------|
| Metered API | `https://api.z.ai/api/paas/v4` | Per-token ($0.60/1M input, $2.20/1M output) |
| **Coding Plan** | `https://api.z.ai/api/coding/paas/v4` | Quota-based ($3-15/month) |

The Coding Plan is marketed for "supported tools" (Claude Code, OpenCode, Cline, etc.), but the restriction is **not technical** - it's simply a different API endpoint path.

## The "Secret"

Any agent can use the Coding Plan by calling the `/api/coding/paas/v4` endpoint instead of `/api/paas/v4`. Same API key, same request format, different billing.

## API Details

### Endpoint
```
https://api.z.ai/api/coding/paas/v4/chat/completions
```

### Authentication
```
Authorization: Bearer <your_zai_api_key>
```

### Request Format (OpenAI-compatible)
```json
{
  "model": "glm-4.7",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"}
  ],
  "max_tokens": 1000,
  "temperature": 0.7
}
```

### Available Models
- `glm-4.7` - Flagship model (200K context, 128K output)
- `glm-4.5-flash` - Fast/free tier
- `glm-4.5-air` - Budget option
- `glm-4.5v` - Vision model

## Rust Implementation

```rust
use reqwest::Client;
use serde::{Deserialize, Serialize};

const ZAI_CODING_ENDPOINT: &str = "https://api.z.ai/api/coding/paas/v4/chat/completions";

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: u32,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

async fn call_glm_coding_plan(
    api_key: &str,
    messages: Vec<Message>,
) -> Result<String, Box<dyn std::error::Error>> {
    let client = Client::new();

    let request = ChatRequest {
        model: "glm-4.7".to_string(),
        messages,
        max_tokens: 4096,
    };

    let response = client
        .post(ZAI_CODING_ENDPOINT)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await?;

    let json: serde_json::Value = response.json().await?;

    // GLM-4.7 is a reasoning model - content may be in reasoning_content
    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    let reasoning = json["choices"][0]["message"]["reasoning_content"]
        .as_str()
        .unwrap_or("");

    Ok(if content.is_empty() { reasoning.to_string() } else { content.to_string() })
}
```

## LiteLLM Configuration

To use the Coding Plan through LiteLLM proxy:

```yaml
# In config.yaml
model_list:
  - model_name: glm-4.7
    litellm_params:
      model: openai/glm-4.7           # Use openai/ prefix for custom endpoint
      api_key: os.environ/ZAI_API_KEY
      api_base: https://api.z.ai/api/coding/paas/v4  # Coding Plan endpoint
```

Note: Using `openai/` prefix with custom `api_base` instead of `zai/` prefix, because `zai/` hardcodes the metered endpoint.

## Quota Limits

The Coding Plan has quota limits that reset every 5 hours:

| Tier | Price | Prompts per 5 hours |
|------|-------|---------------------|
| Lite | $3/month | ~120 prompts |
| Pro | $15/month | ~600 prompts |
| Max | Higher | ~2400 prompts |

If quota is exhausted, wait for the next 5-hour cycle. The API will return a rate limit error, not charge you metered rates.

## Important Notes

1. **Reasoning Model Format**: GLM-4.7 returns responses in `reasoning_content` field, not just `content`. Handle both.

2. **Same API Key**: Your Z.ai API key works for both endpoints. The billing is tracked separately based on which endpoint you call.

3. **No User-Agent Restrictions**: There's no technical verification of which agent is calling. Any HTTP client works.

4. **Token Caching**: Z.ai offers context caching at reduced rates (~$0.11/1M for cached tokens).

## Comparison: Coding Plan vs Metered

For RLM orchestrator testing (based on our benchmarks):

| Model | Endpoint | Cost for 100K tokens |
|-------|----------|---------------------|
| GLM-4.7 (Metered) | /api/paas/v4 | ~$0.22 |
| GLM-4.7 (Coding Plan) | /api/coding/paas/v4 | $0 (within quota) |
| DeepSeek V3 | api.deepseek.com | ~$0.03 |

The Coding Plan is effectively free if you stay within quota limits.

## References

- [Z.ai Developer Documentation](https://docs.z.ai/devpack/overview)
- [GLM Coding Plan Subscription](https://z.ai/subscribe)
- [Mastra Z.ai Provider](https://mastra.ai/models/providers/zai-coding-plan)
- [LiteLLM Z.ai Docs](https://docs.litellm.ai/docs/providers/zai)
