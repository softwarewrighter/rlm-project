# RLM API Reference

The RLM server exposes a REST API for running queries and checking status.

## Base URL

```
http://localhost:8080
```

## Endpoints

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

### POST /query

Run an RLM query and get the final answer.

**Request:**
```json
{
  "query": "How many ERROR lines are there?",
  "context": "Line 1: OK\nLine 2: ERROR timeout\nLine 3: OK"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | yes | The question to answer |
| `context` | string | yes | The text to analyze |

**Response:**
```json
{
  "answer": "There is 1 ERROR line in the log.",
  "iterations": 3,
  "sub_calls": 0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `answer` | string | The final answer from the LLM |
| `iterations` | int | Number of command iterations |
| `sub_calls` | int | Number of llm_query sub-calls |

**Error Response:**
```json
{
  "error": "Max iterations reached without final answer"
}
```

### POST /debug

Run an RLM query with full iteration history.

**Request:** Same as `/query`

**Response:**
```json
{
  "answer": "There is 1 ERROR line.",
  "iterations": [
    {
      "iteration": 1,
      "command": "{\"op\": \"find\", \"text\": \"ERROR\", \"store\": \"errors\"}",
      "output": "Found 1 occurrence",
      "error": null
    },
    {
      "iteration": 2,
      "command": "{\"op\": \"final\", \"answer\": \"There is 1 ERROR line.\"}",
      "output": null,
      "error": null
    }
  ],
  "total_sub_calls": 0,
  "context_size": 52
}
```

### GET /visualize

Serve the interactive HTML visualizer.

Opens a web interface for running queries with visual feedback. See [Visualizer Guide](visualizer.md).

## Example Usage

### curl

```bash
# Simple query
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Count the lines", "context": "one\ntwo\nthree"}'

# Debug mode
curl -X POST http://localhost:8080/debug \
  -H "Content-Type: application/json" \
  -d '{"query": "Count the lines", "context": "one\ntwo\nthree"}'
```

### Python

```python
import httpx

response = httpx.post(
    "http://localhost:8080/query",
    json={
        "query": "How many functions?",
        "context": open("code.py").read()
    }
)
print(response.json()["answer"])
```

### JavaScript

```javascript
const response = await fetch("http://localhost:8080/query", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query: "Summarize the errors",
    context: logContent
  })
});
const { answer } = await response.json();
console.log(answer);
```

## Error Handling

| Status | Meaning |
|--------|---------|
| 200 | Success |
| 400 | Invalid request (missing query/context) |
| 500 | Server error (LLM failure, max iterations) |

Always check for the `error` field in responses.

## Rate Limits

No built-in rate limiting. The server processes one request at a time. For production use, consider adding a reverse proxy with rate limiting.
