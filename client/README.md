# sanctum-engine-client

Async Python client for [Sanctum Engine](https://github.com/SanctumSuite/sanctum-engine).

## Install

```bash
pip install "git+https://github.com/SanctumSuite/sanctum-engine.git@main#subdirectory=client"
```

## Use

```python
from sanctum_engine_client import engine_client

# Generic task call
result, latency_ms = await engine_client.run_task(
    task_type="generate_text",
    model_preference="reasoning",
    system_prompt="You are a helpful assistant.",
    user_prompt="Explain retries.",
    max_retries=2,
)

# Translation helper (same signature as translachat's)
translated, latency_ms = await engine_client.translate(
    text="Hello.",
    source_lang_label="English",
    target_lang_label="French",
)

# Embeddings
vectors = await engine_client.embed_texts(["hello world", "goodnight moon"])
query_vec = await engine_client.embed_query("what does foo mean?")

# Run many tasks in parallel (multi-model compare)
results = await engine_client.run_tasks_parallel([
    {"task_type": "generate_text", "model": "qwen3:32b", "user_prompt": "Summarize X"},
    {"task_type": "generate_text", "model": "gemma4:31b", "user_prompt": "Summarize X"},
    {"task_type": "generate_text", "model": "gpt-oss:latest", "user_prompt": "Summarize X"},
])
# results[i] is (result, latency_ms) or an Exception per-task

# Health check
is_up = await engine_client.engine_health()
```

### Cost / telemetry tracking

Pass `on_complete=` to `run_task` to receive Engine's full `meta` dict on success
(model_used, runtime, tokens_in, tokens_out, cost_usd, attempts, latency_ms).
The callback runs per-call and is the idiomatic way for apps to roll up their
own cost / usage telemetry without forking the client.

```python
def on_task(meta: dict):
    cost_log.append({
        "model": meta.get("model_used"),
        "cost_usd": meta.get("cost_usd", 0),
        "tokens_in": meta.get("tokens_in", 0),
        "tokens_out": meta.get("tokens_out", 0),
    })

result, _ = await engine_client.run_task(
    task_type="extract_json",
    model_preference="reasoning",
    system_prompt="…", user_prompt="…",
    on_complete=on_task,
)
```

## Config

Environment variables:

| Var | Default | Purpose |
|---|---|---|
| `ENGINE_URL` | `http://localhost:8100` | Engine base URL |
| `ENGINE_TIMEOUT_CONNECT` | `10.0` | Connect timeout (seconds) |
| `ENGINE_TIMEOUT_READ` | `120.0` | Read timeout (seconds) |

Or pass `base_url=` / `timeout=` kwargs to override per call.

## License

Apache 2.0.
