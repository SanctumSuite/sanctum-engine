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

# Health check
is_up = await engine_client.engine_health()
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
