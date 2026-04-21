# Sanctum Engine

**Local LLM service layer for the [Sanctum Suite](https://github.com/SanctumSuite/sanctum-suite).**

A task-oriented HTTP service that every Sanctum app calls in place of directly invoking Ollama, OpenRouter, or other model providers. Engine handles:

- **Capability → model resolution** (ask for `"reasoning"` or `"translation"`, get a concrete model)
- **Context budgeting** with safe-limit inference per model
- **Input chunking** for oversized prompts
- **JSON output validation + auto-repair** for `extract_json` tasks
- **Retry with temperature decay** and format reminders
- **Model registry** (Postgres-backed, autoregistered from Ollama on startup)
- **Task log** (every call, with tokens, latency, retries)
- **Runtimes**: Ollama (local), OpenRouter (cloud)

Engine was originally built inside [Sanctum Analyst (ACH)](https://github.com/lafintiger/ACH) and extracted into a standalone service so every Sanctum app can share it.

---

## API (quick reference)

### `POST /task`

```json
{
  "task_type": "generate_text",
  "model_preference": "reasoning",
  "system_prompt": "You are a helpful assistant.",
  "user_prompt": "Explain X in one paragraph.",
  "max_retries": 2
}
```

Supported `task_type`: `extract_json`, `generate_text`, `embed`, `vision`, `translate`, `rerank`.
Supported `model_preference`: `reasoning`, `fast`, `vision`, `embedding`, `translation`, `ocr`, or an exact model name.

**Response:**

```json
{
  "status": "success",
  "result": "...",
  "meta": {
    "task_id": "uuid",
    "model_used": "qwen2.5:14b-instruct-q4_K_M",
    "runtime": "ollama",
    "tokens_in": 42,
    "tokens_out": 180,
    "latency_ms": 2340,
    "attempts": 1
  }
}
```

### `GET /health`

Returns `200` with connection status for Ollama, OpenRouter, memory use, and 24h task stats.

### `GET /models`

Lists registered models with capabilities and context limits.

Full schema: [`app/models/schemas.py`](app/models/schemas.py).

---

## Running locally

```bash
# First time
cp .env.example .env
docker compose up -d

# Verify
curl http://localhost:8100/health
```

Engine expects a running Ollama instance. By default it hits `host.docker.internal:11434` (your host's Ollama from inside the container). Override with `SANCTUM_OLLAMA_HOST` in `.env`.

---

## Consuming Engine from a Python app

Install the client package:

```bash
pip install "git+https://github.com/SanctumSuite/sanctum-engine.git@main#subdirectory=client"
```

Use it:

```python
from sanctum_engine_client import engine_client, ENGINE_URL

# Simple text generation
result, latency_ms = await engine_client.run_task(
    task_type="generate_text",
    model_preference="reasoning",
    system_prompt="You are a helpful assistant.",
    user_prompt="Summarize the following...",
    max_retries=2,
)

# Translation (used by TranslaChat)
translated, latency_ms = await engine_client.translate(
    text="Hello, world.",
    source_lang_label="English",
    target_lang_label="French",
)

# Health check
is_up = await engine_client.engine_health()
```

The client reads `ENGINE_URL` (default `http://localhost:8100`) and `ENGINE_TIMEOUT_READ` (default `120.0`) from env vars.

---

## Config

All settings are env vars with the `SANCTUM_` prefix:

| Env var | Default | Purpose |
|---|---|---|
| `SANCTUM_OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `SANCTUM_DATABASE_URL` | `postgresql://sanctum:sanctum@localhost:5432/sanctum_engine` | Postgres for registry + task log |
| `SANCTUM_OPENROUTER_API_KEY` | _(empty)_ | OpenRouter API key; omit to disable cloud runtime |
| `SANCTUM_OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter endpoint |
| `SANCTUM_DEFAULT_RUNTIME` | `ollama` | `ollama` or `openrouter` |
| `SANCTUM_DEFAULT_CONTEXT_BUDGET` | `32768` | Default max input tokens per task |
| `SANCTUM_DEFAULT_MAX_TOKENS` | `4096` | Default max output tokens |
| `SANCTUM_DEFAULT_TEMPERATURE` | `0.3` | Default sampling temperature |
| `SANCTUM_MAX_RETRIES` | `3` | Ceiling on per-task retries |
| `SANCTUM_LLM_READ_TIMEOUT` | `600` | Per-request read timeout (seconds) |
| `SANCTUM_LOG_TASK_PROMPTS` | `false` | Log full prompts (noisy, off by default) |
| `SANCTUM_LOG_TASK_RESPONSES` | `false` | Log full responses (noisy, off by default) |

See [`app/config.py`](app/config.py) for the full list.

---

## Architecture

```
app/
├── main.py                  # FastAPI app, startup model discovery
├── config.py                # pydantic-settings
├── db.py                    # SQLAlchemy engine/session
├── api/
│   ├── tasks.py             # POST /task
│   ├── models.py            # GET /models, model registry admin
│   ├── health.py            # GET /health
│   └── settings.py          # runtime settings
├── core/
│   ├── task_runner.py       # the retry + validate loop
│   ├── context_manager.py   # tokenization + chunking
│   ├── output_validator.py  # JSON validate + repair
│   └── model_resolver.py    # capability → concrete model
├── models/
│   ├── db.py                # SQLAlchemy models (ModelRegistry, TaskLog)
│   └── schemas.py           # Pydantic request/response shapes
└── runtimes/
    ├── base.py              # abstract LLMRuntime
    ├── ollama.py            # Ollama runtime
    └── openrouter.py        # OpenRouter runtime
```

---

## Development

```bash
# Run tests
pip install -r requirements.txt
pytest tests/ -v

# Type check (FastAPI/Pydantic give a lot for free)
python -m mypy app/ --ignore-missing-imports  # optional

# Local run (needs Ollama on host + Postgres somewhere)
uvicorn app.main:app --reload --port 8100
```

See the Sanctum Suite's [`SANCTUM_STANDARDS.md`](https://github.com/SanctumSuite/sanctum-suite/blob/main/SANCTUM_STANDARDS.md) for the suite-wide dev conventions.

---

## Specification

- **Phase 0 Spec** (authoritative): [`docs/PHASE0_SPEC.md`](docs/PHASE0_SPEC.md) — the technical design doc that defined Engine's responsibilities and API surface. (Imported from ACH's docs in a follow-up.)

---

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
