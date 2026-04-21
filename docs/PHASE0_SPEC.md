# Sanctum Engine — Phase 0 Technical Specification

## Overview

Sanctum Engine is a standalone service that sits between any Sanctum application and local LLM runtimes (Ollama, MLX, llama.cpp, vLLM). It provides a task-oriented API with built-in context management, output validation, retry logic, and model management.

**This spec covers the initial implementation: Ollama runtime on Apple Silicon (M3 Max 128GB).**

---

## 1. Docker Architecture

```
docker-compose.yml
├── sanctum-engine    (FastAPI, port 8100)
├── ollama            (LLM runtime, port 11434, GPU passthrough)
└── postgres          (metadata + task logs, port 5432)
```

### Why Engine Gets Its Own Container

- Ollama already runs as a system service on the Mac. The engine container talks to the host Ollama.
- Engine manages the API layer, validation, retries, and logging. Ollama does the inference.
- Later runtimes (MLX, vLLM) plug in alongside Ollama without changing the engine API.

### Docker Compose Layout

```yaml
services:
  sanctum-engine:
    build: ./engine
    ports:
      - "8100:8100"
    environment:
      - OLLAMA_HOST=host.docker.internal:11434   # Mac Docker host access
      - DATABASE_URL=postgresql://sanctum:sanctum@postgres:5432/sanctum_engine
      - LOG_LEVEL=INFO
    depends_on:
      - postgres

  postgres:
    image: pgvector/pgvector:pg16
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=sanctum
      - POSTGRES_PASSWORD=sanctum
      - POSTGRES_DB=sanctum_engine
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

Note: Ollama runs on the host (not in Docker) to get direct Metal GPU access. The engine container reaches it via `host.docker.internal`.

---

## 2. API Contract

Base URL: `http://localhost:8100`

### 2.1 POST /task — Execute a Task

The core endpoint. Every LLM interaction goes through here.

**Request:**
```json
{
  "task_type": "extract_json",
  "model_preference": "reasoning",
  "system_prompt": "You are an intelligence analyst...",
  "user_prompt": "Analyze this document and extract...",
  "output_schema": {
    "type": "object",
    "required": ["evidence_items"],
    "properties": {
      "evidence_items": {
        "type": "array",
        "items": {
          "type": "object",
          "required": ["content"],
          "properties": {
            "content": {"type": "string"},
            "context": {"type": "string"},
            "page_or_section": {"type": "string"}
          }
        }
      }
    }
  },
  "max_retries": 3,
  "context_budget": 16384,
  "temperature": 0.3
}
```

**Task Types:**

| task_type | Description | Output |
|-----------|-------------|--------|
| `extract_json` | Generate and validate JSON output | Parsed JSON dict |
| `generate_text` | Free-form text generation | String |
| `embed` | Generate embeddings for text(s) | Array of float arrays |
| `vision` | Analyze image(s) with text prompt | String or JSON |
| `translate` | Translate text to target language | String |
| `rerank` | Rerank candidates against a query | Sorted array with scores |

**Model Preferences:**

| preference | Resolves to (initial) | Use case |
|------------|----------------------|----------|
| `reasoning` | qwen3.5:latest | Hypothesis generation, complex analysis |
| `fast` | ministral-3:latest | Batch work: rating, extraction, summarization |
| `vision` | deepseek-ocr:latest or glm-ocr:latest | Image/OCR analysis |
| `embedding` | nomic-embed-text:latest | Document embeddings |
| `translation` | translategemma:27b-it-fp16 | Foreign language docs |
| (specific name) | That exact model | Override for comparison runs |

**Response (success):**
```json
{
  "status": "success",
  "result": { "evidence_items": [ ... ] },
  "meta": {
    "task_id": "uuid",
    "model_used": "qwen3.5:latest",
    "runtime": "ollama",
    "tokens_in": 2400,
    "tokens_out": 890,
    "context_window": 32768,
    "context_utilization": 0.42,
    "latency_ms": 14200,
    "attempts": 1,
    "temperature": 0.3
  }
}
```

**Response (failure after retries):**
```json
{
  "status": "error",
  "error": {
    "code": "JSON_VALIDATION_FAILED",
    "message": "Output did not match schema after 3 attempts",
    "attempts": [
      {"attempt": 1, "error": "Truncated JSON at position 1055", "raw_length": 1055},
      {"attempt": 2, "error": "Missing required field: evidence_items", "raw_length": 434},
      {"attempt": 3, "error": "Truncated JSON at position 780", "raw_length": 780}
    ]
  },
  "meta": { ... }
}
```

### 2.2 POST /task/embed — Batch Embeddings

Optimized endpoint for embedding multiple texts in one call.

```json
{
  "texts": ["passage 1 text...", "passage 2 text...", "..."],
  "model": "nomic-embed-text:latest"
}
```

Response:
```json
{
  "embeddings": [[0.012, -0.034, ...], [0.008, 0.045, ...], ...],
  "model": "nomic-embed-text:latest",
  "dimensions": 768
}
```

### 2.3 POST /task/rerank — Rerank Candidates

```json
{
  "query": "Iranian energy infrastructure sanctions",
  "candidates": [
    {"id": "passage_42", "text": "Treasury Secretary said..."},
    {"id": "passage_17", "text": "Nuclear talks resumed..."}
  ],
  "top_k": 10
}
```

Response:
```json
{
  "rankings": [
    {"id": "passage_17", "score": 0.92, "rank": 1},
    {"id": "passage_42", "score": 0.78, "rank": 2}
  ]
}
```

### 2.4 GET /health — Health Check

```json
{
  "status": "healthy",
  "ollama_connected": true,
  "models_loaded": ["ministral-3:latest"],
  "models_available": ["ministral-3:latest", "qwen3.5:latest", "nomic-embed-text:latest", ...],
  "memory": {
    "system_total_gb": 128,
    "estimated_available_gb": 85,
    "models_loaded_gb": 6.0
  },
  "stats": {
    "tasks_completed_24h": 247,
    "tasks_failed_24h": 3,
    "avg_latency_ms": 8400,
    "error_rate": 0.012
  }
}
```

### 2.5 GET /models — List Available Models

```json
{
  "models": [
    {
      "name": "qwen3.5:latest",
      "parameter_size": "9.7B",
      "quantization": "Q4_K_M",
      "context_length": 262144,
      "safe_context_limit": 32768,
      "capabilities": ["reasoning", "fast", "json"],
      "size_gb": 6.1,
      "loaded": false
    },
    {
      "name": "nomic-embed-text:latest",
      "parameter_size": "137M",
      "quantization": "F16",
      "context_length": 2048,
      "safe_context_limit": 1800,
      "capabilities": ["embedding"],
      "size_gb": 0.3,
      "loaded": true
    }
  ]
}
```

### 2.6 GET /models/{model_name}/stats — Model Performance Stats

```json
{
  "model": "ministral-3:latest",
  "total_tasks": 523,
  "success_rate": 0.96,
  "avg_tokens_per_sec": 32.5,
  "avg_latency_ms": 8200,
  "json_parse_failures": 18,
  "json_repair_successes": 12,
  "by_task_type": {
    "extract_json": {"count": 340, "success_rate": 0.94},
    "generate_text": {"count": 183, "success_rate": 0.99}
  }
}
```

---

## 3. Context Window Management

This is where the old codebase failed. The engine must make context overflow **impossible**.

### 3.1 Context Budget Calculation

For every task, before calling the LLM:

```
model_max_context = model's advertised context length (e.g., 262144 for qwen3.5)
safe_context_limit = min(model_max_context * 0.7, context_budget or 32768)

input_tokens = count_tokens(system_prompt + user_prompt)
reserved_output = max(max_tokens_setting, 4096)

if input_tokens + reserved_output > safe_context_limit:
    → REJECT with error "Input too large for context budget"
    → Include: input_tokens, reserved_output, safe_context_limit
    → Suggest: smaller input or chunked processing
```

### 3.2 Safe Context Limits by Model

These are the **actual limits we enforce**, not what the model advertises:

| Model | Advertised | Safe Limit | Notes |
|-------|-----------|------------|-------|
| ministral-3:latest | 262K | 32K | 8.9B model, quality degrades past 32K on M3 Max |
| qwen3.5:latest | 262K | 32K | Same — large context eats memory and slows inference |
| deepseek-ocr:latest | 8K | 6K | Small model, tight window |
| glm-ocr:latest | 8K | 6K | Small model, tight window |
| nomic-embed-text | 2K | 1800 | Embedding model, fixed window |
| translategemma:27b | 8K | 6K | Translation model |

**Why 32K not higher:** On M3 Max 128GB, a 10B Q4 model at 32K context uses ~12GB. At 64K it's ~18GB and inference speed drops 40%. At 128K+ quality measurably degrades. 32K is the sweet spot for reliability and speed.

### 3.3 Token Counting

Use tiktoken (cl100k_base) for estimation. It's not model-exact but close enough for budgeting (within 10%).

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))
```

### 3.4 Chunked Processing

When a caller sends input that exceeds the context budget, the engine can optionally chunk it:

```json
{
  "task_type": "extract_json",
  "chunking": {
    "enabled": true,
    "chunk_tokens": 2000,
    "overlap_tokens": 200,
    "merge_strategy": "concatenate_arrays"
  },
  "user_prompt": "...very long document text..."
}
```

The engine:
1. Splits the user prompt's document text into chunks
2. Runs each chunk as a separate LLM call
3. Merges results according to `merge_strategy`
4. Returns the merged result as if it were a single call

Merge strategies:
- `concatenate_arrays` — combine array items from each chunk (e.g., evidence_items)
- `keep_last` — use only the final chunk's result
- `summarize` — (future) use LLM to merge chunk results

---

## 4. Output Validation & Retry Logic

### 4.1 Validation Pipeline

For `extract_json` tasks:

```
Attempt 1: Call LLM → Parse JSON → Validate against schema
  ├── Success → return result
  └── Failure → classify error → Attempt 2

Attempt 2: Call LLM (temperature * 0.5, add format reminder to prompt) → Parse → Validate
  ├── Success → return result
  └── Failure → classify error → Attempt 3

Attempt 3: Call LLM (temperature 0.1, simplified prompt) → Parse → Validate
  ├── Success → return result
  └── Failure → return error with all 3 attempt diagnostics
```

### 4.2 JSON Repair

Before declaring a parse failure, attempt repair:

1. **Strip wrapping:** Remove markdown code fences, thinking tags, preamble text
2. **Find JSON boundaries:** Locate outermost `{` and `}` or `[` and `]`
3. **Fix truncation:** If JSON is truncated (no closing brackets):
   - Find the last complete object/array element
   - Close open brackets/braces
   - Log warning: "Repaired truncated JSON, may have lost N items"
4. **Fix escaping:** Unescape literal newlines in string values
5. **Schema validation:** If `output_schema` provided, validate with jsonschema

### 4.3 Error Classification

| Error Code | Meaning | Retry Strategy |
|------------|---------|---------------|
| `EMPTY_RESPONSE` | Model returned empty string | Retry, check model is loaded |
| `JSON_PARSE_FAILED` | Response is not valid JSON | Retry with lower temp + format reminder |
| `JSON_TRUNCATED` | JSON cut off mid-output | Attempt repair; if fails, retry with shorter input |
| `SCHEMA_VALIDATION_FAILED` | Valid JSON but wrong structure | Retry with schema in prompt |
| `CONTEXT_OVERFLOW` | Input exceeds safe limit | Reject (don't retry), suggest chunking |
| `MODEL_UNAVAILABLE` | Model not loaded/available | Try fallback model |
| `TIMEOUT` | Inference took too long | Retry once, then fail |

---

## 5. Model Registry

### 5.1 Database Table

```sql
CREATE TABLE model_registry (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,        -- "qwen3.5:latest"
    parameter_size VARCHAR(50),                -- "9.7B"
    quantization VARCHAR(50),                  -- "Q4_K_M"
    context_length INTEGER,                    -- 262144 (advertised)
    safe_context_limit INTEGER,                -- 32768 (enforced)
    capabilities TEXT[] NOT NULL DEFAULT '{}', -- {"reasoning", "json", "fast"}
    size_bytes BIGINT,
    runtime VARCHAR(50) DEFAULT 'ollama',      -- "ollama", "mlx", "llamacpp", "vllm"
    is_available BOOLEAN DEFAULT true,
    priority INTEGER DEFAULT 50,               -- higher = preferred for its capability
    config JSONB DEFAULT '{}',                 -- model-specific config overrides
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 5.2 Capability-to-Model Resolution

When a task comes in with `model_preference: "reasoning"`:

```sql
SELECT name FROM model_registry
WHERE 'reasoning' = ANY(capabilities)
  AND is_available = true
  AND runtime = 'ollama'
ORDER BY priority DESC
LIMIT 1;
```

If the primary model fails, fall back to the next in priority order.

### 5.3 Auto-Discovery

On startup, the engine:
1. Queries Ollama for available models (`/api/tags`)
2. For each model, queries capabilities (`/api/show`)
3. Inserts/updates the model_registry table
4. Assigns default capabilities based on model family:
   - `qwen*` → reasoning, fast, json
   - `ministral*` → reasoning, fast, json
   - `nomic-embed*` → embedding
   - `deepseek-ocr*`, `glm-ocr*` → vision, ocr
   - `translategemma*` → translation
   - `llava*` → vision

### 5.4 Qwen3.5 /nothink Handling

qwen3.5 ignores `think=false` in the Ollama API. When the engine detects qwen3.5 as the selected model and the task does NOT require thinking, it appends `/nothink` to the user prompt automatically. This is tracked in model config:

```json
{"append_to_prompt": "/nothink", "strip_thinking_tags": true}
```

---

## 6. Task Logging

Every task is logged for performance tracking, debugging, and model comparison.

```sql
CREATE TABLE task_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_type VARCHAR(50) NOT NULL,
    model_requested VARCHAR(255),              -- what the caller asked for
    model_used VARCHAR(255) NOT NULL,          -- what actually ran
    runtime VARCHAR(50) NOT NULL,

    tokens_in INTEGER,
    tokens_out INTEGER,
    context_utilization FLOAT,

    status VARCHAR(20) NOT NULL,               -- "success", "error", "repaired"
    error_code VARCHAR(50),
    error_detail TEXT,

    attempts INTEGER DEFAULT 1,
    was_repaired BOOLEAN DEFAULT false,        -- JSON was truncated and repaired

    latency_ms INTEGER,
    temperature FLOAT,

    -- For multi-model comparison
    comparison_group_id UUID,                  -- groups tasks that ran same prompt on different models

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_task_log_model ON task_log(model_used);
CREATE INDEX idx_task_log_status ON task_log(status);
CREATE INDEX idx_task_log_created ON task_log(created_at);
CREATE INDEX idx_task_log_comparison ON task_log(comparison_group_id);
```

---

## 7. File Structure

```
engine/
├── Dockerfile
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app, startup, health check
│   ├── config.py                # Settings from env vars
│   ├── api/
│   │   ├── __init__.py
│   │   ├── tasks.py             # POST /task endpoint
│   │   ├── embeddings.py        # POST /task/embed endpoint
│   │   ├── rerank.py            # POST /task/rerank endpoint
│   │   ├── models.py            # GET /models, GET /models/{name}/stats
│   │   └── health.py            # GET /health
│   ├── core/
│   │   ├── __init__.py
│   │   ├── task_runner.py       # Orchestrates: validate → call LLM → validate output → retry
│   │   ├── context_manager.py   # Token counting, budget enforcement, chunking
│   │   ├── output_validator.py  # JSON parsing, repair, schema validation
│   │   └── model_resolver.py    # Capability → model resolution, fallback logic
│   ├── runtimes/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract runtime interface
│   │   ├── ollama.py            # Ollama runtime implementation
│   │   ├── mlx.py               # (stub) MLX runtime — future
│   │   ├── llamacpp.py          # (stub) llama.cpp runtime — future
│   │   └── vllm.py              # (stub) vLLM runtime — future
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py           # Pydantic models for API request/response
│   │   └── db.py                # SQLAlchemy models (model_registry, task_log)
│   └── db.py                    # Database connection, session management
├── tests/
│   ├── conftest.py              # Fixtures: test DB, mock Ollama
│   ├── test_context_manager.py  # Token counting, budget checks
│   ├── test_output_validator.py # JSON parse, repair, schema validation
│   ├── test_model_resolver.py   # Capability resolution, fallback
│   ├── test_task_runner.py      # End-to-end task execution
│   ├── test_ollama_runtime.py   # Ollama API integration
│   └── test_api.py              # HTTP endpoint tests
└── alembic/                     # Database migrations
    ├── alembic.ini
    └── versions/
```

---

## 8. Runtime Interface

All runtimes implement the same abstract interface:

```python
from abc import ABC, abstractmethod

class LLMRuntime(ABC):
    """Abstract interface for LLM runtimes."""

    @abstractmethod
    async def generate(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        num_ctx: int,
    ) -> RuntimeResponse:
        """Generate a completion. Returns response text + metadata."""
        ...

    @abstractmethod
    async def embed(
        self,
        model: str,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        ...

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """List available models on this runtime."""
        ...

    @abstractmethod
    async def model_info(self, model: str) -> ModelInfo:
        """Get detailed info about a model."""
        ...

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Check if the runtime is reachable and working."""
        ...
```

```python
@dataclass
class RuntimeResponse:
    content: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    model: str

@dataclass
class ModelInfo:
    name: str
    parameter_size: str
    quantization: str
    context_length: int
    size_bytes: int
    families: list[str]
```

### 8.1 Ollama Runtime Implementation

The Ollama runtime uses the **native Ollama API** (`/api/chat`, `/api/embed`) not the OpenAI-compatible endpoint. This gives us access to:
- `num_ctx` control
- `num_predict` (max output tokens)
- Token count reporting (`prompt_eval_count`, `eval_count`)
- Model load/unload
- Thinking content (for models that support it)

Key behaviors:
- Timeout: 60s connect, 600s read (local models can be slow on first load)
- When a model isn't loaded, Ollama auto-loads it. The engine tracks this latency separately.
- For qwen3.5: append `/nothink` to user message for non-thinking tasks

---

## 9. Test Plan

### 9.1 Unit Tests (no Ollama required)

| Test | What it validates |
|------|------------------|
| `test_context_manager.py` | Token counting accuracy (within 15% of actual), budget rejection when input too large, chunking splits correctly |
| `test_output_validator.py` | Clean JSON parses correctly, truncated JSON gets repaired, code-fenced JSON gets stripped, schema validation catches missing fields, thinking tags get stripped |
| `test_model_resolver.py` | "reasoning" resolves to qwen3.5, fallback to ministral-3 when qwen3.5 unavailable, unknown capability returns error |

### 9.2 Integration Tests (requires Ollama running)

| Test | What it validates |
|------|------------------|
| `test_ollama_runtime.py` | Can connect to Ollama, list models, generate completion, generate embedding |
| `test_task_runner.py` | Full task flow: submit extract_json task → get valid JSON back. Retry on simulated failure. Chunked processing of large input. |

### 9.3 Reliability Tests (THE CRITICAL ONES)

Run 100 varied extract_json tasks and measure:

| Metric | Target |
|--------|--------|
| Valid JSON returned | >= 95% |
| Schema-valid JSON | >= 90% |
| Repaired truncations (salvaged) | Track count |
| Total failures (no valid output) | <= 5% |
| Average latency | Track baseline |
| Context overflow rejections | 0 unexpected |

**Test prompts should include:**
- Short prompts (100 tokens) → small JSON
- Medium prompts (2000 tokens) → medium JSON
- Large prompts (8000 tokens) → should chunk
- Prompts requesting large array output (10+ items)
- Prompts with complex nested JSON schemas
- Vision tasks with image input
- Embedding tasks with batch input

### 9.4 How to Run Tests

```bash
# Unit tests (no dependencies)
cd engine && pytest tests/ -k "not integration" -v

# Integration tests (Ollama must be running)
cd engine && pytest tests/ -k "integration" -v

# Reliability suite
cd engine && python tests/reliability_suite.py --tasks 100 --report
```

---

## 10. Configuration

All via environment variables with sensible defaults:

```python
class EngineConfig:
    # Runtime
    OLLAMA_HOST: str = "http://localhost:11434"
    DEFAULT_RUNTIME: str = "ollama"

    # Context defaults
    DEFAULT_CONTEXT_BUDGET: int = 32768
    CONTEXT_SAFETY_FACTOR: float = 0.7       # use 70% of advertised max
    DEFAULT_MAX_TOKENS: int = 4096

    # Retry
    MAX_RETRIES: int = 3
    RETRY_TEMPERATURE_DECAY: float = 0.5     # multiply temp by this each retry

    # Timeouts (seconds)
    LLM_CONNECT_TIMEOUT: int = 60
    LLM_READ_TIMEOUT: int = 600

    # Database
    DATABASE_URL: str = "postgresql://sanctum:sanctum@localhost:5432/sanctum_engine"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_TASK_PROMPTS: bool = False            # log full prompts (large, disable in prod)
    LOG_TASK_RESPONSES: bool = False          # log full responses

    # Model defaults
    DEFAULT_TEMPERATURE: float = 0.3
    EMBEDDING_MODEL: str = "nomic-embed-text:latest"
```

---

## 11. Startup Sequence

1. Connect to PostgreSQL, run migrations
2. Connect to Ollama, verify healthy
3. Auto-discover models, update model_registry
4. Log: available models, capabilities, memory estimate
5. Pre-load embedding model (always needed, lightweight)
6. Start FastAPI server on port 8100
7. Health endpoint returns `healthy`

---

## 12. What Phase 0 Does NOT Include

- MLX, llama.cpp, or vLLM runtimes (stubs only)
- Multi-model comparison (task logging supports it, but no comparison API yet)
- Model download/approval workflow
- Heretic decensoring integration
- Reranking endpoint (stub only — requires bge-reranker model setup)
- Frontend/UI of any kind
- Authentication/authorization

These are deferred to later phases. The stubs and interfaces are in place so they can be added without restructuring.

---

## 13. Definition of Done

Phase 0 is complete when:

- [ ] Docker compose brings up engine + postgres
- [ ] Engine connects to host Ollama and discovers models
- [ ] `POST /task` with `extract_json` returns valid, schema-validated JSON
- [ ] Context budget enforcement prevents overflow (tested)
- [ ] Truncated JSON gets repaired (tested)
- [ ] Retry logic recovers from first-attempt failures (tested)
- [ ] Failed tasks return structured error diagnostics
- [ ] Task logging records all calls with performance metrics
- [ ] `GET /health` reports accurate system state
- [ ] `GET /models` lists models with capabilities and safe context limits
- [ ] Reliability suite passes: >= 95% valid JSON on 100 varied prompts
- [ ] All unit tests pass
- [ ] All integration tests pass (with Ollama running)
