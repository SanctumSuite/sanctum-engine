"""Task execution API endpoints."""
import json
import logging
import time
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..db import get_db
from ..config import settings
from ..models.db import TaskLog
from ..models.schemas import TaskRequest, TaskResponse, EmbedRequest, EmbedResponse
from ..core import context_manager, model_resolver, task_runner
from ..runtimes.base import LLMRuntime, StreamDelta, StreamDone
from ..runtimes.ollama import OllamaRuntime

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tasks"])

# Runtime registry — instantiated on first use
_runtimes: dict[str, LLMRuntime] = {}


def get_runtime(name: str) -> LLMRuntime:
    """Get or create a runtime instance by name."""
    if name not in _runtimes:
        if name == "ollama":
            _runtimes["ollama"] = OllamaRuntime()
        elif name == "openrouter":
            if not settings.openrouter_api_key:
                raise ValueError("OpenRouter API key not configured. Set SANCTUM_OPENROUTER_API_KEY.")
            from ..runtimes.openrouter import OpenRouterRuntime
            _runtimes["openrouter"] = OpenRouterRuntime(
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
            )
        else:
            raise ValueError(f"Unknown runtime: {name}")
    return _runtimes[name]


@router.post("/task", response_model=TaskResponse)
async def execute_task(request: TaskRequest, db: Session = Depends(get_db)):
    """Execute an LLM task with context management, validation, and retry logic."""
    # Select runtime: request override > default
    runtime_name = request.runtime or settings.default_runtime
    try:
        runtime = get_runtime(runtime_name)
    except ValueError as e:
        from ..models.schemas import TaskResponse, TaskError, TaskMeta
        import uuid, time
        return TaskResponse(
            status="error",
            error=TaskError(code="RUNTIME_UNAVAILABLE", message=str(e)),
            meta=TaskMeta(
                task_id=uuid.uuid4(), model_used="none", runtime=runtime_name,
                tokens_in=0, tokens_out=0, context_window=0,
                context_utilization=0.0, latency_ms=0, attempts=0, temperature=0.0,
            ),
        )
    return await task_runner.run_task(request, runtime, db)


@router.post("/task/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    """Generate embeddings — always uses Ollama (local only)."""
    model = request.model or settings.embedding_model
    ollama = get_runtime("ollama")
    embeddings = await ollama.embed(model, request.texts)
    dimensions = len(embeddings[0]) if embeddings else 0
    return EmbedResponse(embeddings=embeddings, model=model, dimensions=dimensions)


@router.post("/task/stream")
async def stream_task(request: TaskRequest, db: Session = Depends(get_db)):
    """Stream a generate_text task as NDJSON chunks.

    Response is `application/x-ndjson`: one JSON object per line. Event types:
      - {"type": "delta", "content": "..."}     — a chunk of generated text
      - {"type": "done",  "meta": { ... }}      — stream finished, totals + model
      - {"type": "error", "code": "...", "message": "..."}

    Only `task_type="generate_text"` is supported in streaming mode. Other
    task types (extract_json, embed, rerank, vision) require the non-streaming
    /task endpoint because they need the full response to validate/repair.

    Streaming skips retry and output validation — the caller takes the raw
    stream. Context budgeting is still applied upfront; oversize prompts
    produce a single error event instead of a partial stream.
    """
    async def generator():
        task_id = uuid.uuid4()
        start = time.monotonic()

        if request.task_type != "generate_text":
            yield _ndjson({
                "type": "error",
                "code": "UNSUPPORTED_TASK_TYPE",
                "message": f"/task/stream only supports task_type='generate_text', got '{request.task_type}'",
            })
            return

        # Resolve model (same logic as /task).
        runtime_name = request.runtime or settings.default_runtime
        try:
            runtime: LLMRuntime = get_runtime(runtime_name)
        except ValueError as e:
            yield _ndjson({"type": "error", "code": "RUNTIME_UNAVAILABLE", "message": str(e)})
            return

        if request.model:
            resolved = model_resolver.ResolvedModel(
                name=request.model,
                safe_context_limit=request.context_budget or settings.default_context_budget,
                runtime=runtime_name,
                config=model_resolver.get_model_config(request.model),
            )
        else:
            resolved = model_resolver.resolve_model_with_fallback(
                db, request.model_preference, runtime_name,
            )
        if not resolved:
            yield _ndjson({
                "type": "error",
                "code": "MODEL_UNAVAILABLE",
                "message": f"No model available for preference '{request.model_preference}'",
            })
            return

        temperature = request.temperature if request.temperature is not None else settings.default_temperature
        max_tokens = request.max_tokens or settings.default_max_tokens
        safe_limit = min(
            request.context_budget or settings.default_context_budget,
            resolved.safe_context_limit,
        )

        system_prompt = request.system_prompt
        user_prompt = request.user_prompt
        model_config = resolved.config
        if model_config.get("append_to_prompt"):
            user_prompt = user_prompt + " " + model_config["append_to_prompt"]

        budget = context_manager.check_context_budget(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            safe_context_limit=safe_limit,
            max_output_tokens=max_tokens,
        )
        if not budget.fits:
            yield _ndjson({
                "type": "error",
                "code": "CONTEXT_OVERFLOW",
                "message": budget.message,
            })
            return

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        tokens_in = 0
        tokens_out = 0
        cost_usd = 0.0
        model_used = resolved.name
        try:
            async for chunk in runtime.generate_stream(
                model=resolved.name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                num_ctx=safe_limit,
            ):
                if isinstance(chunk, StreamDelta):
                    yield _ndjson({"type": "delta", "content": chunk.content})
                elif isinstance(chunk, StreamDone):
                    tokens_in = chunk.tokens_in
                    tokens_out = chunk.tokens_out
                    cost_usd = chunk.cost_usd
                    model_used = chunk.model
        except NotImplementedError as e:
            yield _ndjson({"type": "error", "code": "STREAMING_UNSUPPORTED", "message": str(e)})
            return
        except Exception as e:
            logger.exception("Stream task failed")
            yield _ndjson({
                "type": "error",
                "code": "RUNTIME_ERROR",
                "message": f"{type(e).__name__}: {str(e)[:300]}",
            })
            # fall through to log the failure
            _log_stream_task(
                db, task_id, request, resolved, status="error",
                tokens_in=tokens_in, tokens_out=tokens_out,
                elapsed_ms=int((time.monotonic() - start) * 1000),
                temperature=temperature, context_utilization=budget.utilization,
                error_code="RUNTIME_ERROR", error_detail=str(e)[:500],
            )
            return

        elapsed_ms = int((time.monotonic() - start) * 1000)
        _log_stream_task(
            db, task_id, request, resolved, status="success",
            tokens_in=tokens_in, tokens_out=tokens_out,
            elapsed_ms=elapsed_ms,
            temperature=temperature, context_utilization=budget.utilization,
        )

        yield _ndjson({
            "type": "done",
            "meta": {
                "task_id": str(task_id),
                "model_used": model_used,
                "runtime": resolved.runtime,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "context_window": safe_limit,
                "context_utilization": budget.utilization,
                "latency_ms": elapsed_ms,
                "attempts": 1,
                "temperature": temperature,
                "cost_usd": cost_usd,
            },
        })

    return StreamingResponse(generator(), media_type="application/x-ndjson")


def _ndjson(obj: dict) -> bytes:
    return (json.dumps(obj) + "\n").encode("utf-8")


def _log_stream_task(
    db: Session,
    task_id,
    request: TaskRequest,
    resolved,
    status: str,
    tokens_in: int,
    tokens_out: int,
    elapsed_ms: int,
    temperature: float,
    context_utilization: float,
    error_code: str | None = None,
    error_detail: str | None = None,
) -> None:
    """Minimal task log for streamed tasks. Streaming skips retry/validation,
    so fewer fields are meaningful than the non-streaming path."""
    try:
        db.add(TaskLog(
            id=task_id,
            task_type=request.task_type,
            model_requested=request.model_preference,
            model_used=resolved.name,
            runtime=resolved.runtime,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            context_utilization=context_utilization,
            status=status,
            error_code=error_code,
            error_detail=error_detail,
            attempts=1,
            was_repaired=False,
            latency_ms=elapsed_ms,
            temperature=temperature,
        ))
        db.commit()
    except Exception as e:
        logger.error(f"Failed to log stream task {task_id}: {e}")
        db.rollback()
