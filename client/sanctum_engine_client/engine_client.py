"""Async HTTP client for Sanctum Engine.

All Sanctum Suite apps call Engine through this client instead of hitting
Ollama/OpenRouter directly. It mirrors the shape translachat's in-tree
engine_client.py established — run_task() for general use, translate() for
the most common app case, plus embed_texts() for embedding work.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, AsyncIterator, Callable

import httpx

logger = logging.getLogger(__name__)

OnCompleteCallback = Callable[[dict], None]

ENGINE_URL: str = os.environ.get("ENGINE_URL", "http://localhost:8100")
_DEFAULT_CONNECT_TIMEOUT = float(os.environ.get("ENGINE_TIMEOUT_CONNECT", "10.0"))
_DEFAULT_READ_TIMEOUT = float(os.environ.get("ENGINE_TIMEOUT_READ", "120.0"))


class EngineError(RuntimeError):
    """Engine returned status=error or the HTTP call failed."""

    def __init__(self, code: str, message: str, attempts: list | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.attempts = attempts or []


def _timeout(connect: float | None = None, read: float | None = None) -> httpx.Timeout:
    return httpx.Timeout(
        connect=connect if connect is not None else _DEFAULT_CONNECT_TIMEOUT,
        read=read if read is not None else _DEFAULT_READ_TIMEOUT,
        write=30.0,
        pool=60.0,
    )


async def engine_health(base_url: str | None = None) -> bool:
    """Is Engine reachable? Swallows errors, returns bool."""
    url = base_url or ENGINE_URL
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{url}/health")
            return resp.status_code == 200
    except Exception:
        return False


async def run_task(
    task_type: str,
    model_preference: str = "fast",
    *,
    system_prompt: str = "",
    user_prompt: str = "",
    model: str | None = None,
    output_schema: dict[str, Any] | None = None,
    max_retries: int = 3,
    context_budget: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    runtime: str | None = None,
    chunking: dict[str, Any] | None = None,
    base_url: str | None = None,
    connect_timeout: float | None = None,
    read_timeout: float | None = None,
    on_complete: OnCompleteCallback | None = None,
) -> tuple[Any, int]:
    """Run a task on Engine. Returns (result, latency_ms).

    Mirrors the full TaskRequest surface. Raises EngineError on failure.

    task_type:       "generate_text" | "extract_json" | "embed" | "vision" | "translate" | "rerank"
    model_preference:"reasoning" | "fast" | "vision" | "embedding" | "translation" | "ocr" | <model-name>

    on_complete:     optional callback receiving Engine's full `meta` dict
                     (model_used, runtime, tokens_in, tokens_out, cost_usd,
                     attempts, latency_ms, …) on success. Use for cost
                     tracking, telemetry, per-task logging in the caller.
    """
    url = base_url or ENGINE_URL
    payload: dict[str, Any] = {
        "task_type": task_type,
        "model_preference": model_preference,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "max_retries": max_retries,
    }
    if model is not None:
        payload["model"] = model
    if output_schema is not None:
        payload["output_schema"] = output_schema
    if context_budget is not None:
        payload["context_budget"] = context_budget
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if runtime is not None:
        payload["runtime"] = runtime
    if chunking is not None:
        payload["chunking"] = chunking

    started = time.time()
    async with httpx.AsyncClient(timeout=_timeout(connect_timeout, read_timeout)) as client:
        resp = await client.post(f"{url}/task", json=payload)
        resp.raise_for_status()
        data = resp.json()
    latency_ms = int((time.time() - started) * 1000)

    if data.get("status") != "success":
        err = data.get("error") or {}
        raise EngineError(
            code=err.get("code", "UNKNOWN"),
            message=err.get("message", ""),
            attempts=err.get("attempts", []),
        )

    if on_complete is not None:
        try:
            on_complete(data.get("meta") or {})
        except Exception as e:
            logger.warning("on_complete callback raised: %s", e)

    return data["result"], latency_ms


async def embed_texts(
    texts: list[str],
    model: str | None = None,
    *,
    base_url: str | None = None,
    read_timeout: float | None = None,
) -> list[list[float]]:
    """Generate embeddings for `texts`. Returns a list of vectors (one per text).

    Model defaults to the Engine server's `embedding_model` setting when None.
    """
    if not texts:
        return []
    url = base_url or ENGINE_URL
    payload: dict[str, Any] = {"texts": texts}
    if model is not None:
        payload["model"] = model

    async with httpx.AsyncClient(timeout=_timeout(None, read_timeout)) as client:
        resp = await client.post(f"{url}/task/embed", json=payload)
        resp.raise_for_status()
        data = resp.json()
    return data.get("embeddings") or []


async def embed_query(
    text: str,
    model: str | None = None,
    *,
    base_url: str | None = None,
    read_timeout: float | None = None,
) -> list[float]:
    """Embed a single string. Convenience wrapper over embed_texts()."""
    vectors = await embed_texts([text], model=model, base_url=base_url, read_timeout=read_timeout)
    return vectors[0] if vectors else []


async def run_task_stream(
    task_type: str = "generate_text",
    model_preference: str = "fast",
    *,
    system_prompt: str = "",
    user_prompt: str = "",
    model: str | None = None,
    max_retries: int = 1,
    context_budget: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    runtime: str | None = None,
    base_url: str | None = None,
    connect_timeout: float | None = None,
    read_timeout: float | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream a task from Engine. Yields parsed NDJSON chunks.

    Each yielded dict is one of:
      - {"type": "delta", "content": "..."}   — generated text chunk
      - {"type": "done",  "meta": { ... }}    — stream finished
      - {"type": "error", "code": "...", "message": "..."}

    Only task_type="generate_text" is supported in streaming mode.
    Consumers should assemble the full text by concatenating deltas and
    use the done chunk's meta for tokens/latency/cost telemetry.
    """
    url = base_url or ENGINE_URL
    payload: dict[str, Any] = {
        "task_type": task_type,
        "model_preference": model_preference,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "max_retries": max_retries,
    }
    if model is not None:
        payload["model"] = model
    if context_budget is not None:
        payload["context_budget"] = context_budget
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if runtime is not None:
        payload["runtime"] = runtime

    async with httpx.AsyncClient(timeout=_timeout(connect_timeout, read_timeout)) as client:
        async with client.stream("POST", f"{url}/task/stream", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("sanctum-engine-client: unparseable stream line: %s", e)


async def run_tasks_parallel(
    task_specs: list[dict[str, Any]],
    *,
    base_url: str | None = None,
    connect_timeout: float | None = None,
    read_timeout: float | None = None,
    return_exceptions: bool = True,
) -> list[tuple[Any, int] | BaseException]:
    """Run many tasks concurrently against Engine.

    Each entry in `task_specs` is a kwargs dict for `run_task(...)` — e.g.
    `{"task_type": "generate_text", "model": "qwen3:32b", "user_prompt": "…"}`.

    Used for multi-model compare (Consilium's core feature, also llm-counsil's
    parallel-query pattern). Engine itself processes each task serially inside
    the single runtime; this helper just overlaps the HTTP calls.

    Returns a list aligned with `task_specs`. When `return_exceptions=True`
    (default), failed tasks come back as the exception object so a single
    model error doesn't fail the whole batch.
    """
    coros = [
        run_task(
            base_url=base_url,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            **spec,
        )
        for spec in task_specs
    ]
    return await asyncio.gather(*coros, return_exceptions=return_exceptions)


async def translate(
    text: str,
    source_lang_label: str,
    target_lang_label: str,
    model: str | None = None,
    *,
    base_url: str | None = None,
    read_timeout: float | None = None,
) -> tuple[str, int]:
    """Translate `text` from one language to another via Engine.

    Compatible with translachat's in-tree engine_client.translate signature.
    Returns (translated_text, latency_ms).
    """
    system_prompt = (
        f"Translate the following text from {source_lang_label} to {target_lang_label}. "
        f"Output ONLY the translation — no commentary, no quotes, no language labels."
    )
    return await run_task(
        task_type="generate_text",
        model_preference="translation",
        system_prompt=system_prompt,
        user_prompt=text,
        model=model,
        max_retries=2,
        base_url=base_url,
        read_timeout=read_timeout,
    )
