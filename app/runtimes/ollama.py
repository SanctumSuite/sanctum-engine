"""Ollama runtime implementation.

Uses the native Ollama API (not OpenAI-compatible) for access to:
- num_ctx control
- num_predict (max output tokens)
- Token count reporting (prompt_eval_count, eval_count)
- Thinking content handling
"""
import logging
import re
import time

import httpx

from ..config import settings
from .base import LLMRuntime, RuntimeResponse, RuntimeModelInfo

logger = logging.getLogger(__name__)


class OllamaRuntime(LLMRuntime):

    def __init__(self, host: str | None = None):
        self._host = (host or settings.ollama_host).rstrip("/")
        self._timeout = httpx.Timeout(
            connect=settings.llm_connect_timeout,
            read=settings.llm_read_timeout,
            write=30.0,
            pool=60.0,
        )

    async def generate(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        num_ctx: int,
    ) -> RuntimeResponse:
        """Call Ollama native chat API."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": num_ctx,
            },
        }

        logger.info(f"Ollama call: model={model}, num_ctx={num_ctx}, num_predict={max_tokens}")
        start = time.monotonic()

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                resp = await client.post(f"{self._host}/api/chat", json=payload)
                resp.raise_for_status()
            except httpx.ReadTimeout:
                logger.error(f"Ollama timeout: model={model}, num_ctx={num_ctx}")
                raise
            except httpx.ConnectError:
                logger.error(f"Cannot connect to Ollama at {self._host}")
                raise
            except httpx.HTTPStatusError as e:
                logger.error(f"Ollama HTTP error: {e.response.status_code} {e.response.text[:200]}")
                raise

        elapsed_ms = int((time.monotonic() - start) * 1000)
        data = resp.json()
        content = data.get("message", {}).get("content", "")

        # Extract token counts from Ollama response
        tokens_in = data.get("prompt_eval_count", 0)
        tokens_out = data.get("eval_count", 0)
        duration = data.get("total_duration", 0) / 1e9

        logger.info(
            f"Ollama done: {duration:.1f}s, prompt={tokens_in}, "
            f"completion={tokens_out}, latency={elapsed_ms}ms"
        )

        # Strip any residual thinking tags
        if "<think>" in content:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        if not content:
            logger.warning(f"Ollama returned empty content for model={model}")

        return RuntimeResponse(
            content=content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=elapsed_ms,
            model=model,
        )

    async def embed(
        self,
        model: str,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings using Ollama embed API."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            # Ollama supports batch embedding via /api/embed
            resp = await client.post(
                f"{self._host}/api/embed",
                json={"model": model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("embeddings", [])

    async def list_models(self) -> list[RuntimeModelInfo]:
        """List models available in Ollama."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                resp = await client.get(f"{self._host}/api/tags")
                resp.raise_for_status()
            except Exception as e:
                logger.error(f"Failed to list Ollama models: {e}")
                return []

        models = []
        for m in resp.json().get("models", []):
            details = m.get("details", {})
            models.append(RuntimeModelInfo(
                name=m["name"],
                parameter_size=details.get("parameter_size", "unknown"),
                quantization=details.get("quantization_level", "unknown"),
                context_length=0,  # requires /api/show per model
                size_bytes=m.get("size", 0),
                families=details.get("families", []),
            ))
        return models

    async def model_info(self, model: str) -> RuntimeModelInfo | None:
        """Get detailed model info including context length."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                resp = await client.post(
                    f"{self._host}/api/show",
                    json={"name": model},
                )
                resp.raise_for_status()
            except Exception as e:
                logger.warning(f"Failed to get info for model {model}: {e}")
                return None

        data = resp.json()
        details = data.get("details", {})
        model_info_dict = data.get("model_info", {})

        # Find context length from model_info keys
        context_length = 0
        for key, value in model_info_dict.items():
            if "context_length" in key and not key.endswith("original_context_length"):
                context_length = int(value)
                break

        parameter_size = details.get("parameter_size", "unknown")
        # Also check model_info for parameter count
        if parameter_size == "unknown":
            param_count = model_info_dict.get("general.parameter_count", 0)
            if param_count:
                if param_count > 1e9:
                    parameter_size = f"{param_count / 1e9:.1f}B"
                elif param_count > 1e6:
                    parameter_size = f"{param_count / 1e6:.0f}M"

        return RuntimeModelInfo(
            name=model,
            parameter_size=parameter_size,
            quantization=details.get("quantization_level", "unknown"),
            context_length=context_length,
            size_bytes=0,
            families=details.get("families", []),
        )

    async def is_healthy(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._host}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
