"""OpenRouter runtime implementation.

Uses OpenRouter's OpenAI-compatible API for cloud model access.
Embeddings are NOT supported — use Ollama for embeddings.
"""
import logging
import time

import httpx

from .base import LLMRuntime, RuntimeResponse, RuntimeModelInfo

logger = logging.getLogger(__name__)

# Curated model families and their capabilities
OPENROUTER_CAPABILITIES: dict[str, list[str]] = {
    "anthropic/claude": ["reasoning", "fast", "json"],
    "google/gemini": ["reasoning", "fast", "json"],
    "openai/gpt-4": ["reasoning", "json"],
    "openai/gpt-3.5": ["fast", "json"],
    "meta-llama/llama": ["reasoning", "fast", "json"],
    "mistralai/": ["reasoning", "fast", "json"],
    "deepseek/": ["reasoning", "json"],
    "qwen/": ["reasoning", "fast", "json"],
}


class OpenRouterRuntime(LLMRuntime):

    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=60.0)

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": "http://localhost:5173",
            "X-Title": "Sanctum Analyst",
            "Content-Type": "application/json",
        }

    async def generate(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        num_ctx: int,
    ) -> RuntimeResponse:
        """Call OpenRouter chat completions API."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        logger.info(f"OpenRouter call: model={model}, max_tokens={max_tokens}")
        start = time.monotonic()

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
                resp.raise_for_status()
            except httpx.ReadTimeout:
                logger.error(f"OpenRouter timeout: model={model}")
                raise
            except httpx.ConnectError:
                logger.error(f"Cannot connect to OpenRouter at {self._base_url}")
                raise
            except httpx.HTTPStatusError as e:
                logger.error(f"OpenRouter HTTP error: {e.response.status_code} {e.response.text[:300]}")
                raise

        elapsed_ms = int((time.monotonic() - start) * 1000)
        data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)
        cost_usd = float(usage.get("cost", 0) or 0)

        logger.info(
            f"OpenRouter done: prompt={tokens_in}, completion={tokens_out}, "
            f"cost=${cost_usd:.6f}, latency={elapsed_ms}ms, model={data.get('model', model)}"
        )

        if not content:
            logger.warning(f"OpenRouter returned empty content for model={model}")

        return RuntimeResponse(
            content=content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=elapsed_ms,
            model=data.get("model", model),
            cost_usd=cost_usd,
        )

    async def embed(
        self,
        model: str,
        texts: list[str],
    ) -> list[list[float]]:
        """Not supported — use Ollama for embeddings."""
        raise NotImplementedError(
            "OpenRouter does not support embeddings. Use Ollama with nomic-embed-text."
        )

    async def list_models(self) -> list[RuntimeModelInfo]:
        """List available models from OpenRouter."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                resp = await client.get(
                    f"{self._base_url}/models",
                    headers=self._headers(),
                )
                resp.raise_for_status()
            except Exception as e:
                logger.error(f"Failed to list OpenRouter models: {e}")
                return []

        models = []
        for m in resp.json().get("data", []):
            model_id = m.get("id", "")
            context = m.get("context_length", 0)
            # Estimate size from pricing tier (rough heuristic)
            pricing = m.get("pricing", {})
            prompt_price = float(pricing.get("prompt", "0") or "0")

            models.append(RuntimeModelInfo(
                name=model_id,
                parameter_size="cloud",
                quantization="none",
                context_length=context,
                size_bytes=0,
                families=[],
            ))
        return models

    async def model_info(self, model: str) -> RuntimeModelInfo | None:
        """Get info for a specific OpenRouter model."""
        all_models = await self.list_models()
        for m in all_models:
            if m.name == model:
                return m
        return None

    async def is_healthy(self) -> bool:
        """Check if OpenRouter API is reachable."""
        if not self._api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self._base_url}/models",
                    headers=self._headers(),
                )
                return resp.status_code == 200
        except Exception:
            return False

    async def get_account_info(self) -> dict:
        """Get account balance and usage from OpenRouter."""
        if not self._api_key:
            return {"error": "No API key configured"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self._base_url}/key",
                    headers=self._headers(),
                )
                resp.raise_for_status()
                data = resp.json().get("data", {})
                return {
                    "limit": data.get("limit"),
                    "usage": data.get("usage", 0),
                    "limit_remaining": data.get("limit_remaining"),
                    "usage_daily": data.get("usage_daily", 0),
                    "usage_monthly": data.get("usage_monthly", 0),
                    "is_free_tier": data.get("is_free_tier", False),
                }
        except Exception as e:
            logger.error(f"Failed to get OpenRouter account info: {e}")
            return {"error": str(e)}

    async def get_model_pricing(self) -> list[dict]:
        """Get pricing for all OpenRouter models."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(f"{self._base_url}/models")
                resp.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to get OpenRouter pricing: {e}")
            return []

        result = []
        for m in resp.json().get("data", []):
            pricing = m.get("pricing", {})
            prompt_per_token = float(pricing.get("prompt", "0") or "0")
            completion_per_token = float(pricing.get("completion", "0") or "0")
            result.append({
                "model": m.get("id", ""),
                "name": m.get("name", ""),
                "prompt_per_1m": round(prompt_per_token * 1_000_000, 2),
                "completion_per_1m": round(completion_per_token * 1_000_000, 2),
                "context_length": m.get("context_length", 0),
            })
        return result


def infer_openrouter_capabilities(model_id: str) -> list[str]:
    """Infer capabilities from an OpenRouter model ID."""
    model_lower = model_id.lower()
    for prefix, caps in OPENROUTER_CAPABILITIES.items():
        if model_lower.startswith(prefix):
            return caps
    return ["reasoning", "fast", "json"]
