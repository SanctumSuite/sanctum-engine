"""Abstract runtime interface for LLM backends."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class RuntimeResponse:
    """Response from an LLM runtime."""
    content: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    model: str
    cost_usd: float = 0.0  # Actual cost in USD (OpenRouter provides this; 0 for local)


@dataclass
class StreamDelta:
    """One chunk of streamed text from a runtime."""
    content: str


@dataclass
class StreamDone:
    """Terminal event on a runtime stream: totals + metadata."""
    tokens_in: int
    tokens_out: int
    latency_ms: int
    model: str
    cost_usd: float = 0.0


@dataclass
class RuntimeModelInfo:
    """Model information from a runtime."""
    name: str
    parameter_size: str
    quantization: str
    context_length: int
    size_bytes: int
    families: list[str]


class LLMRuntime(ABC):
    """Abstract interface for LLM runtimes (Ollama, MLX, llama.cpp, vLLM)."""

    @abstractmethod
    async def generate(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        num_ctx: int,
    ) -> RuntimeResponse:
        """Generate a completion (single-shot)."""
        ...

    async def generate_stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        num_ctx: int,
    ) -> AsyncIterator["StreamDelta | StreamDone"]:
        """Generate a completion as a stream of deltas, ending with StreamDone.

        Default implementation raises NotImplementedError — runtimes that
        support streaming override this.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")
        yield  # pragma: no cover  (makes this an async generator for typing)

    @abstractmethod
    async def embed(
        self,
        model: str,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for texts."""
        ...

    @abstractmethod
    async def list_models(self) -> list[RuntimeModelInfo]:
        """List available models."""
        ...

    @abstractmethod
    async def model_info(self, model: str) -> RuntimeModelInfo | None:
        """Get detailed info about a specific model."""
        ...

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Check if the runtime is reachable."""
        ...
