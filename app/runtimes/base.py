"""Abstract runtime interface for LLM backends."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RuntimeResponse:
    """Response from an LLM runtime."""
    content: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    model: str


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
        """Generate a completion."""
        ...

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
