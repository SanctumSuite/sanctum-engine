"""Task execution API endpoints."""
import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..db import get_db
from ..config import settings
from ..models.schemas import TaskRequest, TaskResponse, EmbedRequest, EmbedResponse
from ..core import task_runner
from ..runtimes.base import LLMRuntime
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
