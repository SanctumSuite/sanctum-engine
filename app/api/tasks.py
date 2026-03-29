"""Task execution API endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..db import get_db
from ..models.schemas import TaskRequest, TaskResponse, EmbedRequest, EmbedResponse
from ..core import task_runner
from ..runtimes.ollama import OllamaRuntime

router = APIRouter(tags=["tasks"])

# Singleton runtime — created once, reused for all requests
_runtime = OllamaRuntime()


@router.post("/task", response_model=TaskResponse)
async def execute_task(request: TaskRequest, db: Session = Depends(get_db)):
    """Execute an LLM task with context management, validation, and retry logic."""
    return await task_runner.run_task(request, _runtime, db)


@router.post("/task/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    """Generate embeddings for a batch of texts."""
    from ..config import settings
    model = request.model or settings.embedding_model

    embeddings = await _runtime.embed(model, request.texts)
    dimensions = len(embeddings[0]) if embeddings else 0

    return EmbedResponse(
        embeddings=embeddings,
        model=model,
        dimensions=dimensions,
    )
