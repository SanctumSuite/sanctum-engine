"""Pydantic models for API request/response."""
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# --- Task API ---

class ChunkingConfig(BaseModel):
    enabled: bool = False
    chunk_tokens: int = 2000
    overlap_tokens: int = 200
    merge_strategy: str = "concatenate_arrays"


class TaskRequest(BaseModel):
    task_type: str = Field(..., description="extract_json, generate_text, embed, vision, translate, rerank")
    model_preference: str = Field("fast", description="reasoning, fast, vision, embedding, translation, or model name")
    system_prompt: str = ""
    user_prompt: str = ""
    output_schema: dict[str, Any] | None = Field(None, description="JSON Schema for output validation")
    max_retries: int = Field(3, ge=1, le=5)
    context_budget: int | None = Field(None, description="Max tokens for this task")
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, description="Max output tokens")
    chunking: ChunkingConfig | None = None


class TaskMeta(BaseModel):
    task_id: UUID
    model_used: str
    runtime: str
    tokens_in: int
    tokens_out: int
    context_window: int
    context_utilization: float
    latency_ms: int
    attempts: int
    temperature: float


class AttemptError(BaseModel):
    attempt: int
    error: str
    raw_length: int


class TaskError(BaseModel):
    code: str
    message: str
    attempts: list[AttemptError] = []


class TaskResponse(BaseModel):
    status: str  # "success" or "error"
    result: Any | None = None
    error: TaskError | None = None
    meta: TaskMeta


# --- Embeddings API ---

class EmbedRequest(BaseModel):
    texts: list[str]
    model: str | None = None


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dimensions: int


# --- Rerank API ---

class RerankCandidate(BaseModel):
    id: str
    text: str


class RerankRequest(BaseModel):
    query: str
    candidates: list[RerankCandidate]
    top_k: int = 10


class RerankResult(BaseModel):
    id: str
    score: float
    rank: int


class RerankResponse(BaseModel):
    rankings: list[RerankResult]


# --- Models API ---

class ModelInfo(BaseModel):
    name: str
    parameter_size: str
    quantization: str
    context_length: int
    safe_context_limit: int
    capabilities: list[str]
    size_gb: float
    loaded: bool


class ModelListResponse(BaseModel):
    models: list[ModelInfo]


class ModelStats(BaseModel):
    model: str
    total_tasks: int
    success_rate: float
    avg_tokens_per_sec: float
    avg_latency_ms: float
    json_parse_failures: int
    json_repair_successes: int
    by_task_type: dict[str, dict[str, Any]]


# --- Health API ---

class MemoryInfo(BaseModel):
    system_total_gb: float
    estimated_available_gb: float
    models_loaded_gb: float


class HealthStats(BaseModel):
    tasks_completed_24h: int
    tasks_failed_24h: int
    avg_latency_ms: float
    error_rate: float


class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    models_loaded: list[str]
    models_available: list[str]
    memory: MemoryInfo
    stats: HealthStats
