"""Health check endpoint."""
import platform
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..db import get_db
from ..models.db import ModelRegistry, TaskLog
from ..models.schemas import HealthResponse, MemoryInfo, HealthStats
from ..runtimes.ollama import OllamaRuntime

router = APIRouter(tags=["health"])

_runtime = OllamaRuntime()


@router.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """Health check with system status."""
    ollama_ok = await _runtime.is_healthy()

    # Available models from registry
    models = db.query(ModelRegistry).filter(ModelRegistry.is_available == True).all()
    model_names = [m.name for m in models]

    # Task stats for last 24h
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    recent_logs = db.query(TaskLog).filter(TaskLog.created_at >= cutoff).all()
    completed = sum(1 for l in recent_logs if l.status == "success")
    failed = sum(1 for l in recent_logs if l.status == "error")
    latencies = [l.latency_ms for l in recent_logs if l.latency_ms]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    total = completed + failed
    error_rate = failed / total if total > 0 else 0

    # Memory (rough estimate for macOS)
    import os
    try:
        total_mem_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024 ** 3)
    except (ValueError, OSError):
        total_mem_gb = 128.0  # default assumption

    # Estimate model memory from sizes
    model_mem_gb = sum((m.size_bytes or 0) for m in models) / 1e9

    return HealthResponse(
        status="healthy" if ollama_ok else "degraded",
        ollama_connected=ollama_ok,
        models_loaded=[],  # TODO: query Ollama /api/ps
        models_available=model_names,
        memory=MemoryInfo(
            system_total_gb=round(total_mem_gb, 1),
            estimated_available_gb=round(total_mem_gb - model_mem_gb - 30, 1),  # rough: 30GB system overhead
            models_loaded_gb=round(model_mem_gb, 1),
        ),
        stats=HealthStats(
            tasks_completed_24h=completed,
            tasks_failed_24h=failed,
            avg_latency_ms=round(avg_latency),
            error_rate=round(error_rate, 3),
        ),
    )
