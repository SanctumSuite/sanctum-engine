"""Model management API endpoints."""
import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..db import get_db
from ..models.db import ModelRegistry, TaskLog
from ..models.schemas import ModelInfo, ModelListResponse, ModelStats

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelListResponse)
def list_models(db: Session = Depends(get_db)):
    """List all registered models with capabilities."""
    models = db.query(ModelRegistry).order_by(ModelRegistry.priority.desc()).all()

    return ModelListResponse(
        models=[
            ModelInfo(
                name=m.name,
                parameter_size=m.parameter_size or "unknown",
                quantization=m.quantization or "unknown",
                context_length=m.context_length or 0,
                safe_context_limit=m.safe_context_limit or 0,
                capabilities=m.capabilities or [],
                size_gb=round((m.size_bytes or 0) / 1e9, 1),
                loaded=False,  # TODO: track loaded state via Ollama ps
            )
            for m in models
        ]
    )


@router.get("/{model_name:path}/stats", response_model=ModelStats)
def model_stats(model_name: str, db: Session = Depends(get_db)):
    """Get performance statistics for a model."""
    model = db.query(ModelRegistry).filter(ModelRegistry.name == model_name).first()
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    # Query task logs for this model
    logs = db.query(TaskLog).filter(TaskLog.model_used == model_name).all()
    total = len(logs)
    if total == 0:
        return ModelStats(
            model=model_name,
            total_tasks=0,
            success_rate=0.0,
            avg_tokens_per_sec=0.0,
            avg_latency_ms=0.0,
            json_parse_failures=0,
            json_repair_successes=0,
            by_task_type={},
        )

    successes = sum(1 for l in logs if l.status == "success")
    failures_json = sum(1 for l in logs if l.error_code and "JSON" in l.error_code)
    repairs = sum(1 for l in logs if l.was_repaired)

    latencies = [l.latency_ms for l in logs if l.latency_ms]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    # Tokens per second
    tps_values = []
    for l in logs:
        if l.tokens_out and l.latency_ms and l.latency_ms > 0:
            tps_values.append(l.tokens_out / (l.latency_ms / 1000))
    avg_tps = sum(tps_values) / len(tps_values) if tps_values else 0

    # By task type
    by_type: dict[str, dict] = {}
    for l in logs:
        tt = l.task_type
        if tt not in by_type:
            by_type[tt] = {"count": 0, "success": 0}
        by_type[tt]["count"] += 1
        if l.status == "success":
            by_type[tt]["success"] += 1

    by_type_final = {
        tt: {"count": d["count"], "success_rate": round(d["success"] / d["count"], 2)}
        for tt, d in by_type.items()
    }

    return ModelStats(
        model=model_name,
        total_tasks=total,
        success_rate=round(successes / total, 3),
        avg_tokens_per_sec=round(avg_tps, 1),
        avg_latency_ms=round(avg_latency),
        json_parse_failures=failures_json,
        json_repair_successes=repairs,
        by_task_type=by_type_final,
    )
