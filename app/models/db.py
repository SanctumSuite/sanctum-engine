import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, BigInteger, Text,
    DateTime, JSON,
)
from sqlalchemy.dialects.postgresql import UUID

from ..db import Base


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    parameter_size = Column(String(50))
    quantization = Column(String(50))
    context_length = Column(Integer)
    safe_context_limit = Column(Integer)
    capabilities = Column(JSON, nullable=False, default=list)  # stored as JSON array for SQLite compat
    size_bytes = Column(BigInteger)
    runtime = Column(String(50), default="ollama")
    is_available = Column(Boolean, default=True)
    priority = Column(Integer, default=50)
    config = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))


class TaskLog(Base):
    __tablename__ = "task_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_type = Column(String(50), nullable=False)
    model_requested = Column(String(255))
    model_used = Column(String(255), nullable=False)
    runtime = Column(String(50), nullable=False)

    tokens_in = Column(Integer)
    tokens_out = Column(Integer)
    context_utilization = Column(Float)

    status = Column(String(20), nullable=False)
    error_code = Column(String(50))
    error_detail = Column(Text)

    attempts = Column(Integer, default=1)
    was_repaired = Column(Boolean, default=False)

    latency_ms = Column(Integer)
    temperature = Column(Float)

    comparison_group_id = Column(UUID(as_uuid=True))

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
