"""Test fixtures for Sanctum Engine tests."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models.db import ModelRegistry


@pytest.fixture
def db_session():
    """In-memory SQLite session for unit tests (no PostgreSQL needed)."""
    # Use SQLite for unit tests — PostgreSQL only needed for integration
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Seed with test models
    session.add_all([
        ModelRegistry(
            name="test-reasoning:latest",
            parameter_size="10B",
            quantization="Q4_K_M",
            context_length=32768,
            safe_context_limit=22000,
            capabilities=["reasoning", "json"],
            size_bytes=6_000_000_000,
            runtime="ollama",
            is_available=True,
            priority=70,
            config={},
        ),
        ModelRegistry(
            name="test-fast:latest",
            parameter_size="3B",
            quantization="Q4_K_M",
            context_length=8192,
            safe_context_limit=6000,
            capabilities=["fast", "json"],
            size_bytes=2_000_000_000,
            runtime="ollama",
            is_available=True,
            priority=50,
            config={},
        ),
        ModelRegistry(
            name="test-embed:latest",
            parameter_size="137M",
            quantization="F16",
            context_length=2048,
            safe_context_limit=1800,
            capabilities=["embedding"],
            size_bytes=274_000_000,
            runtime="ollama",
            is_available=True,
            priority=90,
            config={},
        ),
        ModelRegistry(
            name="qwen3.5:latest",
            parameter_size="9.7B",
            quantization="Q4_K_M",
            context_length=262144,
            safe_context_limit=32768,
            capabilities=["reasoning", "fast", "json"],
            size_bytes=6_500_000_000,
            runtime="ollama",
            is_available=True,
            priority=60,
            config={"append_to_prompt": "/nothink", "strip_thinking_tags": True},
        ),
    ])
    session.commit()

    yield session
    session.close()
