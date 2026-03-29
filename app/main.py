"""Sanctum Engine — Local LLM Service.

Startup sequence:
1. Connect to PostgreSQL, create tables
2. Connect to Ollama, verify healthy
3. Auto-discover models, populate registry
4. Start FastAPI server
"""
import logging
import sys

from fastapi import FastAPI

from .config import settings
from .db import init_db, SessionLocal
from .api import tasks, models, health

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sanctum Engine",
    version="0.1.0",
    description="Local LLM service layer for Sanctum applications.",
)

# Register routers
app.include_router(tasks.router)
app.include_router(models.router)
app.include_router(health.router)


@app.on_event("startup")
async def startup():
    """Initialize database and discover models."""
    logger.info("Sanctum Engine starting...")

    # 1. Create database tables
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database init failed: {e}")
        raise

    # 2. Check Ollama connectivity
    from .runtimes.ollama import OllamaRuntime
    runtime = OllamaRuntime()
    ollama_ok = await runtime.is_healthy()

    if ollama_ok:
        logger.info(f"Ollama connected at {settings.ollama_host}")
    else:
        logger.warning(f"Ollama not reachable at {settings.ollama_host} — engine will start but tasks will fail")

    # 3. Auto-discover and register models
    if ollama_ok:
        await _discover_models(runtime)

    logger.info("Sanctum Engine ready")


async def _discover_models(runtime):
    """Query Ollama for available models and register them."""
    from .models.db import ModelRegistry
    from .core.model_resolver import infer_capabilities, infer_safe_context_limit, get_model_config

    models_list = await runtime.list_models()
    logger.info(f"Found {len(models_list)} models in Ollama")

    db = SessionLocal()
    try:
        for model in models_list:
            # Get detailed info for context length
            info = await runtime.model_info(model.name)
            context_length = info.context_length if info else 0

            capabilities = infer_capabilities(model.name)
            safe_limit = infer_safe_context_limit(model.name, context_length) if context_length else 32768
            config = get_model_config(model.name)

            # Assign priority based on model size and capability.
            # For reasoning/analysis: prefer mid-size models (8-32B) that balance quality and speed.
            # Very large models (>50B) are slower and only used when explicitly requested.
            priority = 50
            size_gb = (model.size_bytes or 0) / 1e9
            if "embedding" in capabilities:
                priority = 90
            elif "ocr" in capabilities or "vision" in capabilities:
                priority = 80
            elif size_gb > 50:
                priority = 30  # very large models: available but not default
            elif size_gb > 15:
                priority = 60  # medium-large models: good for reasoning
            elif size_gb > 3:
                priority = 70  # sweet-spot models (8-15B): best balance

            # Upsert
            existing = db.query(ModelRegistry).filter(ModelRegistry.name == model.name).first()
            if existing:
                existing.parameter_size = model.parameter_size
                existing.quantization = model.quantization
                existing.context_length = context_length
                existing.safe_context_limit = safe_limit
                existing.capabilities = capabilities
                existing.size_bytes = model.size_bytes
                existing.is_available = True
                existing.priority = priority
                existing.config = config
            else:
                db.add(ModelRegistry(
                    name=model.name,
                    parameter_size=model.parameter_size,
                    quantization=model.quantization,
                    context_length=context_length,
                    safe_context_limit=safe_limit,
                    capabilities=capabilities,
                    size_bytes=model.size_bytes,
                    runtime="ollama",
                    is_available=True,
                    priority=priority,
                    config=config,
                ))

            logger.info(f"  {model.name}: {model.parameter_size} {model.quantization}, "
                        f"ctx={context_length}, safe={safe_limit}, caps={capabilities}")

        db.commit()
    except Exception as e:
        logger.error(f"Model discovery failed: {e}")
        db.rollback()
    finally:
        db.close()
