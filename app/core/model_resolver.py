"""Model resolution: maps capability preferences to actual models.

Handles capability-based lookup, fallback chains, and model-specific
configuration (e.g., qwen3.5 /nothink quirk).
"""
import logging
from dataclasses import dataclass

from sqlalchemy.orm import Session

from ..models.db import ModelRegistry

logger = logging.getLogger(__name__)

# Default capability mappings used when auto-discovering models.
# Maps model family prefixes to their capabilities.
DEFAULT_CAPABILITIES: dict[str, list[str]] = {
    "qwen": ["reasoning", "fast", "json"],
    "ministral": ["reasoning", "fast", "json"],
    "nomic-embed": ["embedding"],
    "deepseek-ocr": ["vision", "ocr"],
    "glm-ocr": ["vision", "ocr"],
    "llava": ["vision"],
    "translategemma": ["translation"],
    "gemma": ["reasoning", "fast", "json"],
    "olmo": ["reasoning"],
    "nemotron": ["reasoning"],
}

# Safe context limits by model family.
# These are the actual enforced limits, not advertised.
DEFAULT_SAFE_LIMITS: dict[str, int] = {
    "qwen": 32768,
    "ministral": 32768,
    "nomic-embed": 1800,
    "deepseek-ocr": 6000,
    "glm-ocr": 6000,
    "llava": 8192,
    "translategemma": 6000,
    "gemma": 32768,
    "olmo": 32768,
    "nemotron": 32768,
}

# Model-specific config applied automatically
MODEL_CONFIGS: dict[str, dict] = {
    "qwen3.5": {
        "append_to_prompt": "/nothink",
        "strip_thinking_tags": True,
    },
}


@dataclass
class ResolvedModel:
    """Result of model resolution."""
    name: str
    safe_context_limit: int
    runtime: str
    config: dict


def infer_capabilities(model_name: str) -> list[str]:
    """Infer capabilities from model name using family prefix matching."""
    name_lower = model_name.lower().split(":")[0]  # strip tag
    for prefix, caps in DEFAULT_CAPABILITIES.items():
        if name_lower.startswith(prefix):
            return caps
    return ["reasoning", "fast"]  # sensible default


def infer_safe_context_limit(model_name: str, advertised_context: int) -> int:
    """Infer safe context limit from model name and advertised context."""
    name_lower = model_name.lower().split(":")[0]
    for prefix, limit in DEFAULT_SAFE_LIMITS.items():
        if name_lower.startswith(prefix):
            return min(limit, int(advertised_context * 0.7))
    # Default: 70% of advertised, capped at 32K
    return min(int(advertised_context * 0.7), 32768)


def get_model_config(model_name: str) -> dict:
    """Get model-specific configuration overrides."""
    name_lower = model_name.lower().split(":")[0]
    for prefix, config in MODEL_CONFIGS.items():
        if name_lower.startswith(prefix):
            return config
    return {}


def resolve_model(
    db: Session,
    preference: str,
    runtime: str = "ollama",
) -> ResolvedModel | None:
    """Resolve a model preference to an actual model.

    Preference can be:
    - A capability name: "reasoning", "fast", "vision", "embedding", "translation"
    - A specific model name: "qwen3.5:latest"

    Returns the best matching model, or None if nothing matches.
    """
    # First: check if preference is a specific model name
    specific = db.query(ModelRegistry).filter(
        ModelRegistry.name == preference,
        ModelRegistry.is_available == True,
    ).first()

    if specific:
        return ResolvedModel(
            name=specific.name,
            safe_context_limit=specific.safe_context_limit or 32768,
            runtime=specific.runtime or runtime,
            config=specific.config or get_model_config(specific.name),
        )

    # Otherwise: treat as capability and find best match.
    # capabilities is stored as JSON array, so we query all available models
    # and filter in Python for cross-database compatibility.
    all_models = db.query(ModelRegistry).filter(
        ModelRegistry.is_available == True,
        ModelRegistry.runtime == runtime,
    ).order_by(ModelRegistry.priority.desc()).all()

    models = [m for m in all_models if preference in (m.capabilities or [])]

    if not models:
        logger.warning(f"No model found for preference '{preference}' on runtime '{runtime}'")
        return None

    model = models[0]
    return ResolvedModel(
        name=model.name,
        safe_context_limit=model.safe_context_limit or 32768,
        runtime=model.runtime or runtime,
        config=model.config or get_model_config(model.name),
    )


def resolve_model_with_fallback(
    db: Session,
    preference: str,
    runtime: str = "ollama",
) -> ResolvedModel | None:
    """Resolve model with fallback to any available model.

    If the preferred capability/model isn't available,
    falls back to any available model with JSON capability.
    """
    result = resolve_model(db, preference, runtime)
    if result:
        return result

    # Fallback: any available model with json/reasoning capability
    for fallback_cap in ["json", "reasoning", "fast"]:
        result = resolve_model(db, fallback_cap, runtime)
        if result:
            logger.warning(f"Falling back from '{preference}' to '{result.name}' ({fallback_cap})")
            return result

    # Last resort: any available model
    any_model = db.query(ModelRegistry).filter(
        ModelRegistry.is_available == True,
        ModelRegistry.runtime == runtime,
    ).order_by(ModelRegistry.priority.desc()).first()

    if any_model:
        logger.warning(f"Last-resort fallback from '{preference}' to '{any_model.name}'")
        return ResolvedModel(
            name=any_model.name,
            safe_context_limit=any_model.safe_context_limit or 32768,
            runtime=any_model.runtime or runtime,
            config=any_model.config or get_model_config(any_model.name),
        )

    return None
