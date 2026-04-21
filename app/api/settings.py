"""Engine settings API — OpenRouter key management, account info, pricing."""
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/settings", tags=["settings"])


class SetKeyRequest(BaseModel):
    api_key: str


@router.post("/openrouter-key")
async def set_openrouter_key(body: SetKeyRequest):
    """Set OpenRouter API key at runtime (does not persist across restarts)."""
    settings.openrouter_api_key = body.api_key

    # Re-initialize the runtime with the new key
    from .tasks import _runtimes, get_runtime
    if "openrouter" in _runtimes:
        del _runtimes["openrouter"]

    # Test connectivity
    try:
        runtime = get_runtime("openrouter")
        healthy = await runtime.is_healthy()
    except Exception as e:
        return {"status": "error", "message": str(e), "connected": False}

    if healthy:
        # Trigger model discovery
        from ..main import _discover_openrouter_models
        await _discover_openrouter_models(runtime)
        return {"status": "ok", "connected": True, "message": "OpenRouter connected and models discovered"}
    else:
        return {"status": "error", "connected": False, "message": "API key set but connection failed"}


@router.get("/openrouter-account")
async def openrouter_account():
    """Get OpenRouter account balance and usage."""
    if not settings.openrouter_api_key:
        return {"configured": False}

    from .tasks import get_runtime
    try:
        runtime = get_runtime("openrouter")
        info = await runtime.get_account_info()
        return {"configured": True, **info}
    except Exception as e:
        return {"configured": False, "error": str(e)}


@router.get("/openrouter-pricing")
async def openrouter_pricing():
    """Get pricing for all OpenRouter models."""
    if not settings.openrouter_api_key:
        return {"models": []}

    from .tasks import get_runtime
    try:
        runtime = get_runtime("openrouter")
        pricing = await runtime.get_model_pricing()
        return {"models": pricing}
    except Exception as e:
        return {"models": [], "error": str(e)}
