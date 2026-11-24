"""The models a user manages from the chat UI: add, list, check, remove."""

import re
import uuid

from fastapi import HTTPException

from app.agents import user_models
from app.agents.mcp_guard import McpPolicyError, check_url
from app.core.config import settings
from app.db import user_model_repository
from app.services import model_check

KINDS = ("openrouter", "endpoint")
_MODEL_NAME = re.compile(r"^\S{1,200}$")
MIN_CONTEXT, MAX_CONTEXT = 4_000, 2_000_000


def _bad(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


async def list_models(user_id: str) -> dict:
    rows = await user_model_repository.list_models(user_id)
    return {
        "models": [user_models.describe(r) for r in rows],
        "catalog": [user_models.catalog_entry(r) for r in rows] if settings.USER_MODELS_ENABLED else [],       # for the model picker
        "keys_supported": user_models.secrets_configured(),
        "max_models": settings.USER_MODELS_MAX,
        "enabled": settings.USER_MODELS_ENABLED,
    }


async def catalog(user_id: str) -> list[dict]:
    """The user's models in the shape of the model picker's list."""
    if not settings.USER_MODELS_ENABLED:
        return []
    return [user_models.catalog_entry(r) for r in await user_model_repository.list_models(user_id)]


async def add_model(user_id: str, name: str, kind: str, model: str, base_url: str | None, api_key: str | None,
                    context_window: int, send_reasoning: bool) -> dict:
    if not settings.USER_MODELS_ENABLED:
        raise HTTPException(status_code=403, detail="Adding models is turned off.")
    name, model = name.strip(), model.strip()
    if not 1 <= len(name) <= 40:
        raise _bad("Give the model a name of 1 to 40 characters.")
    if kind not in KINDS:
        raise _bad(f"Kind must be one of: {', '.join(KINDS)}.")
    if not _MODEL_NAME.match(model):
        raise _bad("The model name is what the API is asked for, e.g. 'mistralai/mistral-large-4-0'; it has no spaces.")
    if not MIN_CONTEXT <= context_window <= MAX_CONTEXT:
        raise _bad(f"The context window must be between {MIN_CONTEXT:,} and {MAX_CONTEXT:,} tokens.")
    api_key_enc = None
    if kind == "endpoint":
        base_url = (base_url or "").strip()
        if not base_url:
            raise _bad("An own endpoint needs a base URL, e.g. https://api.example.com/v1.")
        try:
            await check_url(base_url)
        except McpPolicyError as e:
            raise _bad(str(e)) from None
        if api_key:
            if len(api_key) > 512:
                raise _bad("The API key is too long.")
            try:
                api_key_enc = user_models.encrypt_key(api_key)
            except user_models.UserModelError as e:
                raise _bad(str(e)) from None
    else:
        base_url, send_reasoning = None, True            # the server's own OpenRouter key and endpoint
    if len(await user_model_repository.list_models(user_id)) >= settings.USER_MODELS_MAX:
        raise _bad(f"At most {settings.USER_MODELS_MAX} models.")
    row = await user_model_repository.create_model(user_id, name, kind, model, base_url, api_key_enc, context_window, send_reasoning)
    if row is None:
        raise HTTPException(status_code=409, detail=f"You already have a model called {name!r}.")
    return user_models.describe(row)


def _uuid_or_404(model_id: str) -> str:
    try:
        return str(uuid.UUID(model_id))
    except ValueError:
        raise HTTPException(status_code=404, detail="Model not found") from None


async def remove_model(user_id: str, model_id: str) -> None:
    model_id = _uuid_or_404(model_id)
    if not await user_model_repository.delete_model(user_id, model_id):
        raise HTTPException(status_code=404, detail="Model not found")
    user_models.forget(user_id, user_models.PREFIX + model_id)


async def check_model(user_id: str, model_id: str) -> dict:
    model_id = _uuid_or_404(model_id)
    row = await user_model_repository.get_model(user_id, model_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Model not found")
    try:
        custom = user_models._custom(row)
    except user_models.UserModelError as e:
        raise _bad(str(e)) from None
    status, result, vision = await model_check.check(custom)
    await user_model_repository.save_check(user_id, model_id, status, result, vision)
    user_models.forget(user_id, user_models.PREFIX + model_id)               # the saved vision flag may have changed
    return user_models.describe(await user_model_repository.get_model(user_id, model_id))
