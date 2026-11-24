"""Models users add themselves, and how a turn reaches them.

A user's model is picked like any other, by an id: `user:<uuid>`. That id stays in
the conversation, the request and the saved state; what it means (which model name to
send, to which endpoint, with which key) is looked up when the call is made. Two kinds:

* `openrouter`: a model id on OpenRouter, called with the server's own OpenRouter key.
* `endpoint`: the user's own OpenAI-compatible endpoint and API key (Ollama, vLLM,
  OpenAI, Together...). The key is stored encrypted (the same Fernet key the MCP
  headers use) and is read only here, at call time. It is never put in the run's
  `configurable` (that gets copied into traces) and never sent back to the browser.

A turn on an own endpoint must not quietly move to the server's fallback models when
the endpoint fails: that would spend the operator's credit and answer with a model the
user did not choose. See `no_fallback` in `info()`.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import settings
from app.db import user_model_repository

logger = logging.getLogger(__name__)

PREFIX = "user:"
CACHE_SECONDS = 60
DEFAULT_CONTEXT_WINDOW = 128_000


class UserModelError(Exception):
    """A problem the user should be told about (shown as is)."""


@dataclass
class CustomModel:
    id: str                    # "user:<uuid>"
    name: str
    kind: str                  # "openrouter" | "endpoint"
    model: str                 # the name sent to the API
    base_url: str
    api_key: str
    vision: bool
    context_window: int
    send_reasoning: bool


def is_user_model(model_id: str | None) -> bool:
    return bool(model_id) and model_id.startswith(PREFIX)


def raw_id(model_id: str) -> str | None:
    """The uuid part of `user:<uuid>`, or None if it is not one."""
    try:
        return str(uuid.UUID(model_id[len(PREFIX):]))
    except ValueError:
        return None


# ── the API key ────────────────────────────────────────────────────────────

def secrets_configured() -> bool:
    return bool(settings.MCP_SECRET_KEY)


def encrypt_key(api_key: str) -> str:
    if not secrets_configured():
        raise UserModelError("API keys cannot be stored: the server has no MCP_SECRET_KEY configured.")
    return Fernet(settings.MCP_SECRET_KEY.encode()).encrypt(api_key.encode()).decode()


def decrypt_key(token: str | None) -> str:
    if not token:
        return ""
    if not secrets_configured():
        raise UserModelError("This model's API key cannot be read: the server has no MCP_SECRET_KEY configured.")
    try:
        return Fernet(settings.MCP_SECRET_KEY.encode()).decrypt(token.encode()).decode()
    except InvalidToken:
        raise UserModelError("This model's API key cannot be decrypted (the server's secret key changed). "
                             "Delete the model and add it again.") from None


# ── looking a model up ─────────────────────────────────────────────────────

def _custom(row: dict) -> CustomModel:
    if row["kind"] == "openrouter":
        base_url, api_key = settings.OPENROUTER_BASE_URL, settings.OPENROUTER_API_KEY
    else:
        base_url, api_key = row["base_url"] or "", decrypt_key(row["api_key_enc"])
    return CustomModel(id=PREFIX + row["id"], name=row["name"], kind=row["kind"], model=row["model"], base_url=base_url,
                       api_key=api_key, vision=bool(row["vision"]), context_window=row["context_window"],
                       send_reasoning=bool(row["send_reasoning"]) or row["kind"] == "openrouter")


_cache: dict[tuple[str, str], tuple[float, CustomModel | None]] = {}


async def resolve(user_id: str, model_id: str) -> CustomModel | None:
    """The model, with its key, for the call; None if there is no such model for this user."""
    key = (user_id, model_id)
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < CACHE_SECONDS:
        return hit[1]
    uid = raw_id(model_id)
    row = await user_model_repository.get_model(user_id, uid) if uid else None
    custom = _custom(row) if row else None
    _cache[key] = (time.monotonic(), custom)
    return custom


def forget(user_id: str, model_id: str) -> None:
    _cache.pop((user_id, model_id), None)


async def info(user_id: str | None, model_id: str | None) -> dict:
    """What the rest of a turn needs to know about the chosen model, without its key: what goes into the run's
    `configurable` (model_vision, model_context_window, no_fallback). {} for a model that is not a user's."""
    if not user_id or not is_user_model(model_id):
        return {}
    uid = raw_id(model_id)
    row = await user_model_repository.get_model(user_id, uid) if uid else None
    if row is None:
        raise UserModelError("The model you picked was removed. Choose another one.")
    return {"model_vision": bool(row["vision"]), "model_context_window": int(row["context_window"]),
            "no_fallback": row["kind"] == "endpoint"}


def supports_vision(configurable: dict, builtin) -> bool:
    """Whether the turn's model can see images: what the model's own record says for a user's model, the
    built-in list (`builtin(model_id)`) otherwise."""
    model = configurable.get("model")
    if is_user_model(model):
        return bool(configurable.get("model_vision"))
    return builtin(model)


def describe(row: dict) -> dict:
    """A model as the browser sees it. Never the key."""
    result = row.get("last_result")
    if isinstance(result, str):
        result = json.loads(result)
    checked = row.get("last_checked_at")
    return {
        "id": PREFIX + row["id"],
        "name": row["name"],
        "kind": row["kind"],
        "model": row["model"],
        "base_url": row["base_url"],
        "has_key": bool(row["api_key_enc"]),
        "vision": bool(row["vision"]),
        "context_window": row["context_window"],
        "send_reasoning": bool(row["send_reasoning"]),
        "status": row["last_status"] or "unchecked",
        "checked_at": checked.isoformat() if checked else None,
        "result": result,
    }


def catalog_entry(row: dict) -> dict:
    """The model in the list the model picker shows (same shape as AVAILABLE_MODELS)."""
    return {
        "id": PREFIX + row["id"],
        "name": row["name"],
        "desc": "Your own model" + (" (OpenRouter)" if row["kind"] == "openrouter" else " (your endpoint)"),
        "inputPrice": "—",
        "outputPrice": "—",
        "context": f"{row['context_window'] // 1000}K",
        "contextWindow": row["context_window"],
        "size": "Undisclosed",
        "vision": bool(row["vision"]),
        "effortOptions": ["none", "high"],
        "huggingFaceId": None,
        "custom": True,
        "status": row["last_status"] or "unchecked",
    }
