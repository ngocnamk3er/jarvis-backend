"""HTTP client for jarvis-conversation-service — the service that now owns
the conversations/subagent_traces tables (see Chapter 2 decomposition:
jarvis-backend kept LangGraph's own checkpoint tables, since those are
mechanically tied to the graph object, and everything else moved out).

Every function here has the same name/signature app/db/repository.py used to
have, so callers barely changed — only the import + a network hop got added.
No retry/circuit breaker on purpose: a failure here should be loudly visible
right now (that's the point of doing this split before Chapter 3's
reliability patterns), not silently masked.
"""
from datetime import datetime

import httpx
from pydantic import BaseModel

from app.core.config import settings

_client: httpx.AsyncClient | None = None


def init_client() -> None:
    global _client
    _client = httpx.AsyncClient(
        base_url=f"{settings.CONVERSATION_SERVICE_URL}{settings.API_PREFIX}",
        headers={"X-Internal-Api-Key": settings.INTERNAL_API_KEY},
        timeout=10.0,
    )


async def close_client() -> None:
    if _client is not None:
        await _client.aclose()


def _get_client() -> httpx.AsyncClient:
    if _client is None:
        raise RuntimeError("conversation_client not initialised")
    return _client


class ConversationDTO(BaseModel):
    """Mirrors the shape app/db/repository.py's Conversation ORM object used
    to expose — callers access it the same way (conv.user_id,
    conv.context_tokens, conv.sandbox_session_id, ...)."""

    id: str
    user_id: str
    title: str
    last_model: str | None = None
    last_subagent_model: str | None = None
    context_tokens: int = 0
    sandbox_session_id: str | None = None
    created_at: datetime
    updated_at: datetime


def _conversation_dict(c: ConversationDTO) -> dict:
    return c.model_dump()


async def list_conversations(user_id: str) -> list[dict]:
    resp = await _get_client().get("/conversations", params={"user_id": user_id})
    resp.raise_for_status()
    return [_conversation_dict(ConversationDTO(**c)) for c in resp.json()]


async def create_conversation(title: str, user_id: str) -> dict:
    resp = await _get_client().post("/conversations", json={"title": title, "user_id": user_id})
    resp.raise_for_status()
    return _conversation_dict(ConversationDTO(**resp.json()))


async def get_conversation(thread_id: str) -> ConversationDTO | None:
    resp = await _get_client().get(f"/conversations/{thread_id}")
    resp.raise_for_status()
    data = resp.json()
    return ConversationDTO(**data) if data is not None else None


async def update_conversation_title(thread_id: str, title: str) -> None:
    resp = await _get_client().patch(f"/conversations/{thread_id}/title", json={"title": title})
    resp.raise_for_status()


async def touch_conversation(
    thread_id: str,
    model: str | None = None,
    subagent_model: str | None = None,
) -> None:
    resp = await _get_client().patch(
        f"/conversations/{thread_id}/touch",
        json={"model": model, "subagent_model": subagent_model},
    )
    resp.raise_for_status()


async def set_context_tokens(thread_id: str, tokens: int) -> None:
    resp = await _get_client().patch(f"/conversations/{thread_id}/context-tokens", json={"tokens": tokens})
    resp.raise_for_status()


async def set_sandbox_session_id(thread_id: str, session_id: str | None) -> None:
    resp = await _get_client().patch(
        f"/conversations/{thread_id}/sandbox-session", json={"session_id": session_id}
    )
    resp.raise_for_status()


async def delete_conversation(thread_id: str) -> None:
    resp = await _get_client().delete(f"/conversations/{thread_id}")
    resp.raise_for_status()


async def save_subagent_trace(conversation_id: str, tool_call_id: str, events: list[dict]) -> None:
    resp = await _get_client().put(
        f"/conversations/{conversation_id}/subagent-traces/{tool_call_id}",
        json={"events": events},
    )
    resp.raise_for_status()


async def get_subagent_traces(conversation_id: str) -> dict[str, list[dict]]:
    resp = await _get_client().get(f"/conversations/{conversation_id}/subagent-traces")
    resp.raise_for_status()
    return resp.json()
