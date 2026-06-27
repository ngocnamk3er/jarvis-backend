import re
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from app.core.config import settings
from app.storage.minio_client import get_minio
from app.db import repository

# ── serialization ─────────────────────────────────────────────────────────────


def serialize_messages(messages: list) -> list[dict]:
    tool_outputs: dict[str, str] = {
        msg.tool_call_id: str(msg.content)
        for msg in messages
        if isinstance(msg, ToolMessage)
    }

    result: list[dict] = []
    pending_parts: list[dict] = []

    def flush():
        if pending_parts:
            result.append({"role": "assistant", "parts": list(pending_parts)})
            pending_parts.clear()

    for msg in messages:
        if isinstance(msg, HumanMessage):
            flush()
            result.append(
                {
                    "role": "user",
                    "parts": [{"type": "text", "content": str(msg.content)}],
                }
            )
        elif isinstance(msg, AIMessage):
            reasoning = msg.additional_kwargs.get("reasoning", "")
            if reasoning:
                pending_parts.append({"type": "thinking", "content": reasoning, "isStreaming": False})
            for tc in msg.tool_calls or []:
                pending_parts.append(
                    {
                        "type": "tool",
                        "tool": {
                            "name": tc["name"],
                            "input": tc["args"],
                            "output": tool_outputs.get(tc["id"], ""),
                            "status": "done",
                        },
                    }
                )
            if msg.content:
                pending_parts.append({"type": "text", "content": str(msg.content)})

    flush()
    return result


# ── CRUD ──────────────────────────────────────────────────────────────────────


async def list_conversations(pool):
    return await repository.list_conversations(pool)


async def create_conversation(pool, title: str):
    return await repository.create_conversation(pool, title)


async def delete_conversation(pool, graph, conversation_id: str) -> None:
    config = {"configurable": {"thread_id": conversation_id}}
    state = await graph.aget_state(config)
    messages = state.values.get("messages", []) if state.values else []

    minio = get_minio()
    for obj in _extract_minio_objects(messages):
        try:
            minio.remove_object(settings.MINIO_BUCKET, obj)
        except Exception:
            pass

    await repository.delete_conversation(pool, conversation_id)


async def update_title(pool, conversation_id: str, title: str):
    await repository.update_conversation_title(pool, conversation_id, title)


async def get_messages(graph, conversation_id: str) -> dict:
    config = {"configurable": {"thread_id": conversation_id}}
    state = await graph.aget_state(config)
    messages = state.values.get("messages", []) if state.values else []
    return {
        "messages": serialize_messages(messages),
        "is_pending": bool(state.next),
    }


# ── delete (with MinIO cleanup) ───────────────────────────────────────────────


def _extract_minio_objects(messages: list) -> list[str]:
    pattern = re.compile(
        r"https?://[^/]+/" + re.escape(settings.MINIO_BUCKET) + r"/([^\s\)\"]+)"
    )
    objects = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            for match in pattern.finditer(str(msg.content)):
                objects.append(match.group(1))
    return objects
