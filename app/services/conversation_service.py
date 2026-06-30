import shutil
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from app.core.config import settings
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
            if msg.content:
                pending_parts.append({"type": "text", "content": str(msg.content)})
            for tc in msg.tool_calls or []:
                raw_output = tool_outputs.get(tc["id"], "")
                try:
                    import json as _json
                    data = _json.loads(raw_output)
                    if "__viz__" in data:
                        pending_parts.append({"type": "viz", "format": data["__viz__"], "code": data["code"], "title": data.get("title", "")})
                        continue
                except Exception:
                    pass
                pending_parts.append(
                    {
                        "type": "tool",
                        "tool": {
                            "name": tc["name"],
                            "input": tc["args"],
                            "output": raw_output,
                            "status": "done",
                        },
                    }
                )

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

    sandbox_dir = Path(settings.SANDBOX_DATA_DIR) / conversation_id
    if sandbox_dir.exists():
        shutil.rmtree(sandbox_dir, ignore_errors=True)

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
