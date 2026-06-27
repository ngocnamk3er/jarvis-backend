from fastapi import APIRouter, Request
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from app.db import repository
from app.db.connection import get_pool
from app.schemas.conversation import ConversationOut, ConversationCreate

router = APIRouter()


def _serialize_messages(messages: list) -> list[dict]:
    # Build tool_call_id → output map from ToolMessages
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
            result.append({
                "role": "user",
                "parts": [{"type": "text", "content": str(msg.content)}],
            })

        elif isinstance(msg, AIMessage):
            for tc in (msg.tool_calls or []):
                pending_parts.append({
                    "type": "tool",
                    "tool": {
                        "name": tc["name"],
                        "input": tc["args"],
                        "output": tool_outputs.get(tc["id"], ""),
                        "status": "done",
                    },
                })
            if msg.content:
                pending_parts.append({"type": "text", "content": str(msg.content)})

    flush()
    return result


@router.get("", response_model=list[ConversationOut])
async def list_conversations():
    pool = get_pool()
    return await repository.list_conversations(pool)


@router.post("", response_model=ConversationOut)
async def create_conversation(body: ConversationCreate):
    pool = get_pool()
    return await repository.create_conversation(pool, body.title)


@router.get("/{conversation_id}/messages")
async def get_messages(conversation_id: str, req: Request):
    graph = req.app.state.graph
    config = {"configurable": {"thread_id": conversation_id}}
    state = await graph.aget_state(config)
    messages = state.values.get("messages", []) if state.values else []
    return _serialize_messages(messages)


@router.patch("/{conversation_id}/title")
async def update_title(conversation_id: str, body: ConversationCreate):
    pool = get_pool()
    await repository.update_conversation_title(pool, conversation_id, body.title)
    return {"ok": True}


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str):
    pool = get_pool()
    await repository.delete_conversation(pool, conversation_id)
    return {"ok": True}
