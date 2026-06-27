from fastapi import APIRouter, Request
from langchain_core.messages import HumanMessage, AIMessage

from app.db import repository
from app.db.connection import get_pool
from app.schemas.conversation import ConversationOut, ConversationCreate

router = APIRouter()


def _serialize_messages(messages: list) -> list[dict]:
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage) and msg.content:
            result.append({"role": "assistant", "content": str(msg.content)})
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
