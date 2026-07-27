from fastapi import APIRouter, Request

from app.schemas.conversation import ConversationOut, ConversationCreate
from app.services import conversation_service

router = APIRouter()


@router.get("", response_model=list[ConversationOut])
async def list_conversations():
    return await conversation_service.list_conversations()


@router.post("", response_model=ConversationOut)
async def create_conversation(body: ConversationCreate):
    return await conversation_service.create_conversation(body.title)


@router.get("/{conversation_id}/messages")
async def get_messages(conversation_id: str, req: Request):
    return await conversation_service.get_messages(req.app.state.graph, conversation_id)


@router.patch("/{conversation_id}/title")
async def update_title(conversation_id: str, body: ConversationCreate):
    await conversation_service.update_title(conversation_id, body.title)
    return {"ok": True}


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str, req: Request):
    await conversation_service.delete_conversation(req.app.state.graph, conversation_id)
    return {"ok": True}
