from fastapi import APIRouter, Depends, Request

from app.api.deps import CurrentUser, get_current_user
from app.schemas.conversation import ConversationOut, ConversationCreate
from app.services import conversation_service

router = APIRouter()


@router.get("", response_model=list[ConversationOut])
async def list_conversations(user: CurrentUser = Depends(get_current_user)):
    return await conversation_service.list_conversations(user.sub)


@router.post("", response_model=ConversationOut)
async def create_conversation(body: ConversationCreate, user: CurrentUser = Depends(get_current_user)):
    return await conversation_service.create_conversation(body.title, user.sub)


@router.get("/{conversation_id}/messages")
async def get_messages(conversation_id: str, req: Request, user: CurrentUser = Depends(get_current_user)):
    return await conversation_service.get_messages(req.app.state.graph, conversation_id, user.sub)


@router.patch("/{conversation_id}/title")
async def update_title(conversation_id: str, body: ConversationCreate, user: CurrentUser = Depends(get_current_user)):
    await conversation_service.update_title(conversation_id, body.title, user.sub)
    return {"ok": True}


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str, req: Request, user: CurrentUser = Depends(get_current_user)):
    await conversation_service.delete_conversation(req.app.state.graph, conversation_id, user.sub)
    return {"ok": True}
