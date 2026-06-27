from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest
from app.services.chat_service import chat_service

router = APIRouter()


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        chat_service.stream(request.messages),
        media_type="text/event-stream",
    )
