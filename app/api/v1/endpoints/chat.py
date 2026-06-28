from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, AVAILABLE_MODELS
from app.services.chat_service import chat_service
from app.db import repository
from app.db.connection import get_pool

router = APIRouter()


@router.get("/models")
async def list_models():
    return AVAILABLE_MODELS


@router.post("/stream")
async def chat_stream(request: ChatRequest, req: Request):
    graph = req.app.state.graph
    pool = get_pool()

    await repository.touch_conversation(pool, request.thread_id)

    return StreamingResponse(
        chat_service.stream(request.thread_id, request.content, graph, request.thinking_effort, request.model),
        media_type="text/event-stream",
    )
