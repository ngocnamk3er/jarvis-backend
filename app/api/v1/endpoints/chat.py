from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ResumeRequest, AVAILABLE_MODELS
from app.services.chat_service import chat_service
from app.db import repository

router = APIRouter()


@router.get("/models")
async def list_models():
    return AVAILABLE_MODELS


@router.post("/stream")
async def chat_stream(request: ChatRequest, req: Request):
    graph = req.app.state.graph
    await repository.touch_conversation(request.thread_id, request.model, request.subagent_model)
    return StreamingResponse(
        chat_service.stream(
            request.thread_id,
            request.content,
            graph,
            request.thinking_effort,
            request.model,
            request.subagent_model,
        ),
        media_type="text/event-stream",
    )


@router.post("/resume")
async def chat_resume(request: ResumeRequest, req: Request):
    graph = req.app.state.graph
    await repository.touch_conversation(request.thread_id, request.model, request.subagent_model)
    return StreamingResponse(
        chat_service.resume(request.thread_id, request.decision, graph, request.model, request.subagent_model),
        media_type="text/event-stream",
    )
