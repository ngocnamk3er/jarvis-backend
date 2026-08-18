from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.deps import CurrentUser, get_current_user
from app.schemas.chat import ChatRequest, ClarifyResumeRequest, StopRequest, AVAILABLE_MODELS
from app.services.chat_service import chat_service
from app.db import repository

router = APIRouter()


async def _check_owns_thread(thread_id: str, user: CurrentUser) -> None:
    conv = await repository.get_conversation(thread_id)
    if conv is None or conv.user_id != user.sub:
        raise HTTPException(status_code=404, detail="Conversation not found")


@router.get("/models")
async def list_models():
    return AVAILABLE_MODELS


@router.post("/stream")
async def chat_stream(request: ChatRequest, req: Request, user: CurrentUser = Depends(get_current_user)):
    await _check_owns_thread(request.thread_id, user)
    graph = req.app.state.graph
    await repository.touch_conversation(request.thread_id, request.model, request.subagent_model)
    return StreamingResponse(
        chat_service.stream(
            request.thread_id,
            request.content,
            graph,
            user.sub,
            request.thinking_effort,
            request.model,
            request.subagent_model,
        ),
        media_type="text/event-stream",
    )


@router.post("/stop")
async def chat_stop(request: StopRequest, req: Request, user: CurrentUser = Depends(get_current_user)):
    await _check_owns_thread(request.thread_id, user)
    stopped = chat_service.stop(request.thread_id)
    return {"stopped": stopped}


@router.post("/resume_clarify")
async def chat_resume_clarify(request: ClarifyResumeRequest, req: Request, user: CurrentUser = Depends(get_current_user)):
    await _check_owns_thread(request.thread_id, user)
    graph = req.app.state.graph
    await repository.touch_conversation(request.thread_id, request.model, request.subagent_model)
    return StreamingResponse(
        chat_service.resume_clarify(request.thread_id, request.answer, graph, user.sub, request.model, request.subagent_model),
        media_type="text/event-stream",
    )
