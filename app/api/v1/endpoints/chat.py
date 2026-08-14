from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.deps import CurrentUser, get_current_user
from app.schemas.chat import ChatRequest, ResumeRequest, ClarifyResumeRequest, CompactRequest, AVAILABLE_MODELS
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


@router.post("/resume")
async def chat_resume(request: ResumeRequest, req: Request, user: CurrentUser = Depends(get_current_user)):
    await _check_owns_thread(request.thread_id, user)
    graph = req.app.state.graph
    await repository.touch_conversation(request.thread_id, request.model, request.subagent_model)
    return StreamingResponse(
        chat_service.resume(request.thread_id, request.decision, graph, user.sub, request.model, request.subagent_model),
        media_type="text/event-stream",
    )


@router.post("/resume_clarify")
async def chat_resume_clarify(request: ClarifyResumeRequest, req: Request, user: CurrentUser = Depends(get_current_user)):
    await _check_owns_thread(request.thread_id, user)
    graph = req.app.state.graph
    await repository.touch_conversation(request.thread_id, request.model, request.subagent_model)
    return StreamingResponse(
        chat_service.resume_clarify(request.thread_id, request.answer, graph, user.sub, request.model, request.subagent_model),
        media_type="text/event-stream",
    )


@router.post("/compact")
async def chat_compact(request: CompactRequest, req: Request, user: CurrentUser = Depends(get_current_user)):
    await _check_owns_thread(request.thread_id, user)
    # The only place compact_conversation is bound to the model — see
    # build_graph()'s docstring for why normal chat/resume deliberately use
    # the plain app.state.graph instead.
    graph = req.app.state.compact_graph
    return StreamingResponse(
        chat_service.compact(request.thread_id, graph, user.sub, request.model, request.subagent_model),
        media_type="text/event-stream",
    )
