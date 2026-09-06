import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from app.api.deps import CurrentUser, get_current_user
from app.agents.tools import sandbox_manager
from app.schemas.chat import ChatRequest, ResumeRequest, ClarifyResumeRequest, StopRequest, AVAILABLE_MODELS
from app.services.chat_service import chat_service
from app.clients import conversation_client

router = APIRouter()


async def _check_owns_thread(thread_id: str, user: CurrentUser) -> None:
    conv = await conversation_client.get_conversation(thread_id)
    if conv is None or conv.user_id != user.sub:
        raise HTTPException(status_code=404, detail="Conversation not found")


@router.get("/models")
async def list_models():
    return AVAILABLE_MODELS


@router.post("/stream")
async def chat_stream(request: ChatRequest, req: Request, user: CurrentUser = Depends(get_current_user)):
    await _check_owns_thread(request.thread_id, user)
    graph = req.app.state.graph
    await conversation_client.touch_conversation(request.thread_id, request.model, request.subagent_model)
    return StreamingResponse(
        chat_service.stream(
            request.thread_id,
            request.content,
            graph,
            user.sub,
            request.thinking_effort,
            request.model,
            request.subagent_model,
            request.web_search,
        ),
        media_type="text/event-stream",
    )


@router.post("/resume")
async def chat_resume(request: ResumeRequest, req: Request, user: CurrentUser = Depends(get_current_user)):
    await _check_owns_thread(request.thread_id, user)
    graph = req.app.state.graph
    await conversation_client.touch_conversation(request.thread_id, request.model, request.subagent_model)
    return StreamingResponse(
        chat_service.resume(
            request.thread_id, request.decision, graph, user.sub,
            request.model, request.subagent_model, request.web_search,
        ),
        media_type="text/event-stream",
    )


@router.get("/sandbox-file")
async def chat_sandbox_file(thread_id: str, name: str, user: CurrentUser = Depends(get_current_user)):
    """Download a file the agent surfaced with `present_file` — proxied from
    the sandbox. Chat-scoped: it lives in the sandbox's ephemeral workspace,
    so a link stops working once that pod restarts."""
    await _check_owns_thread(thread_id, user)
    try:
        content, mime, filename = await sandbox_manager.read_file(thread_id, name)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail="File no longer available")
    except httpx.HTTPError:
        raise HTTPException(status_code=503, detail="Sandbox unavailable")
    return Response(
        content=content,
        media_type=mime,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@router.post("/stop")
async def chat_stop(request: StopRequest, req: Request, user: CurrentUser = Depends(get_current_user)):
    await _check_owns_thread(request.thread_id, user)
    stopped = await chat_service.stop(request.thread_id)
    return {"stopped": stopped}


@router.post("/resume_clarify")
async def chat_resume_clarify(request: ClarifyResumeRequest, req: Request, user: CurrentUser = Depends(get_current_user)):
    await _check_owns_thread(request.thread_id, user)
    graph = req.app.state.graph
    await conversation_client.touch_conversation(request.thread_id, request.model, request.subagent_model)
    return StreamingResponse(
        chat_service.resume_clarify(
            request.thread_id, request.answer, graph, user.sub,
            request.model, request.subagent_model, request.web_search,
        ),
        media_type="text/event-stream",
    )
