import logging
from pathlib import PurePosixPath

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import StreamingResponse
from k8s_agent_sandbox.exceptions import SandboxRequestError

from app.agents.tools import sandbox_manager
from app.agents.tools.files import sandbox_name
from app.api.deps import CurrentUser, get_current_user
from app.clients import conversation_client
from app.core import observability
from app.core.config import settings
from app.core.conversation_lock import Lease, conversation_locks
from app.core.sse import with_heartbeat
from app.db import attachment_repository, feedback_repository
from app.schemas.chat import (
    DEFAULT_MODEL,
    AVAILABLE_MODELS,
    ChatRequest,
    ClarifyResumeRequest,
    FeedbackOut,
    FeedbackRequest,
    ResumeRequest,
    StopRequest,
)
from app.services.chat_service import chat_service

logger = logging.getLogger(__name__)
router = APIRouter()


async def _check_owns_thread(thread_id: str, user: CurrentUser):
    """The conversation, if it exists and belongs to `user`; 404 otherwise."""
    conv = await conversation_client.get_conversation(thread_id)
    if conv is None or conv.user_id != user.sub:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


async def _lock_conversation(thread_id: str) -> Lease:
    """The conversation's run lock, or 409 if another request is running on it. Without this two runs on one thread
    each write their own checkpoint and one turn is lost (see app/core/conversation_lock.py)."""
    lease = await conversation_locks.acquire(thread_id)
    if lease is None:
        raise HTTPException(status_code=409, detail="This conversation is already running. Wait for it to finish.")
    return lease


def _resume_models(request, conv) -> tuple[str, str | None]:
    """(model, subagent model) a resume should run with.

    A resume that names no model carries on with the one the conversation was
    already using. It used to fall back to the default model, so a conversation
    started on a vision model carried on, after a bash approval, on DeepSeek Flash,
    which cannot see images. A model the request does name is always honoured."""
    model = request.model
    if not model:
        model = conv.last_model or DEFAULT_MODEL
        logger.warning("resume of %s named no model; continuing with %s", request.thread_id, model)
    subagent = request.subagent_model if request.subagent_model is not None else conv.last_subagent_model
    return model, subagent


@router.get("/models")
async def list_models():
    return AVAILABLE_MODELS


@router.post("/stream")
async def chat_stream(
    request: ChatRequest, req: Request, user: CurrentUser = Depends(get_current_user)
):
    await _check_owns_thread(request.thread_id, user)
    graph = req.app.state.graph
    lease = await _lock_conversation(request.thread_id)
    try:
        await conversation_client.touch_conversation(
            request.thread_id, request.model, request.subagent_model
        )
    except BaseException:
        lease.release()
        raise
    return StreamingResponse(
        with_heartbeat(chat_service.guarded(lease, chat_service.stream(
            request.thread_id,
            request.content,
            graph,
            user.sub,
            request.thinking_effort,
            request.model,
            request.subagent_model,
            request.web_search,
            request.unattended,
        )), settings.SSE_HEARTBEAT_SECONDS),
        media_type="text/event-stream",
    )


@router.post("/resume")
async def chat_resume(
    request: ResumeRequest, req: Request, user: CurrentUser = Depends(get_current_user)
):
    conv = await _check_owns_thread(request.thread_id, user)
    model, subagent_model = _resume_models(request, conv)
    graph = req.app.state.graph
    lease = await _lock_conversation(request.thread_id)
    try:
        await conversation_client.touch_conversation(request.thread_id, model, subagent_model)
    except BaseException:
        lease.release()
        raise
    return StreamingResponse(
        with_heartbeat(chat_service.guarded(lease, chat_service.resume(
            request.thread_id,
            request.decision,
            graph,
            user.sub,
            model,
            subagent_model,
            request.web_search,
        )), settings.SSE_HEARTBEAT_SECONDS),
        media_type="text/event-stream",
    )


@router.get("/attachments/{attachment_id}")
async def chat_attachment(attachment_id: str, user: CurrentUser = Depends(get_current_user)):
    """An image the agent looked at with `view_image`, for the chat to show under
    that step. Only its owner can fetch it; unlike /sandbox-file it does not
    depend on the sandbox still existing."""
    found = await attachment_repository.get(attachment_id, user.sub)
    if found is None:
        raise HTTPException(status_code=404, detail="Attachment not found")
    data, mime = found
    return Response(content=data, media_type=mime, headers={"Cache-Control": "private, max-age=3600"})


# What one chat attachment may weigh: the same cap fetch_file puts on a file it
# copies into the sandbox, so anything attached can also be opened there.
MAX_ATTACHMENT_BYTES = 50 << 20


async def _free_name(thread_id: str, name: str) -> str:
    """`name`, or `name (2).ext`, `name (3).ext`... so an attachment never
    overwrites a file already in the sandbox."""
    listing = await sandbox_manager.exec_bash(thread_id, "ls -1A")
    taken = set((listing.get("stdout") or "").splitlines())
    if name not in taken:
        return name
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    n = 2
    while True:
        candidate = f"{stem} ({n}){'.' + ext if ext else ''}"
        if candidate not in taken:
            return candidate
        n += 1


@router.post("/sandbox-upload")
async def chat_sandbox_upload(
    thread_id: str = Form(...),
    file: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
):
    """A file attached to a chat message. It goes into this conversation's
    sandbox working directory and nowhere else — not the file workspace — so the
    agent can open it with its sandbox tools and it disappears with the
    conversation. Returns the name it was saved under."""
    await _check_owns_thread(thread_id, user)
    chunks, size = [], 0
    while chunk := await file.read(1 << 20):
        size += len(chunk)
        if size > MAX_ATTACHMENT_BYTES:
            raise HTTPException(status_code=413, detail=f"File is over the {MAX_ATTACHMENT_BYTES >> 20} MB limit")
        chunks.append(chunk)
    content = b"".join(chunks)
    try:
        name = await _free_name(thread_id, sandbox_name(file.filename or "file"))
        await sandbox_manager.write_file(thread_id, name, content)
    except SandboxRequestError as e:
        raise HTTPException(status_code=503, detail="Sandbox unavailable") from e
    import mimetypes

    mime = file.content_type if file.content_type not in (None, "", "application/octet-stream") \
        else (mimetypes.guess_type(name)[0] or "application/octet-stream")
    return {"name": name, "size": size, "mime": mime}


@router.get("/sandbox-file")
async def chat_sandbox_file(
    thread_id: str, name: str, user: CurrentUser = Depends(get_current_user)
):
    """Download a file the agent surfaced with `present_file` — proxied from
    the sandbox. Chat-scoped: it lives in the sandbox's ephemeral workspace,
    so a link stops working once that pod restarts."""
    await _check_owns_thread(thread_id, user)
    name = sandbox_manager.normalize_workspace_path(name)
    if name.startswith("/") or ".." in PurePosixPath(name).parts:
        raise HTTPException(status_code=400, detail="Invalid file name")
    try:
        content, mime, filename = await sandbox_manager.read_file(thread_id, name)
    except SandboxRequestError as e:
        if e.status_code is not None:
            raise HTTPException(status_code=e.status_code, detail="File no longer available") from e
        raise HTTPException(status_code=503, detail="Sandbox unavailable") from e
    return Response(
        content=content,
        media_type=mime,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@router.post("/stop")
async def chat_stop(
    request: StopRequest, req: Request, user: CurrentUser = Depends(get_current_user)
):
    await _check_owns_thread(request.thread_id, user)
    stopped = await chat_service.stop(request.thread_id)
    return {"stopped": stopped}


@router.post("/resume_clarify")
async def chat_resume_clarify(
    request: ClarifyResumeRequest, req: Request, user: CurrentUser = Depends(get_current_user)
):
    conv = await _check_owns_thread(request.thread_id, user)
    model, subagent_model = _resume_models(request, conv)
    graph = req.app.state.graph
    lease = await _lock_conversation(request.thread_id)
    try:
        await conversation_client.touch_conversation(request.thread_id, model, subagent_model)
    except BaseException:
        lease.release()
        raise
    return StreamingResponse(
        with_heartbeat(chat_service.guarded(lease, chat_service.resume_clarify(
            request.thread_id,
            request.answer,
            graph,
            user.sub,
            model,
            subagent_model,
            request.web_search,
        )), settings.SSE_HEARTBEAT_SECONDS),
        media_type="text/event-stream",
    )


@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest, user: CurrentUser = Depends(get_current_user)
):
    """Thumbs up/down (plus an optional comment) on one assistant message,
    identified by the trace recorded while generating it. thread_id is only
    here to check ownership, the same way every other endpoint on this
    router scopes a request to a conversation the caller actually owns.

    Postgres (feedback_repository) is the source of truth — read-your-own-
    write, no lag. The Langfuse mirror (observability.create_score) is
    best-effort and never raises; see its docstring for why."""
    await _check_owns_thread(request.thread_id, user)
    await feedback_repository.save_feedback(request.trace_id, request.rating, request.comment)
    observability.create_score(request.trace_id, request.rating, request.comment)
    return {"ok": True}


@router.get("/feedback", response_model=FeedbackOut)
async def get_feedback(thread_id: str, trace_id: str, user: CurrentUser = Depends(get_current_user)):
    await _check_owns_thread(thread_id, user)
    feedback = await feedback_repository.get_feedback(trace_id)
    return {"rating": feedback["rating"] if feedback else None, "comment": feedback["comment"] if feedback else None}
