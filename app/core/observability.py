"""Langfuse tracing for agent runs.

A chat turn already emits SSE events the frontend renders, but nothing
persists *why* the turn went the way it did: which model actually answered
after the fallback chain fired, what each tool returned, where the tokens
went, which subagent was responsible for a bad answer. The SSE stream is
gone the moment the request closes, and the checkpoint keeps only the
resulting messages, not the path taken to them.

Langfuse stores that path as a trace tree — one span per graph node, tool
call and model call — and the LangChain integration produces it from a
single callback handler, so nothing in the graph or the tools has to know
tracing exists.

Off unless LANGFUSE_HOST and both keys are set, so local runs and tests
need no server.
"""

import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

_handler = None


def init_tracing() -> None:
    """Register the Langfuse client and build the shared handler. Called once
    from main.lifespan, eagerly rather than on the first chat so that a wrong
    host or a rejected key shows up in the startup log instead of silently
    dropping every trace for the life of the pod.
    """
    global _handler
    if not (settings.LANGFUSE_HOST and settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY):
        logger.info("langfuse not configured — agent tracing disabled")
        return
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        # Constructing the client registers it globally; CallbackHandler()
        # then picks it up, which is why it takes no keys of its own.
        Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
            environment=settings.LANGFUSE_ENVIRONMENT,
            release=settings.APP_VERSION,
        )
        # One shared handler for the whole process, not one per request. It
        # keeps per-run state in dicts keyed by LangChain's run_id, so
        # concurrent chats don't mix. What a fresh instance per request
        # *would* break is the bash approval flow: the handler stores pending
        # resume contexts on itself, and those are what stitch a resumed run
        # (Command(resume=...), a separate HTTP request) back onto the trace
        # it interrupted instead of starting an orphan.
        _handler = CallbackHandler()
        logger.info("langfuse tracing enabled → %s", settings.LANGFUSE_HOST)
    except Exception:
        logger.warning("could not initialise langfuse — tracing disabled", exc_info=True)


def shutdown_tracing() -> None:
    """Flush buffered spans on shutdown. The SDK batches by design, so without
    this the last turns before a rollout are dropped with the process."""
    if _handler is None:
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        logger.warning("langfuse flush failed", exc_info=True)


def callbacks() -> list:
    """Callbacks for a graph run config. Empty list when tracing is off, so
    callers never have to branch."""
    return [_handler] if _handler is not None else []


def trace_metadata(thread_id: str, user_id: str) -> dict:
    """Reserved keys Langfuse reads off the LangChain run metadata.

    session_id is the one that matters here: it collapses every turn of a
    conversation — including the separate requests a bash approval splits a
    turn into — into one timeline, which is the view that matches how jarvis
    is actually used. Returns {} when tracing is off rather than writing keys
    nothing will read into every checkpoint.
    """
    if _handler is None:
        return {}
    return {"langfuse_session_id": thread_id, "langfuse_user_id": user_id}
