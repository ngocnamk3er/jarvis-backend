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

import base64
import contextlib
import logging
from typing import Literal

import httpx

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


def new_trace_id(seed: str | None = None) -> str:
    """A fresh Langfuse trace id: 32 lowercase hex chars, the shape
    trace_as()'s trace_context requires — not a dashed UUID.

    Safe to call even when tracing is off; the id just never gets used,
    since trace_as() is a no-op in that case. Callers don't need to check
    whether tracing is configured before generating one — see
    chat_service.py, which always generates an id and sends it to the
    frontend up front, before the run has even produced its first token.
    """
    from langfuse import Langfuse

    return Langfuse.create_trace_id(seed=seed)


@contextlib.contextmanager
def trace_as(trace_id: str):
    """Pins the Langfuse trace for whatever runs inside this block to
    `trace_id`, instead of letting Langfuse pick one — so the id handed
    to the frontend before the run starts (see new_trace_id above) is
    the same one a later create_score() call can attach feedback to.

    A plain context manager, not an async one — verified live against
    the installed SDK: the underlying start_as_current_observation()
    returns an OpenTelemetry-based manager that only implements `with`,
    even though the code running inside it here is async
    (graph.astream_events). `with` wrapping an `async for` is fine;
    there's nothing in the manager itself that needs awaiting.

    No-ops when tracing is off, same degrade-to-nothing posture as
    callbacks()/trace_metadata() — the caller still gets a context to
    enter, it just does nothing inside it.
    """
    if _handler is None:
        yield
        return
    from langfuse import get_client

    with get_client().start_as_current_observation(
        trace_context={"trace_id": trace_id}, name="chat-turn", as_type="span",
    ):
        yield


def create_score(trace_id: str, rating: Literal["up", "down"], comment: str | None) -> None:
    """Records a thumbs up/down (plus an optional free-text comment) on
    `trace_id` as a Langfuse Score — reusing Langfuse's own feedback
    primitive instead of a bespoke Postgres table, since the trace being
    scored already lives there.

    score_id is deterministic (derived from trace_id), not left for
    Langfuse to auto-generate — verified live: without a fixed id, a
    changed vote creates a second score row instead of replacing the
    first. One fixed id per trace makes a re-vote a true upsert.

    Raises when tracing isn't configured — unlike tracing itself, which
    degrades to a silent no-op throughout this module, feedback with
    nowhere to actually go is a real failure for the caller (the chat
    endpoint turns this into a 400), not something to swallow.
    """
    if _handler is None:
        raise RuntimeError("Langfuse is not configured — feedback cannot be recorded")
    from langfuse import get_client

    get_client().create_score(
        name="user-feedback",
        value=1 if rating == "up" else 0,
        trace_id=trace_id,
        data_type="BOOLEAN",
        score_id=f"user-feedback-{trace_id}",
        comment=comment,
    )


async def get_score(trace_id: str) -> dict | None:
    """The user-feedback rating already recorded for `trace_id`, or None
    if nobody has voted on it (yet, or at all) — lets the frontend
    restore "you already rated this" after a reload.

    No SDK method reads scores back (confirmed against the installed
    client: create_score has no read counterpart), so this is a direct
    REST call against the same credentials init_tracing() already holds.

    Ingestion lag is real here — verified live, newly-created scores
    took several seconds to become visible to this same endpoint in this
    deployment's events_only mode. Callers that just wrote a score
    should update their own UI state optimistically rather than re-fetch
    to confirm; this function is for state that's had time to settle.
    """
    if _handler is None:
        return None
    auth = base64.b64encode(
        f"{settings.LANGFUSE_PUBLIC_KEY}:{settings.LANGFUSE_SECRET_KEY}".encode()
    ).decode()
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{settings.LANGFUSE_HOST}/api/public/v3/scores",
            headers={"Authorization": f"Basic {auth}"},
            params={"traceId": trace_id, "name": "user-feedback"},
            timeout=10.0,
        )
    resp.raise_for_status()
    data = resp.json()["data"]
    if not data:
        return None
    return {"rating": "up" if data[0]["value"] else "down"}
