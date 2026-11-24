"""Admission control: who is refused, who never is, and that a refusal reaches the browser readable."""

import asyncio

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.core.load_shed import LoadShedMiddleware, backend_classifier

PREFIX = "/api/v1"


def build(limits: dict[str, int]):
    """A small app with a handler that holds until released, behind the real middleware and the real classifier."""
    app = FastAPI()
    gate = asyncio.Event()

    @app.get("/api/v1/slow")
    async def slow():
        await gate.wait()
        return {"ok": True}

    @app.get("/api/v1/boom")
    async def boom():
        raise RuntimeError("handler failed")

    @app.get("/api/v1/health")
    async def health():
        await gate.wait()
        return {"status": "ok"}

    @app.post("/api/v1/chat/stream")
    async def stream():
        async def body():
            yield b"first"
            await gate.wait()
            yield b"last"
        return StreamingResponse(body())

    @app.post("/api/v1/chat/stop")
    async def stop():
        return {"stopped": True}

    app.add_middleware(LoadShedMiddleware, classify=backend_classifier(PREFIX), limits=lambda: limits)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://t")
    return client, gate


async def settle():
    await asyncio.sleep(0.05)


async def test_requests_under_the_limit_are_all_served():
    client, gate = build({"default": 3})
    tasks = [asyncio.create_task(client.get("/api/v1/slow")) for _ in range(3)]
    await settle()
    gate.set()
    assert [r.status_code for r in await asyncio.gather(*tasks)] == [200, 200, 200]


async def test_the_request_over_the_limit_is_refused_at_once_with_a_retry_after():
    client, gate = build({"default": 2})
    held = [asyncio.create_task(client.get("/api/v1/slow")) for _ in range(2)]
    await settle()
    refused = await asyncio.wait_for(client.get("/api/v1/slow"), timeout=1)      # answered while the others still wait
    assert refused.status_code == 503
    assert refused.headers["retry-after"] == "1"
    assert refused.headers["cache-control"] == "no-store"
    body = refused.json()
    assert body["code"] == "overloaded" and body["retry_after"] == 1 and body["detail"]
    gate.set()
    assert [r.status_code for r in await asyncio.gather(*held)] == [200, 200]      # the ones let in are not harmed


async def test_a_slot_is_free_again_once_a_request_finishes():
    client, gate = build({"default": 1})
    first = asyncio.create_task(client.get("/api/v1/slow"))
    await settle()
    assert (await client.get("/api/v1/slow")).status_code == 503
    gate.set()
    await first
    assert (await client.get("/api/v1/slow")).status_code == 200


async def test_a_handler_that_fails_still_frees_its_slot():
    client, _ = build({"default": 1})
    for _ in range(3):
        assert (await client.get("/api/v1/boom")).status_code == 500                  # never 503: the slot came back each time


async def test_probes_and_stop_are_never_refused_however_busy_it_is():
    client, gate = build({"default": 1, "chat": 1})
    held = [asyncio.create_task(client.get("/api/v1/slow")), asyncio.create_task(client.post("/api/v1/chat/stream"))]
    await settle()
    assert (await client.post("/api/v1/chat/stop")).status_code == 200
    probe = asyncio.create_task(client.get("/api/v1/health"))
    await settle()
    assert not probe.done() or probe.result().status_code != 503                      # waiting on the handler, not refused
    gate.set()
    assert (await probe).status_code == 200
    await asyncio.gather(*held)


async def test_chat_and_quick_requests_are_counted_apart():
    client, gate = build({"default": 1, "chat": 1})
    chat = asyncio.create_task(client.post("/api/v1/chat/stream"))
    await settle()
    assert (await client.post("/api/v1/chat/stream")).status_code == 503              # chat is full
    quick = asyncio.create_task(client.get("/api/v1/slow"))                           # but a quick read still fits
    await settle()
    assert not quick.done()
    gate.set()
    assert (await quick).status_code == 200
    await chat


async def test_a_streamed_response_holds_its_slot_until_the_stream_ends():
    client, gate = build({"chat": 1})
    first = asyncio.create_task(client.post("/api/v1/chat/stream"))
    await settle()                                                                    # the handler has returned, the body is still going
    assert (await client.post("/api/v1/chat/stream")).status_code == 503
    gate.set()
    await first
    assert (await client.post("/api/v1/chat/stream")).status_code == 200


async def test_zero_or_missing_limit_means_unlimited():
    client, gate = build({"default": 0})
    tasks = [asyncio.create_task(client.get("/api/v1/slow")) for _ in range(20)]
    await settle()
    gate.set()
    assert {r.status_code for r in await asyncio.gather(*tasks)} == {200}


def test_the_classifier_puts_each_real_route_in_the_right_class():
    c = backend_classifier(PREFIX)

    def cls(method, path):
        return c({"method": method, "path": path})

    assert cls("POST", "/api/v1/chat/stream") == "chat"
    assert cls("POST", "/api/v1/chat/resume") == "chat"
    assert cls("POST", "/api/v1/chat/resume_clarify") == "chat"
    assert cls("POST", "/api/v1/chat/stop") is None
    assert cls("GET", "/") is None and cls("GET", "/api/v1/health") is None and cls("GET", "/api/v1/health/") is None
    assert cls("GET", "/api/v1/conversations") == "default"
    assert cls("GET", "/api/v1/chat/models") == "default"                              # a read under /chat is not a run
    assert cls("POST", "/api/v1/chat/sandbox-upload") == "default"


def test_the_real_app_puts_cors_outside_the_shedder():
    """A 503 from the shedder must pass through CORS, or the browser cannot show the frontend its status."""
    from fastapi.middleware.cors import CORSMiddleware

    from app.main import app

    order = [m.cls for m in app.user_middleware]            # first = outermost
    assert order.index(CORSMiddleware) < order.index(LoadShedMiddleware)
    cors = next(m for m in app.user_middleware if m.cls is CORSMiddleware)
    assert "Retry-After" in cors.kwargs["expose_headers"]


# ---- a refusal from a service this backend calls --------------------------------------------------------------------


def downstream_app(status: int, body, headers=None):
    from app.core.load_shed import install_downstream_passthrough

    app = FastAPI()
    install_downstream_passthrough(app)

    @app.get("/call")
    async def call():
        def respond(request):
            return httpx.Response(status, json=body, headers=headers or {})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="http://svc") as c:
            (await c.get("/x")).raise_for_status()
        return {"ok": True}

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://t")


async def test_an_overloaded_503_from_a_downstream_service_reaches_the_browser_as_503_with_its_retry_after():
    client = downstream_app(503, {"code": "overloaded", "detail": "busy", "retry_after": 3}, {"retry-after": "3"})
    r = await client.get("/call")
    assert r.status_code == 503 and r.headers["retry-after"] == "3" and r.json()["code"] == "overloaded"


async def test_any_other_downstream_error_is_still_a_500():
    for status, body in ((503, {"detail": "down for maintenance"}), (503, "plain text"), (500, {"code": "overloaded"}), (404, {})):
        r = await downstream_app(status, body).get("/call")
        assert r.status_code == 500, (status, body)
