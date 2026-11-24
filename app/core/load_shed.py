"""Admission control: when too many requests are already being served, refuse the next one at once.

Without it a saturated process serves everybody slowly: measured on the test cluster, `GET /conversations` tops out at
~120 req/s from about 10 concurrent requests on, and every request past that only adds latency (p50 14 ms at 1
connection, 0.87 s at 100, 4.6 s at 400, with timeouts) and nobody is ever told. Here a request that finds the class it
belongs to full gets `503` + `Retry-After` straight away, which is what lets the frontend back off instead of piling on.

Decided on the number of requests in flight, per class of route, before anything else runs (no auth, no DB): a refusal
has to cost much less than serving, or refusing is itself the overload. Classes are separate so that long chat streams
cannot use up the room the quick reads need, nor the reverse.

Pure ASGI on purpose (not BaseHTTPMiddleware, which buffers and breaks streaming), and it must sit *inside* the CORS
middleware: a 503 that skips CORS has no Access-Control-Allow-Origin, and the browser then shows the frontend a bare
network error instead of the status and the Retry-After header it needs.
"""

import json
import logging
import time
from collections.abc import Callable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

# What a refused caller is told. `code` is the stable part: the frontend switches on it, the sentence is for people.
_BODY_TEXT = "The server is busy right now. Try again in a moment."


class LoadShedMiddleware:
    """`classify(scope)` -> a class name, or None to never refuse that request. `limits()` -> {class: max in flight};
    a class missing from it, or at 0, is unlimited. `limits` is read on every request, so a change to the settings
    behind it takes effect without a restart of the middleware stack."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        classify: Callable[[Scope], str | None],
        limits: Callable[[], dict[str, int]],
        retry_after: Callable[[], int] = lambda: 1,
    ):
        self.app = app
        self.classify = classify
        self.limits = limits
        self.retry_after = retry_after
        self.in_flight: dict[str, int] = {}
        self.shed: dict[str, int] = {}
        self._last_log = 0.0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        name = self.classify(scope)
        cap = self.limits().get(name, 0) if name else 0
        if not name or cap <= 0:
            await self.app(scope, receive, send)
            return
        if self.in_flight.get(name, 0) >= cap:
            await self._refuse(name, cap, send)
            return
        self.in_flight[name] = self.in_flight.get(name, 0) + 1
        try:
            await self.app(scope, receive, send)        # for a streamed response this returns when the stream ends
        finally:
            self.in_flight[name] -= 1

    async def _refuse(self, name: str, cap: int, send: Send) -> None:
        seconds = max(1, int(self.retry_after()))
        body = json.dumps({"code": "overloaded", "detail": _BODY_TEXT, "retry_after": seconds}).encode()
        start: Message = {
            "type": "http.response.start",
            "status": 503,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"retry-after", str(seconds).encode()),
                (b"cache-control", b"no-store"),
            ],
        }
        await send(start)
        await send({"type": "http.response.body", "body": body})
        self.shed[name] = self.shed.get(name, 0) + 1
        now = time.monotonic()
        if now - self._last_log >= 10:                 # one line per ten seconds, not one per refused request
            self._last_log = now
            logger.warning("load shedding: %s class is at its limit of %d in flight; refused so far %s", name, cap, self.shed)


def backend_classifier(api_prefix: str) -> Callable[[Scope], str | None]:
    """Which class a jarvis-backend request is in. `None` is never refused: the probes (a refused probe makes
    Kubernetes restart the pod, which cuts every run in flight) and /chat/stop (the one request that lets the load go
    away)."""
    chat_runs = {f"{api_prefix}/chat/{p}" for p in ("stream", "resume", "resume_clarify")}
    never = {"/", "/docs", "/redoc", "/openapi.json", f"{api_prefix}/chat/stop"}

    def classify(scope: Scope) -> str | None:
        path = scope["path"].rstrip("/") or "/"
        if path in never or path.startswith(f"{api_prefix}/health"):
            return None
        if scope["method"] == "POST" and path in chat_runs:
            return "chat"
        return "default"

    return classify


def install_downstream_passthrough(app) -> None:
    """When a service this backend calls refuses with 503 "overloaded" (conversation-service does, past its pool),
    answer the browser with that same 503 and Retry-After, not the 500 that `raise_for_status()` would turn it into.
    Without it the refusal ends at the first hop and the frontend sees an ordinary server error, which it cannot tell
    from a bug and must not retry. Any other downstream error is raised again, so it is still a 500 as before."""
    import httpx
    from fastapi.responses import JSONResponse

    @app.exception_handler(httpx.HTTPStatusError)
    async def _passthrough(request, exc: httpx.HTTPStatusError):
        response = exc.response
        if response.status_code == 503:
            try:
                body = response.json()
            except ValueError:
                body = None
            if isinstance(body, dict) and body.get("code") == "overloaded":
                return JSONResponse(
                    body, status_code=503, headers={"Retry-After": response.headers.get("retry-after", "1"), "Cache-Control": "no-store"}
                )
        raise exc
