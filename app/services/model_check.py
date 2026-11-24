"""Checking a model a user added: can the server reach it, is the key good, does it answer, can it call tools, can it see.

Plain HTTP against the OpenAI-compatible API (no LangChain), so the result says what the endpoint did, step by step,
and a failure names the step. Steps run in order and stop at the first one that decides the rest (no point asking for
a tool call from an endpoint that cannot be reached). The result is what the model list shows as its status:

  ok        reachable, answers, and calls tools (what Jarvis' agent needs)
  degraded  reachable and answers, but cannot call tools (a chat-only model: the agent's tools would not work)
  error     does not answer; the failing step says why
"""

import asyncio
import base64
import io
import json
import time

import httpx
from PIL import Image

from app.agents import user_models
from app.agents.mcp_guard import McpPolicyError, check_url

CONNECT_TIMEOUT = 8.0
CALL_TIMEOUT = 45.0
ECHO = "pong"


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _red_square_png() -> str:
    """A small solid-red PNG, base64: what a vision check asks the model to name the colour of."""
    img = Image.new("RGB", (64, 64), (220, 20, 20))
    out = io.BytesIO()
    img.save(out, "PNG")
    return base64.b64encode(out.getvalue()).decode()


def _explain(resp: httpx.Response) -> str:
    """A short reason from an error response. The server's own message when it gives one; never headers."""
    try:
        data = resp.json()
        msg = data.get("error", data)
        if isinstance(msg, dict):
            msg = msg.get("message") or json.dumps(msg)
        text = str(msg)
    except Exception:  # noqa: BLE001
        text = resp.text
    return f"HTTP {resp.status_code}: {text.strip()[:240]}"


async def _chat(client: httpx.AsyncClient, custom: user_models.CustomModel, messages: list, **extra) -> httpx.Response:
    body = {"model": custom.model, "messages": messages, "max_tokens": 64, "temperature": 0, **extra}
    return await client.post(custom.base_url.rstrip("/") + "/chat/completions", json=body, headers=_headers(custom.api_key))


def _step(name: str, ok: bool | None, detail: str, started: float) -> dict:
    return {"name": name, "ok": ok, "detail": detail, "ms": int((time.monotonic() - started) * 1000)}


async def _connection(client, custom, steps) -> bool:
    """Reach the endpoint, with the key, and (where the server lists them) find the model among its models."""
    t = time.monotonic()
    if custom.kind == "endpoint":
        try:
            await check_url(custom.base_url)
        except McpPolicyError as e:
            steps.append(_step("Address", False, str(e), t))
            return False
        steps.append(_step("Address", True, "The server may connect to it.", t))
    t = time.monotonic()
    if custom.kind == "openrouter":
        steps.append(_step("Models list", None, "Skipped: the model is looked up when it is called.", t))
        return True
    try:
        resp = await client.get(custom.base_url.rstrip("/") + "/models", headers=_headers(custom.api_key))
    except httpx.TimeoutException:
        steps.append(_step("Connection", False, "Timed out connecting.", t))
        return False
    except httpx.HTTPError as e:
        steps.append(_step("Connection", False, f"Could not connect ({type(e).__name__}).", t))
        return False
    if resp.status_code in (401, 403):
        steps.append(_step("API key", False, _explain(resp), t))
        return False
    if resp.status_code == 404:
        steps.append(_step("Models list", None, "This endpoint does not list models; going on to a test call.", t))
        return True
    if resp.status_code >= 400:
        steps.append(_step("Connection", False, _explain(resp), t))
        return False
    try:
        ids = {m.get("id") for m in resp.json().get("data", []) if isinstance(m, dict)}
    except Exception:  # noqa: BLE001
        ids = set()
    if ids and custom.model not in ids:
        steps.append(_step("Model", False, f"The endpoint answers but has no model called {custom.model!r}.", t))
        return False
    # Many servers list their models without a key (OpenRouter does), so this step cannot vouch for the key;
    # the test call that follows is what does.
    steps.append(_step("Connection", True, "Reached the endpoint.", t))
    return True


async def _answers(client, custom, steps) -> bool:
    t = time.monotonic()
    try:
        resp = await _chat(client, custom, [{"role": "user", "content": f"Reply with the single word {ECHO}."}])
    except httpx.TimeoutException:
        steps.append(_step("Test call", False, f"No answer within {int(CALL_TIMEOUT)} seconds.", t))
        return False
    except httpx.HTTPError as e:
        steps.append(_step("Test call", False, f"Request failed ({type(e).__name__}).", t))
        return False
    if resp.status_code >= 400:
        steps.append(_step("Test call", False, _explain(resp), t))
        return False
    try:
        msg = resp.json()["choices"][0]["message"]
    except Exception:  # noqa: BLE001
        steps.append(_step("Test call", False, "The answer was not in the OpenAI chat format.", t))
        return False
    text = (msg.get("content") or "").strip()
    reasoning = bool(msg.get("reasoning") or msg.get("reasoning_content"))
    detail = f"Answered in {int((time.monotonic() - t) * 1000)} ms" + (" (and returns its reasoning)" if reasoning else "") + "."
    if not text and not reasoning:
        steps.append(_step("Test call", False, "The call went through but the answer was empty.", t))
        return False
    steps.append(_step("Test call", True, detail, t))
    return True


_ADD_TOOL = {"type": "function", "function": {
    "name": "add", "description": "Add two integers.",
    "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}


async def _tools(client, custom, steps) -> bool | None:
    t = time.monotonic()
    try:
        resp = await _chat(client, custom, [{"role": "user", "content": "Use the add tool to add 17 and 25."}],
                           tools=[_ADD_TOOL], tool_choice="auto", max_tokens=200)
    except httpx.HTTPError as e:
        steps.append(_step("Tool calls", None, f"Could not test ({type(e).__name__}).", t))
        return None
    if resp.status_code >= 400:
        steps.append(_step("Tool calls", False, "The endpoint refused a request with tools: " + _explain(resp), t))
        return False
    try:
        calls = resp.json()["choices"][0]["message"].get("tool_calls") or []
    except Exception:  # noqa: BLE001
        calls = []
    if calls and calls[0].get("function", {}).get("name") == "add":
        steps.append(_step("Tool calls", True, "It called the test tool.", t))
        return True
    steps.append(_step("Tool calls", False, "It answered in words instead of calling the test tool; Jarvis' tools would not work with it.", t))
    return False


async def _vision(client, custom, steps) -> bool | None:
    t = time.monotonic()
    content = [{"type": "text", "text": "What colour is this square? Answer with one word."},
               {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _red_square_png()}}]
    try:
        resp = await _chat(client, custom, [{"role": "user", "content": content}], max_tokens=100)
    except httpx.HTTPError as e:
        steps.append(_step("Images", None, f"Could not test ({type(e).__name__}).", t))
        return None
    if resp.status_code >= 400:
        steps.append(_step("Images", False, "It does not accept images: " + _explain(resp), t))
        return False
    try:
        msg = resp.json()["choices"][0]["message"]
        text = ((msg.get("content") or "") + " " + (msg.get("reasoning") or "")).lower()
    except Exception:  # noqa: BLE001
        text = ""
    if any(w in text for w in ("red", "đỏ", "rouge", "rot")):
        steps.append(_step("Images", True, "It read the colour of a test picture.", t))
        return True
    steps.append(_step("Images", False, "It accepted a picture but did not describe it; it probably cannot see images.", t))
    return False


async def check(custom: user_models.CustomModel, transport: httpx.AsyncBaseTransport | None = None) -> tuple[str, dict, bool | None]:
    """(status, result, whether it sees images when that could be told). `transport` is for tests."""
    steps: list[dict] = []
    started = time.monotonic()
    timeout = httpx.Timeout(CALL_TIMEOUT, connect=CONNECT_TIMEOUT)
    tools = vision = None
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, transport=transport) as client:
        ok = await _connection(client, custom, steps) and await _answers(client, custom, steps)
        if ok:
            tools, vision = await asyncio.gather(_tools(client, custom, steps), _vision(client, custom, steps))
    status = "error" if not ok else ("ok" if tools else "degraded")
    result = {"steps": steps, "capabilities": {"tools": tools, "vision": vision} if ok else None,
              "total_ms": int((time.monotonic() - started) * 1000)}
    return status, result, (vision if ok else None)
