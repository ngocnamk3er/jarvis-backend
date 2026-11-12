"""MCP servers the user added themselves, as tools for their conversations.

Unlike the operator's MCP_SERVERS (mcp_tools.py), which are loaded once at
startup and bound to the graph, these differ per user and per conversation,
so they are looked up on every model call: UserMCPMiddleware adds the tools of
the servers that are on for this conversation to what the model is offered,
and runs them when called — the tool node has never heard of them.

All of them are deferred (tool_search.py), and named mcp_<server>_<tool> so
they cannot shadow a built-in tool (mcp_guard.USER_TOOL_PREFIX). What a server
returns is untrusted text, as with any tool that reads from outside; it can
try to instruct the model. Only call servers you trust with the conversation.

A connection is only worth making once, not on every model call, so the tools
of each server are cached (a failure for a shorter time, so a dead server
costs one timeout a minute, not one per call).
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx
from cryptography.fernet import Fernet, InvalidToken
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.agents.mcp_guard import McpPolicyError, USER_TOOL_PREFIX, check_url
from app.core.config import settings
from app.db import mcp_repository

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_SECONDS = 15
TOOL_CALL_TIMEOUT_SECONDS = 60
MAX_TOOLS_PER_SERVER = 40
MAX_TOOL_NAME = 64        # the limit on a function name in the model APIs
CACHE_OK_SECONDS = 300
CACHE_ERROR_SECONDS = 60


# ── secrets ────────────────────────────────────────────────────────────────

def secrets_configured() -> bool:
    return bool(settings.MCP_SECRET_KEY)


def encrypt_headers(headers: dict[str, str]) -> str | None:
    if not headers:
        return None
    if not secrets_configured():
        raise McpPolicyError("Headers cannot be stored: the server has no MCP_SECRET_KEY configured.")
    return Fernet(settings.MCP_SECRET_KEY.encode()).encrypt(json.dumps(headers).encode()).decode()


def decrypt_headers(token: str | None) -> dict[str, str]:
    if not token:
        return {}
    try:
        return json.loads(Fernet(settings.MCP_SECRET_KEY.encode()).decrypt(token.encode()))
    except (InvalidToken, ValueError):
        # key rotated or unset: the server is reachable without them at best
        logger.warning("could not decrypt MCP headers")
        return {}


def header_names(token: str | None) -> list[str]:
    return sorted(decrypt_headers(token)) if token else []


# ── connecting ─────────────────────────────────────────────────────────────

def _no_redirect_client(headers=None, timeout=None, auth=None) -> httpx.AsyncClient:
    # The SDK's own factory follows redirects; a public server must not be able
    # to bounce the backend to an internal address the URL check never saw.
    kw: dict[str, Any] = {"follow_redirects": False}
    if headers is not None:
        kw["headers"] = headers
    if timeout is not None:
        kw["timeout"] = timeout
    if auth is not None:
        kw["auth"] = auth
    return httpx.AsyncClient(**kw)


@dataclass
class ServerTools:
    tools: list[BaseTool] = field(default_factory=list)
    error: str | None = None
    at: float = field(default_factory=time.monotonic)

    def fresh(self) -> bool:
        return time.monotonic() - self.at < (CACHE_ERROR_SECONDS if self.error else CACHE_OK_SECONDS)


_cache: dict[tuple[str, str], ServerTools] = {}
_locks: dict[tuple[str, str], asyncio.Lock] = {}


def _short(e: Exception) -> str:
    return f"{type(e).__name__}: {str(e)[:160]}"


async def _connect(server: dict) -> ServerTools:
    """server: name, url, transport and either `headers` (plain) or `headers_enc`."""
    try:
        await check_url(server["url"])
        headers = server.get("headers")
        if headers is None:
            headers = decrypt_headers(server.get("headers_enc"))
        conn: dict[str, Any] = {
            "transport": server["transport"], "url": server["url"], "httpx_client_factory": _no_redirect_client,
        }
        if headers:
            conn["headers"] = headers
        # The adapter prefixes with the key it is given: mcp_<server>_<tool>.
        client = MultiServerMCPClient({f"{USER_TOOL_PREFIX}{server['name']}": conn}, tool_name_prefix=True)
        got = await asyncio.wait_for(client.get_tools(), CONNECT_TIMEOUT_SECONDS)
    except McpPolicyError as e:
        return ServerTools(error=str(e))
    except Exception as e:  # unreachable, bad auth, not an MCP server, timeout …
        logger.info("user MCP server %r unavailable: %s", server.get("name"), _short(e))
        return ServerTools(error=_short(e) if not isinstance(e, asyncio.TimeoutError) else "Timed out connecting.")
    tools = [t for t in got if len(t.name) <= MAX_TOOL_NAME][:MAX_TOOLS_PER_SERVER]
    return ServerTools(tools=tools)


async def server_tools(server: dict, use_cache: bool = True) -> ServerTools:
    """The tools of one server. `server["id"]` and `updated_at` key the cache,
    so editing a server drops its entry; a server with no id (being tried out
    before it is saved) is never cached."""
    sid = server.get("id")
    if not sid or not use_cache:
        return await _connect(server)
    key = (sid, str(server.get("updated_at")))
    lock = _locks.setdefault(key, asyncio.Lock())
    async with lock:
        hit = _cache.get(key)
        if hit and hit.fresh():
            return hit
        _cache[key] = result = await _connect(server)
        return result


def forget(server_id: str) -> None:
    for key in [k for k in _cache if k[0] == server_id]:
        _cache.pop(key, None)
        _locks.pop(key, None)


# ── per-conversation lookup ────────────────────────────────────────────────

def _ids() -> tuple[str | None, str | None]:
    try:
        from langgraph.config import get_config

        cfg = get_config().get("configurable", {})
        return cfg.get("user_id"), cfg.get("thread_id")
    except RuntimeError:
        return None, None


async def tools_for_conversation(user_id: str | None, conversation_id: str | None) -> dict[str, BaseTool]:
    if not (settings.MCP_USER_ENABLED and user_id and conversation_id):
        return {}
    try:
        servers = await mcp_repository.list_enabled(user_id, conversation_id)
        results = await asyncio.gather(*(server_tools(s) for s in servers))
    except Exception:
        logger.warning("could not load user MCP tools", exc_info=True)
        return {}
    out: dict[str, BaseTool] = {}
    for r in results:
        for t in r.tools:
            out.setdefault(t.name, t)
    return out


async def deferred_catalog() -> dict[str, str]:
    """name -> one line, for tool_search to match against, for the user of the
    current run only."""
    user_id, conv = _ids()
    return {n: (t.description or "").strip().split("\n")[0][:160]
            for n, t in (await tools_for_conversation(user_id, conv)).items()}


class UserMCPMiddleware(AgentMiddleware):
    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> Any:
        dynamic = await tools_for_conversation(*_ids())
        if dynamic:
            request = request.override(tools=[*request.tools, *dynamic.values()])
        return await handler(request)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[Any]]
    ) -> Any:
        name = request.tool_call["name"]
        if request.tool is None and name.startswith(USER_TOOL_PREFIX):
            tool = (await tools_for_conversation(*_ids())).get(name)
            if tool is not None:
                request = request.override(tool=tool)
                try:
                    return await asyncio.wait_for(handler(request), TOOL_CALL_TIMEOUT_SECONDS)
                except asyncio.TimeoutError:
                    return ToolMessage(
                        content=f"'{name}' did not answer within {TOOL_CALL_TIMEOUT_SECONDS}s.",
                        tool_call_id=request.tool_call["id"], name=name, status="error",
                    )
        return await handler(request)
