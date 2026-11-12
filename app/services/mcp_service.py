"""The MCP servers a user manages from the chat UI."""

import asyncio

from fastapi import HTTPException

from app.agents import user_mcp
from app.agents.mcp_guard import (
    McpPolicyError, check_url, validate_headers, validate_name, validate_transport,
)
from app.clients import conversation_client
from app.core.config import settings
from app.db import mcp_repository

LIST_CONNECT_TIMEOUT_SECONDS = 8


async def _owned_conversation(conversation_id: str, user_id: str) -> None:
    conv = await conversation_client.get_conversation(conversation_id)
    if conv is None or conv.user_id != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")


async def _describe(server: dict, enabled: bool | None) -> dict:
    try:
        got = await asyncio.wait_for(user_mcp.server_tools(server), LIST_CONNECT_TIMEOUT_SECONDS)
        tools, error = [t.name for t in got.tools], got.error
    except asyncio.TimeoutError:
        tools, error = [], "Timed out connecting."
    return {
        "id": server["id"],
        "name": server["name"],
        "url": server["url"],
        "transport": server["transport"],
        # Names only — a header's value is never sent back to the browser.
        "header_names": user_mcp.header_names(server.get("headers_enc")),
        "enabled": enabled,
        "tools": tools,
        "error": error,
    }


async def list_servers(user_id: str, conversation_id: str | None) -> dict:
    off: set[str] = set()
    if conversation_id:
        await _owned_conversation(conversation_id, user_id)
        off = await mcp_repository.disabled_ids(conversation_id)
    servers = await mcp_repository.list_servers(user_id)
    described = await asyncio.gather(
        *(_describe(s, (s["id"] not in off) if conversation_id else None) for s in servers)
    )
    return {
        "servers": list(described),
        "headers_supported": user_mcp.secrets_configured(),
        "max_servers": settings.MCP_USER_MAX_SERVERS,
        "enabled": settings.MCP_USER_ENABLED,
    }


async def add_server(user_id: str, name: str, url: str, transport: str, headers: dict[str, str]) -> dict:
    if not settings.MCP_USER_ENABLED:
        raise HTTPException(status_code=403, detail="Adding MCP servers is turned off.")
    try:
        validate_name(name)
        validate_transport(transport)
        validate_headers(headers)
        await check_url(url)
        headers_enc = user_mcp.encrypt_headers(headers)
    except McpPolicyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    if len(await mcp_repository.list_servers(user_id)) >= settings.MCP_USER_MAX_SERVERS:
        raise HTTPException(status_code=400, detail=f"At most {settings.MCP_USER_MAX_SERVERS} servers.")

    # Only keep a server that answers: the person adding it learns now, not on
    # the first message that needed it.
    tried = await user_mcp.server_tools(
        {"name": name, "url": url, "transport": transport, "headers": headers}, use_cache=False
    )
    if tried.error:
        raise HTTPException(status_code=400, detail=f"Could not connect: {tried.error}")
    if not tried.tools:
        raise HTTPException(status_code=400, detail="The server connected but offers no tools.")

    row = await mcp_repository.create_server(user_id, name, url, transport, headers_enc)
    if row is None:
        raise HTTPException(status_code=409, detail=f"You already have a server called {name!r}.")
    return await _describe(row, None)


async def remove_server(user_id: str, server_id: str) -> None:
    if not await mcp_repository.delete_server(user_id, server_id):
        raise HTTPException(status_code=404, detail="Server not found")
    user_mcp.forget(server_id)


async def set_enabled(user_id: str, conversation_id: str, server_id: str, enabled: bool) -> None:
    await _owned_conversation(conversation_id, user_id)
    if await mcp_repository.get_server(user_id, server_id) is None:
        raise HTTPException(status_code=404, detail="Server not found")
    await mcp_repository.set_enabled(conversation_id, server_id, enabled)
