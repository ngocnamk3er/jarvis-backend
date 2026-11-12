"""MCP: tools served by external Model Context Protocol servers.

Servers come from the MCP_SERVERS setting — a JSON object (or the path to a
file holding one) mapping a name to a connection:

    {
      "docs":  {"url": "https://mcp.example.com/mcp", "headers": {"Authorization": "Bearer ${DOCS_TOKEN}"}},
      "files": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data"]}
    }

`url` means streamable HTTP (or SSE if `"transport": "sse"` is given) and
`command` means a stdio subprocess. `${VAR}` in any string is replaced from the
environment, so a token lives in a Secret rather than in the config.

Tools are loaded once at startup and named `<server>_<tool>`. A server that is
down or misconfigured is logged and skipped — it must not keep Jarvis from
starting. By default every MCP tool is deferred (see tool_search.py), since a
server can bring dozens of tools.

These are tools the operator chose to configure, so they run without the
approval prompt `bash` has. Do not point MCP_SERVERS at a server you would not
let the model call freely.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_SECONDS = 20


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def parse_mcp_servers(raw: str) -> dict[str, dict]:
    """The MCP_SERVERS setting as connections for MultiServerMCPClient; {} when
    it is empty or unusable (logged)."""
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw if raw.startswith("{") else Path(raw).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object of name -> connection")
    except Exception as e:
        logger.warning("MCP_SERVERS ignored: %s", e)
        return {}

    servers: dict[str, dict] = {}
    for name, conn in data.items():
        if not isinstance(conn, dict):
            logger.warning("MCP server %r ignored: not an object", name)
            continue
        conn = _expand(conn)
        if "transport" not in conn:
            conn["transport"] = "streamable_http" if "url" in conn else "stdio"
        if conn["transport"] == "stdio" and "command" not in conn:
            logger.warning("MCP server %r ignored: stdio needs a command", name)
            continue
        if conn["transport"] != "stdio" and "url" not in conn:
            logger.warning("MCP server %r ignored: %s needs a url", name, conn["transport"])
            continue
        servers[name] = conn
    return servers


async def load_mcp_tools(servers: dict[str, dict]) -> list[BaseTool]:
    """Connect to each server on its own and collect its tools."""
    tools: list[BaseTool] = []
    for name, conn in servers.items():
        try:
            client = MultiServerMCPClient({name: conn}, tool_name_prefix=True)
            got = await asyncio.wait_for(client.get_tools(), CONNECT_TIMEOUT_SECONDS)
            tools.extend(got)
            logger.info("MCP server %r: %d tools", name, len(got))
        except Exception as e:  # unreachable, wrong URL, bad auth, timeout …
            logger.warning("MCP server %r skipped: %s: %s", name, type(e).__name__, e)
    return tools
