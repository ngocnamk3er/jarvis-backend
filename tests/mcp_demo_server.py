"""A tiny stdio MCP server for the tests: `python mcp_demo_server.py`."""

from mcp.server.fastmcp import FastMCP

import sys

# `python mcp_demo_server.py`            -> stdio
# `python mcp_demo_server.py http PORT`  -> streamable HTTP at http://127.0.0.1:PORT/mcp
_http = len(sys.argv) > 2 and sys.argv[1] == "http"
mcp = FastMCP("demo", host="127.0.0.1", port=int(sys.argv[2])) if _http else FastMCP("demo")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


@mcp.tool()
def shout(text: str) -> str:
    """Return the text in upper case."""
    return text.upper()


if __name__ == "__main__":
    mcp.run(transport="streamable-http" if _http else "stdio")
