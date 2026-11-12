"""What a user may point an MCP connection at.

MCP servers added by users are fetched by the backend, from inside the
cluster. Left unchecked that is a way to make the backend call anything it can
reach — the Kubernetes API, the metadata service, Keycloak, the databases —
with the user choosing the URL. So a user connection is held to:

- http or https only. stdio is never allowed: it would run a command of the
  user's choosing in the backend container.
- a host that resolves only to public addresses. Loopback, private ranges,
  link-local and the like are refused; an operator can name specific internal
  hosts in MCP_USER_ALLOWED_HOSTS (a deployment's own server, say).
- no credentials in the URL, no redirects (see user_mcp.py): a public server
  must not be able to bounce the backend to an internal one.

The check runs when a server is added and again before every connection, since
what a name resolves to can change. A name that resolves differently between
the check and the connect (DNS rebinding) is not defended against.
"""

import asyncio
import ipaddress
import re
import socket
from urllib.parse import urlsplit

from app.core.config import settings

# Every tool a user's server brings is named mcp_<server>_<tool>, so it can
# never shadow a built-in tool ("web" + "search" must not become web_search).
USER_TOOL_PREFIX = "mcp_"

ALLOWED_TRANSPORTS = ("streamable_http", "sse")
MAX_HEADERS = 10
_NAME = re.compile(r"^[a-z][a-z0-9-]{0,23}$")
# RFC 9110 token
_HEADER_NAME = re.compile(r"^[A-Za-z0-9!#$%&'*+.^_`|~-]{1,64}$")
_FORBIDDEN_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "upgrade", "te", "trailer"}


class McpPolicyError(ValueError):
    """The user asked for something not allowed; the message is safe to show."""


def validate_name(name: str) -> str:
    if not _NAME.match(name or ""):
        raise McpPolicyError("Name must start with a letter and use only lowercase letters, digits and hyphens (max 24).")
    return name


def validate_transport(transport: str) -> str:
    if transport not in ALLOWED_TRANSPORTS:
        raise McpPolicyError(f"Transport must be one of: {', '.join(ALLOWED_TRANSPORTS)}.")
    return transport


def validate_headers(headers: dict[str, str]) -> dict[str, str]:
    if len(headers) > MAX_HEADERS:
        raise McpPolicyError(f"At most {MAX_HEADERS} headers.")
    for k, v in headers.items():
        if not _HEADER_NAME.match(k) or k.lower() in _FORBIDDEN_HEADERS:
            raise McpPolicyError(f"Header name {k!r} is not allowed.")
        if not isinstance(v, str) or len(v) > 2048 or any(c in v for c in "\r\n\0"):
            raise McpPolicyError(f"Header {k!r} has an invalid value.")
    return headers


def _allowed_hosts() -> set[str]:
    return {h.strip().lower() for h in settings.MCP_USER_ALLOWED_HOSTS.split(",") if h.strip()}


async def check_url(url: str) -> None:
    """Raise McpPolicyError unless the backend may connect to `url`."""
    if len(url) > 2048:
        raise McpPolicyError("URL is too long.")
    try:
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port
    except ValueError:
        raise McpPolicyError("That is not a valid URL.") from None
    if parts.scheme not in ("http", "https") or not host:
        raise McpPolicyError("URL must start with http:// or https:// and include a host.")
    if parts.username or parts.password:
        raise McpPolicyError("Put credentials in a header, not in the URL.")
    if host.lower() in _allowed_hosts():
        return
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(
            host, port or (443 if parts.scheme == "https" else 80), type=socket.SOCK_STREAM
        )
    except OSError:
        raise McpPolicyError(f"Cannot resolve host {host!r}.") from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if getattr(ip, "ipv4_mapped", None):
            ip = ip.ipv4_mapped
        if not ip.is_global:
            raise McpPolicyError("That address is not publicly reachable; only public servers can be added.")
