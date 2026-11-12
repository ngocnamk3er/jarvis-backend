"""MCP servers users add from the chat UI: what they may point at, how their
tokens are kept, and that one user's tools never reach another's model."""

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from fastapi import HTTPException
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agents import mcp_guard, user_mcp
from app.agents.mcp_guard import McpPolicyError, check_url
from app.agents.tool_search import ToolSearchMiddleware
from app.agents.user_mcp import UserMCPMiddleware, deferred_catalog
from app.clients import conversation_client
from app.core.config import settings
from app.db import mcp_repository
from app.services import mcp_service
from tests.test_skills_deferred_mcp import Scripted, call, web_search

HERE = Path(__file__).parent


# ── the URL rule ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8000/mcp", "http://localhost/mcp", "http://[::1]/mcp", "http://[::ffff:127.0.0.1]/mcp",
    "http://10.1.2.3/mcp", "http://192.168.0.10/mcp", "http://172.16.5.5/mcp",
    "http://169.254.169.254/latest/meta-data", "http://100.64.0.1/mcp",
    "http://keycloak.jarvis.svc.cluster.local:8080/realms", "http://no-such-host.invalid/mcp",
])
async def test_internal_addresses_are_refused(url):
    with pytest.raises(McpPolicyError):
        await check_url(url)


@pytest.mark.parametrize("url", [
    "ftp://93.184.216.34/mcp", "file:///etc/passwd", "gopher://93.184.216.34/", "http:///mcp", "93.184.216.34/mcp",
    "http://user:secret@93.184.216.34/mcp", "http://93.184.216.34:notaport/mcp", "https://93.184.216.34/" + "a" * 3000,
])
async def test_malformed_or_wrong_scheme_is_refused(url):
    with pytest.raises(McpPolicyError):
        await check_url(url)


async def test_a_public_address_passes_and_an_allowlisted_internal_host_passes(monkeypatch):
    await check_url("https://93.184.216.34/mcp")
    monkeypatch.setattr(settings, "MCP_USER_ALLOWED_HOSTS", "mcp-demo.jarvis.svc.cluster.local, 127.0.0.1")
    await check_url("http://127.0.0.1:9/mcp")
    await check_url("http://MCP-DEMO.jarvis.svc.cluster.local:8000/mcp")   # no DNS needed: it is named
    with pytest.raises(McpPolicyError):
        await check_url("http://10.1.2.3/mcp")                              # allowlist is exact, not a range


def test_names_transports_and_headers():
    assert mcp_guard.validate_name("my-docs1") == "my-docs1"
    for bad in ["", "Docs", "1docs", "a_b", "a" * 25, "docs space", "../x"]:
        with pytest.raises(McpPolicyError):
            mcp_guard.validate_name(bad)
    with pytest.raises(McpPolicyError):
        mcp_guard.validate_transport("stdio")           # would run a command on the backend
    mcp_guard.validate_transport("sse")
    mcp_guard.validate_headers({"Authorization": "Bearer x"})
    for bad in [{"Host": "evil"}, {"Content-Length": "1"}, {"bad name": "x"}, {"X": "a\r\nInjected: 1"}, {"X": "v" * 3000}]:
        with pytest.raises(McpPolicyError):
            mcp_guard.validate_headers(bad)


# ── secrets ────────────────────────────────────────────────────────────────

def test_headers_are_encrypted_and_round_trip(monkeypatch):
    monkeypatch.setattr(settings, "MCP_SECRET_KEY", Fernet.generate_key().decode())
    token = user_mcp.encrypt_headers({"Authorization": "Bearer s3cret"})
    assert "s3cret" not in token and "Authorization" not in token
    assert user_mcp.decrypt_headers(token) == {"Authorization": "Bearer s3cret"}
    assert user_mcp.header_names(token) == ["Authorization"]
    assert user_mcp.encrypt_headers({}) is None


def test_headers_are_refused_when_no_key_is_configured(monkeypatch):
    monkeypatch.setattr(settings, "MCP_SECRET_KEY", "")
    with pytest.raises(McpPolicyError, match="MCP_SECRET_KEY"):
        user_mcp.encrypt_headers({"Authorization": "x"})


def test_a_rotated_key_loses_the_headers_rather_than_crashing(monkeypatch):
    monkeypatch.setattr(settings, "MCP_SECRET_KEY", Fernet.generate_key().decode())
    token = user_mcp.encrypt_headers({"A": "b"})
    monkeypatch.setattr(settings, "MCP_SECRET_KEY", Fernet.generate_key().decode())
    assert user_mcp.decrypt_headers(token) == {}


# ── a real server, and an in-memory stand-in for the table ─────────────────

@pytest.fixture(scope="module")
def http_server():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([sys.executable, str(HERE / "mcp_demo_server.py"), "http", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("demo MCP server did not start")
    yield f"http://127.0.0.1:{port}/mcp"
    proc.kill()


@pytest.fixture
def world(monkeypatch):
    """Two users, an allowlisted loopback, a fake table, a fake conversation owner."""
    monkeypatch.setattr(settings, "MCP_USER_ALLOWED_HOSTS", "127.0.0.1")
    monkeypatch.setattr(settings, "MCP_SECRET_KEY", Fernet.generate_key().decode())
    user_mcp._cache.clear()
    rows: dict[str, dict] = {}
    off: set[tuple[str, str]] = set()
    n = [0]

    async def list_servers(user_id):
        return [r for r in rows.values() if r["user_id"] == user_id]

    async def get_server(user_id, sid):
        r = rows.get(sid)
        return r if r and r["user_id"] == user_id else None

    async def create_server(user_id, name, url, transport, headers_enc):
        if any(r["user_id"] == user_id and r["name"] == name for r in rows.values()):
            return None
        n[0] += 1
        rows[str(n[0])] = r = {"id": str(n[0]), "user_id": user_id, "name": name, "url": url,
                               "transport": transport, "headers_enc": headers_enc, "updated_at": "t"}
        return r

    async def delete_server(user_id, sid):
        return rows.pop(sid, None) is not None if (await get_server(user_id, sid)) else False

    async def disabled_ids(conv):
        return {sid for c, sid in off if c == conv}

    async def set_enabled(conv, sid, enabled):
        (off.discard if enabled else off.add)((conv, sid))

    async def list_enabled(user_id, conv):
        d = await disabled_ids(conv)
        return [r for r in await list_servers(user_id) if r["id"] not in d]

    for f in (list_servers, get_server, create_server, delete_server, disabled_ids, set_enabled, list_enabled):
        monkeypatch.setattr(mcp_repository, f.__name__, f)

    owners = {"conv-a": "alice", "conv-b": "bob"}

    class Conv:
        def __init__(self, user_id):
            self.user_id = user_id

    async def get_conversation(cid):
        return Conv(owners[cid]) if cid in owners else None

    monkeypatch.setattr(conversation_client, "get_conversation", get_conversation)
    return rows


async def test_adding_a_server_checks_it_answers_and_never_returns_the_token(world, http_server):
    out = await mcp_service.add_server("alice", "demo", http_server, "streamable_http", {"Authorization": "Bearer s3cret"})
    assert out["tools"] == ["mcp_demo_add", "mcp_demo_shout"]
    assert out["header_names"] == ["Authorization"] and "s3cret" not in str(out)
    assert "s3cret" not in world[out["id"]]["headers_enc"]                      # stored encrypted
    listing = await mcp_service.list_servers("alice", "conv-a")
    assert "s3cret" not in str(listing) and listing["servers"][0]["enabled"] is True


async def test_a_dead_or_internal_server_is_not_saved(world):
    for url, why in [("http://127.0.0.1:9/mcp", "Could not connect"), ("http://10.0.0.1/mcp", "not publicly reachable")]:
        with pytest.raises(HTTPException) as e:
            await mcp_service.add_server("alice", "bad", url, "streamable_http", {})
        assert e.value.status_code == 400 and why in e.value.detail
    assert world == {}


async def test_limits_and_duplicates(world, http_server, monkeypatch):
    await mcp_service.add_server("alice", "demo", http_server, "streamable_http", {})
    with pytest.raises(HTTPException) as e:
        await mcp_service.add_server("alice", "demo", http_server, "streamable_http", {})
    assert e.value.status_code == 409
    monkeypatch.setattr(settings, "MCP_USER_MAX_SERVERS", 1)
    with pytest.raises(HTTPException) as e:
        await mcp_service.add_server("alice", "other", http_server, "streamable_http", {})
    assert e.value.status_code == 400 and "At most 1" in e.value.detail
    with pytest.raises(HTTPException) as e:
        await mcp_service.add_server("alice", "stdio-one", "http://x", "stdio", {})
    assert e.value.status_code == 400


async def test_only_the_owner_can_remove_or_switch_a_server(world, http_server):
    sid = (await mcp_service.add_server("alice", "demo", http_server, "streamable_http", {}))["id"]
    with pytest.raises(HTTPException) as e:
        await mcp_service.remove_server("bob", sid)
    assert e.value.status_code == 404
    with pytest.raises(HTTPException) as e:
        await mcp_service.set_enabled("bob", "conv-b", sid, False)      # not bob's server
    assert e.value.status_code == 404
    with pytest.raises(HTTPException) as e:
        await mcp_service.set_enabled("alice", "conv-b", sid, False)    # not alice's conversation
    assert e.value.status_code == 404
    await mcp_service.remove_server("alice", sid)
    assert world == {}


# ── the model's side ───────────────────────────────────────────────────────

def agent(script, user_id, conv="conv-a"):
    model = Scripted(script=list(script), offered=[], systems=[])
    g = create_agent(
        model=model, tools=[web_search], system_prompt="You are Jarvis.", checkpointer=InMemorySaver(),
        middleware=[UserMCPMiddleware(), ToolSearchMiddleware(frozenset(), extra_catalog=deferred_catalog)],
    )
    return g, model, {"configurable": {"thread_id": conv, "user_id": user_id}}


async def test_a_users_server_is_deferred_loaded_on_request_and_run(world, http_server):
    await mcp_service.add_server("alice", "demo", http_server, "streamable_http", {})
    g, model, cfg = agent(
        [call("tool_search", {"query": "add integers"}), call("mcp_demo_add", {"a": 20, "b": 22}, "c2"), AIMessage("42")],
        "alice",
    )
    await g.ainvoke({"messages": [("user", "20+22")]}, cfg)
    first, second, _ = model.offered
    assert "mcp_demo_add" not in first and "tool_search" in first        # costs nothing until asked for
    assert "`mcp_demo_add`:" in model.systems[0]
    assert "mcp_demo_add" in second
    results = [m.content for m in (await g.aget_state(cfg)).values["messages"] if isinstance(m, ToolMessage)]
    assert any("42" in str(r) for r in results)


async def test_one_users_tools_never_reach_another_user(world, http_server):
    await mcp_service.add_server("alice", "demo", http_server, "streamable_http", {})
    # bob has no servers; his model must not be offered, told of, or able to load alice's tools
    g, model, cfg = agent([call("tool_search", {"query": "select:mcp_demo_add"}), AIMessage("done")], "bob", "conv-b")
    await g.ainvoke({"messages": [("user", "hi")]}, cfg)
    assert all("mcp_demo" not in " ".join(o) for o in model.offered)
    assert all("mcp_demo" not in s for s in model.systems)
    assert "tool_search" not in model.offered[0]                          # nothing deferred, no loader
    # and a call he makes anyway is not run
    g, model, cfg = agent([call("mcp_demo_add", {"a": 1, "b": 1}), AIMessage("done")], "bob", "conv-b")
    await g.ainvoke({"messages": [("user", "hi")]}, cfg)
    msg = next(m for m in (await g.aget_state(cfg)).values["messages"] if isinstance(m, ToolMessage))
    assert msg.status == "error"


async def test_a_server_switched_off_for_a_conversation_is_absent_there_only(world, http_server):
    sid = (await mcp_service.add_server("alice", "demo", http_server, "streamable_http", {}))["id"]
    await mcp_service.set_enabled("alice", "conv-a", sid, False)
    g, model, cfg = agent([AIMessage("done")], "alice", "conv-a")
    await g.ainvoke({"messages": [("user", "hi")]}, cfg)
    assert "mcp_demo" not in model.systems[0]
    assert (await mcp_service.list_servers("alice", "conv-a"))["servers"][0]["enabled"] is False
    await mcp_service.set_enabled("alice", "conv-a", sid, True)
    g, model, cfg = agent([AIMessage("done")], "alice", "conv-a")
    await g.ainvoke({"messages": [("user", "hi")]}, cfg)
    assert "mcp_demo" in model.systems[0]


async def test_a_dead_server_costs_one_try_not_one_per_model_call(world, http_server, monkeypatch):
    await mcp_service.add_server("alice", "demo", http_server, "streamable_http", {})
    user_mcp._cache.clear()
    tries = []
    real = user_mcp._connect

    async def counting(server):
        tries.append(server["name"])
        return user_mcp.ServerTools(error="down")

    monkeypatch.setattr(user_mcp, "_connect", counting)
    for _ in range(3):
        assert await user_mcp.tools_for_conversation("alice", "conv-a") == {}
    assert tries == ["demo"]
