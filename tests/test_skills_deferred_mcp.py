"""Skills, deferred tools and MCP, through a real create_agent graph with a
scripted model: what the model is offered on each call, and what happens when
it asks for more."""

import json
import sys
from pathlib import Path

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.mcp_tools import load_mcp_tools, parse_mcp_servers
from app.agents.middleware import ContextBreakdownMiddleware
from app.agents.skills import SkillsMiddleware, load_skills, parse_skill
from app.agents.tool_search import ToolSearchMiddleware

HERE = Path(__file__).parent


@tool
async def web_search(query: str) -> str:
    """Search the web."""
    return "results"


calls: list[str] = []


@tool
async def render_chart(data: str) -> str:
    """Render a chart from data as an SVG image. Rarely needed, with a long schema."""
    calls.append("render_chart")
    return "<svg/>"


@tool
async def export_csv(rows: str) -> str:
    """Export rows to a CSV file."""
    calls.append("export_csv")
    return "ok"


class Scripted(BaseChatModel):
    """Plays a fixed list of replies; records what each call was offered."""

    script: list = []
    offered: list = []   # tool names per call
    systems: list = []   # system prompt per call

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kw):
        self.offered.append([t.name if hasattr(t, "name") else (t.get("function") or t).get("name") for t in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.systems.append(messages[0].content if isinstance(messages[0].content, str)
                            else "".join(b.get("text", "") for b in messages[0].content))
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


def call(name, args, id_="c1"):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": id_}])


def make(script, middleware, tools=None):
    model = Scripted(script=list(script), offered=[], systems=[])
    g = create_agent(model=model, tools=tools or [web_search, render_chart, export_csv],
                     system_prompt="You are Jarvis.", checkpointer=InMemorySaver(), middleware=middleware)
    return g, model, {"configurable": {"thread_id": "t"}}


# ── skills ─────────────────────────────────────────────────────────────────

def write_skill(root, name, desc="Does a thing. Use when asked.", body="Step one.", dirname=None):
    d = root / (dirname or name)
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\n---\n{body}\n")


def test_shipped_skills_are_valid():
    skills = load_skills(HERE.parent / "app" / "skills")
    assert {"research-report", "spreadsheet-analysis"} <= set(skills)


def test_bad_skills_are_skipped_not_fatal(tmp_path):
    write_skill(tmp_path, "good-one")
    write_skill(tmp_path, "wrong-name", dirname="other-dir")       # name must match its directory
    write_skill(tmp_path, "Bad_Name")                              # lowercase letters, digits, hyphens
    write_skill(tmp_path, "no-body", body="")
    (tmp_path / "no-frontmatter").mkdir()
    (tmp_path / "no-frontmatter" / "SKILL.md").write_text("just text")
    assert list(load_skills(tmp_path)) == ["good-one"]
    assert parse_skill(tmp_path / "nonexistent" / "SKILL.md") is None


async def test_skill_is_listed_cheaply_and_loaded_on_demand(tmp_path):
    write_skill(tmp_path, "pdf-notes", desc="Summarise a PDF.", body="BODY-OF-THE-SKILL " * 50)
    g, model, cfg = make(
        [call("load_skill", {"name": "pdf-notes"}), AIMessage("done")],
        [SkillsMiddleware(load_skills(tmp_path))],
    )
    await g.ainvoke({"messages": [("user", "hi")]}, cfg)
    assert "`pdf-notes`: Summarise a PDF." in model.systems[0]
    assert "BODY-OF-THE-SKILL" not in model.systems[0]            # only the one-liner costs context
    assert "load_skill" in model.offered[0]
    tool_msg = next(m for m in (await g.aget_state(cfg)).values["messages"] if isinstance(m, ToolMessage))
    assert "BODY-OF-THE-SKILL" in tool_msg.content


async def test_unknown_skill_says_what_exists(tmp_path):
    write_skill(tmp_path, "pdf-notes")
    g, _, cfg = make([call("load_skill", {"name": "nope"}), AIMessage("done")],
                     [SkillsMiddleware(load_skills(tmp_path))])
    await g.ainvoke({"messages": [("user", "hi")]}, cfg)
    msg = next(m for m in (await g.aget_state(cfg)).values["messages"] if isinstance(m, ToolMessage))
    assert "no skill named 'nope'" in msg.content and "pdf-notes" in msg.content


# ── deferred tools ─────────────────────────────────────────────────────────

async def test_deferred_tool_is_hidden_until_loaded_then_offered_and_runs():
    calls.clear()
    g, model, cfg = make(
        [call("tool_search", {"query": "select:render_chart"}),
         call("render_chart", {"data": "1,2"}, "c2"),
         AIMessage("done")],
        [ToolSearchMiddleware(frozenset({"render_chart"}))],
    )
    await g.ainvoke({"messages": [("user", "chart please")]}, cfg)

    first, second, third = model.offered
    assert "render_chart" not in first and "tool_search" in first and "web_search" in first
    assert "`render_chart`:" in model.systems[0] and "Deferred tools" in model.systems[0]
    assert "render_chart" in second                                  # loaded: schema now offered
    assert "tool_search" not in second                               # nothing left to load
    assert "Deferred tools" not in model.systems[1]
    assert calls == ["render_chart"]
    assert (await g.aget_state(cfg)).values["loaded_tools"] == ["render_chart"]


async def test_calling_a_deferred_tool_before_loading_is_refused_and_not_run():
    calls.clear()
    g, _, cfg = make(
        [call("render_chart", {"data": "x"}), AIMessage("done")],
        [ToolSearchMiddleware(frozenset({"render_chart"}))],
    )
    await g.ainvoke({"messages": [("user", "hi")]}, cfg)
    msg = next(m for m in (await g.aget_state(cfg)).values["messages"] if isinstance(m, ToolMessage))
    assert msg.status == "error" and "tool_search" in msg.content
    assert calls == []


async def test_keyword_search_loads_the_best_matches():
    g, model, cfg = make(
        [call("tool_search", {"query": "make a csv export"}), AIMessage("done")],
        [ToolSearchMiddleware(frozenset({"render_chart", "export_csv"}))],
    )
    await g.ainvoke({"messages": [("user", "hi")]}, cfg)
    st = (await g.aget_state(cfg)).values
    assert st["loaded_tools"] == ["export_csv"]
    assert "render_chart" not in model.offered[1] and "export_csv" in model.offered[1]


async def test_a_search_that_matches_nothing_loads_nothing():
    g, _, cfg = make([call("tool_search", {"query": "zebra"}), AIMessage("done")],
                     [ToolSearchMiddleware(frozenset({"render_chart"}))])
    await g.ainvoke({"messages": [("user", "hi")]}, cfg)
    st = (await g.aget_state(cfg)).values
    assert st["loaded_tools"] == []
    msg = next(m for m in st["messages"] if isinstance(m, ToolMessage))
    assert "No deferred tool matched" in msg.content and "render_chart" in msg.content


async def test_with_nothing_deferred_the_loader_is_not_even_offered():
    g, model, cfg = make([AIMessage("done")], [ToolSearchMiddleware(frozenset({"not_a_tool"}))])
    await g.ainvoke({"messages": [("user", "hi")]}, cfg)
    assert "tool_search" not in model.offered[0] and "Deferred tools" not in model.systems[0]


# ── context panel numbers ──────────────────────────────────────────────────

async def test_breakdown_separates_skills_deferred_and_mcp(tmp_path):
    write_skill(tmp_path, "pdf-notes", desc="Summarise a PDF. " * 10)
    g, _, cfg = make(
        [AIMessage("done")],
        [SkillsMiddleware(load_skills(tmp_path)),
         ToolSearchMiddleware(frozenset({"render_chart"})),
         ContextBreakdownMiddleware(mcp_tools=frozenset({"export_csv"}))],
    )
    await g.ainvoke({"messages": [("user", "hi")]}, cfg)
    b = (await g.aget_state(cfg)).values["context_breakdown"]
    names = {d["name"]: d["kind"] for d in b["tool_detail"]}
    assert "render_chart" not in names                               # not sent, so not counted as sent
    assert names["export_csv"] == "mcp" and names["web_search"] == "builtin"
    assert b["mcp_tools"] > 0 and b["tools"] > 0 and b["skills"] > 0
    assert [d["name"] for d in b["deferred"]["detail"]] == ["render_chart"] and b["deferred"]["tokens"] > 0


# ── MCP ────────────────────────────────────────────────────────────────────

def test_parse_servers(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCS_TOKEN", "s3cret")
    got = parse_mcp_servers(json.dumps({
        "docs": {"url": "https://x.test/mcp", "headers": {"Authorization": "Bearer ${DOCS_TOKEN}"}},
        "files": {"command": "npx", "args": ["server", "${DOCS_TOKEN}"]},
        "old": {"url": "https://y.test/sse", "transport": "sse"},
        "broken-stdio": {"transport": "stdio"},
        "broken-http": {"transport": "streamable_http"},
        "not-an-object": 3,
    }))
    assert set(got) == {"docs", "files", "old"}
    assert got["docs"]["transport"] == "streamable_http" and got["docs"]["headers"]["Authorization"] == "Bearer s3cret"
    assert got["files"]["transport"] == "stdio" and got["files"]["args"] == ["server", "s3cret"]
    f = tmp_path / "mcp.json"
    f.write_text('{"a": {"url": "https://z.test/mcp"}}')
    assert set(parse_mcp_servers(str(f))) == {"a"}                   # a file path works too


@pytest.mark.parametrize("raw", ["", "   ", "not json", "[1,2]", "/no/such/file.json"])
def test_unusable_config_means_no_servers(raw):
    assert parse_mcp_servers(raw) == {}


def demo_server():
    return {"demo": {"transport": "stdio", "command": sys.executable, "args": [str(HERE / "mcp_demo_server.py")]}}


async def test_loads_tools_from_a_real_server_and_skips_a_dead_one():
    servers = {**demo_server(), "dead": {"transport": "streamable_http", "url": "http://127.0.0.1:9/mcp"}}
    tools = await load_mcp_tools(servers)
    assert sorted(t.name for t in tools) == ["demo_add", "demo_shout"]   # prefixed; dead server skipped
    add = next(t for t in tools if t.name == "demo_add")
    assert "5" in str(await add.ainvoke({"a": 2, "b": 3}))


async def test_mcp_tool_end_to_end_deferred_then_called():
    tools = await load_mcp_tools(demo_server())
    names = frozenset(t.name for t in tools)
    g, model, cfg = make(
        [call("tool_search", {"query": "add integers sum"}),
         call("demo_add", {"a": 20, "b": 22}, "c2"),
         AIMessage("42")],
        [ToolSearchMiddleware(names), ContextBreakdownMiddleware(mcp_tools=names)],
        tools=[web_search, *tools],
    )
    await g.ainvoke({"messages": [("user", "20+22 with the demo server")]}, cfg)
    assert not names & set(model.offered[0])                         # a server's tools cost nothing up front …
    assert "demo_add" in model.offered[1]                            # … until the model asks for one
    results = [m.content for m in (await g.aget_state(cfg)).values["messages"] if isinstance(m, ToolMessage)]
    assert any("42" in str(r) for r in results)


# ── what the chat UI gets for an MCP tool result ───────────────────────────

def test_mcp_content_blocks_become_text():
    from app.agents.tool_output import tool_output_text

    assert tool_output_text("plain") == "plain"
    assert tool_output_text([{"type": "text", "text": "42"}]) == "42"
    assert tool_output_text([{"type": "text", "text": "a"}, {"type": "image", "data": "..."}]) == "a\n[image]"


async def test_the_stream_sends_an_mcp_result_as_a_string():
    from app.services.chat_service import ToolEndEventHandler

    tools = await load_mcp_tools(demo_server())
    add = next(t for t in tools if t.name == "demo_add")
    raw = await add.ainvoke({"type": "tool_call", "id": "x", "name": "demo_add", "args": {"a": 20, "b": 22}})
    ev = ToolEndEventHandler().handle({"event": "on_tool_end", "name": "demo_add", "run_id": "r", "data": {"output": raw}})
    assert ev and all(isinstance(e["output"], str) for e in ev if e["type"] == "tool_end")
    assert "42" in ev[-1]["output"]
