"""Background research: what the main agent's chat stream sees, and what the
side panel reads. A scripted model and fake tools stand in for the LLM and the
web; the runner, the tools and the middleware are the real ones."""

import asyncio
import time

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from app.agents import async_research as ar


@tool
async def web_search(query: str, label: str) -> str:
    """search"""
    await asyncio.sleep(0.3)
    return "results for " + query


@tool
async def web_fetch(url: str, label: str) -> str:
    """fetch"""
    await asyncio.sleep(0.3)
    return "page " + url


class Researcher(BaseChatModel):
    """search, then fetch, then a final report"""

    @property
    def _llm_type(self):
        return "researcher"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        n = sum(isinstance(m, ToolMessage) for m in messages)
        if n == 0:
            msg = AIMessage("", tool_calls=[{"name": "web_search", "id": "s1",
                "args": {"query": "q", "label": "Searching q"}}])
        elif n == 1:
            msg = AIMessage("", tool_calls=[{"name": "web_fetch", "id": "f1",
                "args": {"url": "https://example.test/a", "label": "Reading a"}}])
        else:
            msg = AIMessage("FINAL REPORT")
        return ChatResult(generations=[ChatGeneration(message=msg)])


class Main(BaseChatModel):
    """starts a research task, then keeps the turn open for a while"""

    @property
    def _llm_type(self):
        return "main"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        if isinstance(messages[-1], HumanMessage):
            msg = AIMessage("", tool_calls=[{"name": "start_research", "id": "c1",
                "args": {"description": "compare A and B"}}])
        else:
            time.sleep(1.0)  # the turn is still streaming while the task starts working
            msg = AIMessage("started")
        return ChatResult(generations=[ChatGeneration(message=msg)])


def research_graph(saver):
    return create_agent(model=Researcher(), tools=[web_search, web_fetch], checkpointer=saver,
                        middleware=[TodoListMiddleware()])


@pytest.fixture
def runner():
    saver = InMemorySaver()
    ar.research_runner.init(saver, graph=research_graph(saver))
    yield ar.research_runner


async def test_the_tasks_own_steps_do_not_reach_the_chat_stream(runner):
    """The chat stream (astream_events) must report the main agent's own tool
    calls and nothing from inside the background task. Without an isolated
    context the task inherits the stream's event handler, and every search or
    fetch it runs appears in the main answer as a tool badge that never ends."""
    agent = create_agent(model=Main(), tools=[], middleware=[ar.BackgroundResearchMiddleware()],
                         checkpointer=InMemorySaver())
    tools_seen = []
    async for ev in agent.astream_events(
        {"messages": [HumanMessage("go")]},
        config={"configurable": {"thread_id": "conv-1"}}, version="v2",
    ):
        if ev["event"] == "on_tool_start":
            tools_seen.append(ev["name"])
    assert "start_research" in tools_seen
    assert "web_search" not in tools_seen and "web_fetch" not in tools_seen, tools_seen

    # ... and the task still ran to the end in the background
    (task_id,) = list(runner._jobs)
    await runner._jobs[task_id].task
    status, _, values = await runner.inspect(task_id, "running")
    assert status == "success" and ar._final_report(values) == "FINAL REPORT"


async def test_describe_tasks_is_what_the_panel_needs(runner):
    first = runner.start("first question", {})
    await runner._jobs[first].task
    second = runner.start("second question", {})
    await asyncio.sleep(0.45)  # mid-flight: one search done, working on the fetch

    tasks = {
        first: {"task_id": first, "status": "running", "description": "first question",
                "created_at": "2026-01-01T10:00:00Z"},
        second: {"task_id": second, "status": "running", "description": "second question",
                 "created_at": "2026-01-01T10:05:00Z"},
    }
    rows = await ar.describe_tasks(tasks)

    assert [r["task_id"] for r in rows] == [second, first], "newest first"
    done, running = rows[1], rows[0]
    assert done["status"] == "success" and done["report"] == "FINAL REPORT"
    assert done["steps_total"] == 2 and done["steps_done"] == 2
    assert [s["what"] for s in done["recent_steps"]] == ["Searching q", "Reading a"]
    assert running["status"] == "running" and "report" not in running
    assert running["steps_total"] >= 1 and running["description"] == "second question"
    await runner._jobs[second].task


async def test_describe_tasks_survives_an_unreadable_task(runner, monkeypatch):
    async def boom(*_a, **_k):
        raise RuntimeError("checkpointer unavailable")

    monkeypatch.setattr(runner, "inspect", boom)
    rows = await ar.describe_tasks({"x": {"task_id": "x", "status": "running",
                                          "description": "d", "created_at": "2026-01-01T00:00:00Z"}})
    assert rows[0]["status"] == "running" and rows[0]["steps_total"] == 0


class Looper(BaseChatModel):
    """never stops searching"""

    @property
    def _llm_type(self):
        return "looper"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        n = sum(isinstance(m, ToolMessage) for m in messages)
        msg = AIMessage("", tool_calls=[{"name": "web_search", "id": f"s{n}",
            "args": {"query": f"q{n}", "label": f"Searching q{n}"}}])
        return ChatResult(generations=[ChatGeneration(message=msg)])


async def test_a_task_cut_off_by_its_search_limit_is_not_reported_as_having_a_report(monkeypatch):
    """Ending at the hard limit leaves the stop notice where a report would be.
    It must read as an incomplete task, not as a finished one."""
    from app.agents.middleware import SoftHardToolCallLimitMiddleware

    saver = InMemorySaver()
    graph = create_agent(
        model=Looper(), tools=[web_search], checkpointer=saver,
        middleware=[SoftHardToolCallLimitMiddleware(tool_name="web_search", soft_limit=1, hard_limit=2)],
    )
    runner = ar.ResearchRunner()
    runner.init(saver, graph=graph)
    task_id = runner.start("a question too big for the budget", {})
    await runner._jobs[task_id].task

    monkeypatch.setattr(ar, "research_runner", runner)  # describe_tasks reads the module's runner
    rows = await ar.describe_tasks({task_id: {"task_id": task_id, "status": "running",
                                              "description": "d", "created_at": "2026-01-01T00:00:00Z"}})
    row = rows[0]
    assert row["status"] == "success"
    assert "hard limit exceeded" in row["stopped_early"]
    assert "report" not in row
