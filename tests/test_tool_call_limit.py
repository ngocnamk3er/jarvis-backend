"""SoftHardToolCallLimitMiddleware: past the hard limit the model writes its answer with what it has, instead of the
run ending with nothing.

Seen on the test cluster: background research called web_search in parallel batches, and one batch crossed the soft and
the hard limit in the same step (20 -> 25 searches, limits 20/24). The run ended there with a stop notice and no report,
although it had 20 search results in hand."""

import asyncio

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from app.agents import async_research as ar
from app.agents.middleware import SoftHardToolCallLimitMiddleware
from app.agents.research_prompt import research_system_prompt

ran: list[str] = []


@tool
async def web_search(query: str) -> str:
    """search"""
    ran.append("web_search")
    return "results for " + query


@tool
async def web_fetch(url: str) -> str:
    """fetch"""
    ran.append("web_fetch")
    return "page " + url


class Batcher(BaseChatModel):
    """Calls tools in parallel batches, the way the research model does, until it is told it may not call tools; then
    it writes its report. `obeys=False` is a model that keeps calling tools whatever tool_choice says."""

    calls: list[str] = ["web_search"] * 5
    obeys: bool = True
    choices: list = []

    @property
    def _llm_type(self):
        return "batcher"

    def bind_tools(self, tools, tool_choice=None, **kw):
        return self.bind(tool_choice=tool_choice)

    def _generate(self, messages, stop=None, run_manager=None, tool_choice=None, **kw):
        self.choices.append(tool_choice)
        if tool_choice == "none" and self.obeys:
            msg = AIMessage("REPORT from what I have")
        else:
            step = sum(isinstance(m, AIMessage) for m in messages)
            msg = AIMessage("", tool_calls=[
                {"name": name, "id": f"c{step}-{i}", "args": {"query": f"q{step}-{i}"} if name == "web_search"
                 else {"url": f"u{step}-{i}"}}
                for i, name in enumerate(self.calls)])
        return ChatResult(generations=[ChatGeneration(message=msg)])


def agent(model, checkpointer=None, search=(20, 24), fetch=(12, 15)):
    return create_agent(
        model=model, tools=[web_search, web_fetch], checkpointer=checkpointer,
        middleware=[SoftHardToolCallLimitMiddleware(tool_name="web_search", soft_limit=search[0], hard_limit=search[1]),
                    SoftHardToolCallLimitMiddleware(tool_name="web_fetch", soft_limit=fetch[0], hard_limit=fetch[1])],
    )


def unanswered(messages) -> list[str]:
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    return [c["id"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls if c["id"] not in answered]


async def test_a_batch_that_crosses_both_limits_still_ends_with_a_report():
    ran.clear()
    model = Batcher()
    out = await agent(model).ainvoke({"messages": [HumanMessage("compare five databases")]})
    msgs = out["messages"]
    assert msgs[-1].content == "REPORT from what I have"
    assert ran.count("web_search") == 20                    # 4 batches of 5 ran; the 5th (calls 21-25) did not
    assert model.choices == [None] * 5 + ["none"]           # only the last turn was barred from calling tools
    blocked = [m for m in msgs if isinstance(m, ToolMessage) and m.status == "error"]
    assert len(blocked) == 5 and all("Write your final answer now" in m.content for m in blocked)
    assert unanswered(msgs) == []


async def test_another_tool_called_in_the_same_step_still_runs():
    """This used to raise NotImplementedError: ending the run would have left the other call without a response."""
    ran.clear()
    model = Batcher(calls=["web_search", "web_search", "web_search", "web_fetch"])
    out = await agent(model, search=(2, 3)).ainvoke({"messages": [HumanMessage("q")]})
    assert out["messages"][-1].content == "REPORT from what I have"
    # Step 1: searches 1-2 ran, 3 was blocked (soft), the fetch ran. Step 2: searches 4-6 were blocked (hard) and the
    # fetch beside them still ran. Step 3: the report.
    assert sorted(ran) == ["web_fetch", "web_fetch", "web_search", "web_search"]
    assert model.choices == [None, None, "none"]
    assert unanswered(out["messages"]) == []


async def test_a_model_that_ignores_tool_choice_is_stopped_with_every_call_answered():
    ran.clear()
    mw = SoftHardToolCallLimitMiddleware(tool_name="web_search", soft_limit=2, hard_limit=3)
    model = Batcher(calls=["web_search", "web_fetch"], obeys=False)
    graph = create_agent(model=model, tools=[web_search, web_fetch], middleware=[mw])
    out = await graph.ainvoke({"messages": [HumanMessage("q")]})
    msgs = out["messages"]
    assert ar._LIMIT_STOP.search(msgs[-1].content)          # async_research still tells it from a finished run
    assert unanswered(msgs) == []                            # the next turn of the conversation can be sent
    assert model.choices[-1] == "none"
    assert len(model.choices) == 5                           # no endless loop: one refused turn, then the end


async def test_the_next_run_on_the_thread_starts_with_a_full_budget():
    ran.clear()
    saver = InMemorySaver()
    cfg = {"configurable": {"thread_id": "t"}}
    model = Batcher(calls=["web_search"] * 3)
    graph = agent(model, checkpointer=saver, search=(2, 3))
    await graph.ainvoke({"messages": [HumanMessage("first")]}, cfg)
    model.choices.clear()
    ran.clear()
    out = await graph.ainvoke({"messages": [HumanMessage("second")]}, cfg)
    assert model.choices[0] is None                          # tools are offered again
    assert ran.count("web_search") == 2
    assert out["messages"][-1].content == "REPORT from what I have"


async def test_a_background_task_that_hits_the_limit_has_a_report(monkeypatch):
    saver = InMemorySaver()
    runner = ar.ResearchRunner()
    runner.init(saver, graph=agent(Batcher(), checkpointer=saver))
    task_id = runner.start("compare five databases", {})
    await asyncio.wait_for(runner._jobs[task_id].task, 10)

    monkeypatch.setattr(ar, "research_runner", runner)
    rows = await ar.describe_tasks({task_id: {"task_id": task_id, "status": "running",
                                              "description": "d", "created_at": "2026-01-01T00:00:00Z"}})
    assert rows[0]["status"] == "success"
    assert rows[0].get("report") == "REPORT from what I have"
    assert not rows[0].get("stopped_early")


def test_the_research_prompt_states_the_budget():
    prompt = research_system_prompt(sandbox=False, budget={"web_search": 20, "web_fetch": 12})
    assert "at most 20 `web_search`, 12 `web_fetch` calls" in prompt
    assert "## Budget" not in research_system_prompt(sandbox=False)
