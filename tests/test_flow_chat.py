"""Running a user's flow from chat: offered as a tool, run inside the turn, its steps nested under its badge, its model's
tokens kept out of the answer, and nothing of it checkpointed into the conversation."""

import json
from types import SimpleNamespace

import pytest
from app.agents import user_flows
from app.db import flow_repository
from app.flows import catalog, compiler
from app.flows.runner import run_flow
from app.flows.schema import validate_flow
from app.services import chat_service as cs
from app.services.conversation_service import _replay_subagent_trace, serialize_messages
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from tests import flow_fakes
from tests.flow_fakes import TOOLS, EchoModel, flow, news_flow, node


def row(definition, slug="tin_tuc", name="Tin tức", version=1, description="Searches the news and summarises it."):
    return {"id": f"id-{slug}", "name": name, "slug": slug, "description": description, "version": version,
            "enabled": True, "definition": definition}


@pytest.fixture
def flows(monkeypatch):
    """The user's saved flows, as list_enabled would return them; the flows run on the fake model and tools."""
    saved: list[dict] = []

    async def list_enabled(user_id):
        return saved if user_id == "u1" else []

    monkeypatch.setattr(flow_repository, "list_enabled", list_enabled)
    monkeypatch.setattr(catalog, "flow_tools", lambda: TOOLS)
    real_compile = compiler.compile_flow
    monkeypatch.setattr("app.flows.runner.compile_flow",
                        lambda f, llm=None, tools=None: real_compile(f, llm=EchoModel(), tools=TOOLS))
    user_flows._parsed.clear()
    flow_fakes.tool_calls.clear()
    return saved


@tool
def clock() -> str:
    """The time."""
    return "noon"


class Caller(BaseChatModel):
    """Calls `name` with `args` once, then answers with the tool's result."""
    name: str = ""
    args: dict = {}
    offered: list = []

    @property
    def _llm_type(self):
        return "caller"

    def bind_tools(self, tools, **kw):
        self.offered.append([getattr(t, "name", None) or t.get("name") for t in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if results:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(f"answer: {results[-1].content}"))])
        return ChatResult(generations=[ChatGeneration(message=AIMessage("", tool_calls=[
            {"name": self.name, "id": "c1", "args": self.args}]))])


async def chat(model, config=None, checkpointer=None):
    # `clock` is there so the agent has a tool node at all, as the real one always does
    agent = create_agent(model=model, tools=[clock], middleware=[user_flows.UserFlowsMiddleware()],
                         checkpointer=checkpointer)
    cfg = {"configurable": {"user_id": "u1", "thread_id": "t1", "model": "chat-model", **(config or {})}}
    out = await agent.ainvoke({"messages": [HumanMessage("go")]}, config=cfg)
    return next(m for m in out["messages"] if isinstance(m, ToolMessage)), out


async def test_each_switched_on_flow_is_a_tool_with_its_inputs_as_arguments(flows):
    flows.append(row(news_flow()))
    tools = await user_flows.tools_for_user("u1")
    t = tools["flow_tin_tuc"]
    assert t.args_schema == {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}
    assert t.description.startswith("Searches the news and summarises it.") and '"Tin tức"' in t.description
    assert t.metadata["flow_name"] == "Tin tức"
    assert await user_flows.tools_for_user("someone_else") == {}


async def test_a_saved_flow_that_no_longer_validates_is_not_offered(flows):
    broken = news_flow()
    broken["nodes"][1]["config"]["tool"] = "a_tool_since_removed"
    flows.extend([row(news_flow()), row(broken, slug="old")])
    assert list(await user_flows.tools_for_user("u1")) == ["flow_tin_tuc"]


async def test_the_model_calls_a_flow_and_gets_its_result(flows):
    flows.append(row(news_flow()))
    model = Caller(name="flow_tin_tuc", args={"topic": "Huế"})
    result, out = await chat(model)
    assert "flow_tin_tuc" in model.offered[0]
    assert result.status == "success"
    assert result.content == "Summary: [chat-model] Summarise: results for Huế news"
    # the flow's tools ran as the user, with web results inline
    assert flow_fakes.tool_calls == [{"tool": "web_search", "query": "Huế news", "label": "Step s · web_search",
                                      "user_id": "u1", "sandbox": False}]


async def test_a_flow_that_uses_a_switched_off_tool_says_so_and_does_not_run(flows):
    flows.append(row(news_flow()))
    result, _ = await chat(Caller(name="flow_tin_tuc", args={"topic": "x"}),
                           config={"disabled_tools": ["web_search", "web_fetch"]})
    assert result.status == "error" and "web_search, which is turned off" in result.content
    assert flow_fakes.tool_calls == []


async def test_a_failing_step_is_reported_to_the_model_by_name(flows):
    flows.append(row(flow([node("in", "input", fields=[{"name": "q"}]),
                           node("t", "tool", tool="broken", args={"query": "{{in.q}}"}),
                           node("out", "output", template="{{t}}")], [("in", "t"), ("t", "out")])))
    result, _ = await chat(Caller(name="flow_tin_tuc", args={"q": "x"}))
    assert result.status == "error" and "stopped at step 't'" in result.content and "upstream is down" in result.content


async def test_missing_inputs_are_reported_to_the_model(flows):
    flows.append(row(news_flow()))
    result, _ = await chat(Caller(name="flow_tin_tuc", args={}))
    assert result.status == "error" and "'topic' is required" in result.content


async def test_a_flow_run_from_chat_writes_no_checkpoints_into_the_conversation(flows):
    flows.append(row(news_flow()))
    saver = InMemorySaver()
    result, _ = await chat(Caller(name="flow_tin_tuc", args={"topic": "x"}), checkpointer=saver)
    assert result.status == "success"
    assert list(saver.storage["t1"]) == [""]          # only the conversation's own namespace, no `tools:...` one


async def test_a_long_chain_of_nodes_fits_in_the_recursion_limit():
    n = 30                                            # more steps than LangGraph's default limit of 25
    nodes = [node("in", "input", fields=[{"name": "q"}])] + \
            [node(f"s{i}", "llm", prompt=f"{i}") for i in range(n)] + [node("out", "output", template="{{s29}}")]
    chain = ["in"] + [f"s{i}" for i in range(n)] + ["out"]
    parsed, problems = validate_flow(flow(nodes, list(zip(chain, chain[1:], strict=False))), catalog.tool_specs(TOOLS))
    assert problems == []
    result = await run_flow(parsed, {"q": "x"}, {}, llm=EchoModel(), tools=TOOLS)
    assert result.text == "[None] 29"


# ---- the chat stream ------------------------------------------------------------------------------------------------

class StreamGraph:
    """Replays the events a turn that calls a flow produces."""

    def __init__(self, events):
        self.events = events

    async def astream_events(self, graph_input, config=None, version="v2", **kw):
        for ev in self.events:
            yield ev

    async def aget_state(self, config):
        return SimpleNamespace(values={"messages": [HumanMessage("go"), AIMessage("done", id="m1")]},
                               interrupts=(), next=())

    async def aupdate_state(self, config, values, **kw):
        pass


def ev(kind, name, run_id, parents=(), **data):
    return {"event": kind, "name": name, "run_id": run_id, "parent_ids": list(parents), "metadata": data.pop("metadata", {}),
            "data": data}


async def test_the_chat_stream_nests_a_flows_steps_under_it_keeps_its_tokens_out_and_saves_them(monkeypatch):
    saved = {}

    async def save_subagent_trace(thread_id, tool_call_id, trace):
        saved[tool_call_id] = trace

    async def nothing(*a, **kw):
        return None

    monkeypatch.setattr(cs, "conversation_client", SimpleNamespace(
        save_subagent_trace=save_subagent_trace, save_message_trace=nothing, set_context_tokens=nothing))
    monkeypatch.setattr(cs, "reset_sandbox", nothing)
    monkeypatch.setattr(cs, "prune_thread", nothing)

    events = [
        ev("on_tool_start", "flow_tin_tuc", "F", input={"topic": "Huế"}, metadata={"flow_name": "Tin tức"}),
        ev("on_custom_event", "flow_node", "x1", ["F"], node="sum", type="llm", phase="start"),
        ev("on_chat_model_stream", "model", "m", ["F", "x1"], chunk=AIMessageChunk(content="the flow's own words")),
        ev("on_tool_start", "web_search", "W", ["F"], input={"query": "Huế news", "label": "Step s · web_search"}),
        ev("on_tool_end", "web_search", "W", ["F"], output=ToolMessage("results", tool_call_id="ws")),
        ev("on_custom_event", "flow_node", "x2", ["F"], node="sum", type="llm", phase="end", output={"text": "short"}),
        ev("on_tool_end", "flow_tin_tuc", "F", output=ToolMessage("Summary: short", tool_call_id="c1")),
    ]
    lines = [json.loads(line[6:]) async for line in cs.ChatService()._run_graph(
        {"messages": []}, {"configurable": {"thread_id": "t1"}}, StreamGraph(events))]

    start = next(e for e in lines if e["type"] == "tool_start" and e["name"] == "flow_tin_tuc")
    assert start["label"] == "Flow: Tin tức" and "task_run_id" not in start
    steps = [e for e in lines if e["type"] == "flow_node"]
    assert [(s["node"], s["node_type"], s["phase"], s["task_run_id"]) for s in steps] == [
        ("sum", "llm", "start", "F"), ("sum", "llm", "end", "F")]
    assert steps[1]["output"] == {"text": "short"}
    assert next(e for e in lines if e["type"] == "tool_start" and e["name"] == "web_search")["task_run_id"] == "F"
    assert not any(e["type"] == "token" for e in lines)          # the flow's model is not the answer
    trace = saved["c1"]
    assert [e["type"] for e in trace] == ["flow", "flow_node", "tool_start", "tool_end", "flow_node"]
    assert trace[0] == {"type": "flow", "label": "Flow: Tin tức"}      # what a reload calls the badge


def test_a_reload_shows_a_flows_steps_again_under_its_badge():
    trace = [
        {"type": "flow_node", "node": "sum", "node_type": "llm", "phase": "start", "task_run_id": "F"},
        {"type": "tool_start", "name": "web_search", "label": "Step s · web_search", "input": {"query": "q"}, "run_id": "W"},
        {"type": "tool_end", "name": "web_search", "output": "results", "run_id": "W"},
        {"type": "flow_node", "node": "sum", "node_type": "llm", "phase": "end", "output": {"text": "short"},
         "task_run_id": "F"},
    ]
    parts = _replay_subagent_trace(trace, "c1")
    assert [p["type"] for p in parts] == ["tool", "flow_step"]
    assert parts[0]["tool"]["task_run_id"] == "c1" and parts[0]["tool"]["output"] == "results"
    assert parts[1] == {"type": "flow_step", "node": "sum", "node_type": "llm", "status": "done",
                        "output": {"text": "short"}, "task_run_id": "c1"}


def test_after_a_reload_the_flows_badge_still_has_the_flows_name():
    """Seen in a screenshot of the reloaded chat: the badge said "Running flow", since a flow's call has no `label`
    argument. The name is now saved with the flow's trace."""
    messages = [HumanMessage("go", id="h"),
                AIMessage("", id="a", tool_calls=[{"name": "flow_tin_tuc", "id": "c1", "args": {"topic": "Huế"}}]),
                ToolMessage("Summary", tool_call_id="c1", name="flow_tin_tuc", id="t"), AIMessage("done", id="b")]
    trace = [{"type": "flow", "label": "Flow: Tin tức"},
             {"type": "flow_node", "node": "sum", "node_type": "llm", "phase": "end", "output": {"text": "x"}}]
    parts = [p for m in serialize_messages(messages, {"c1": trace}) for p in m["parts"]]
    badge = next(p["tool"] for p in parts if p["type"] == "tool")
    assert badge["label"] == "Flow: Tin tức" and badge["input"] == {"topic": "Huế"} and badge["run_id"] == "c1"
    assert any(p["type"] == "flow_step" and p["task_run_id"] == "c1" for p in parts)
