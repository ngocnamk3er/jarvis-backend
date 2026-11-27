"""Compiling and running a flow: what each node gets and produces, the order of parallel branches, the events a run
reports, and that a flow run inside a chat turn's tool reports them under that tool."""

import asyncio
import json

import pytest
from app.flows import runner
from app.flows.catalog import tool_specs
from app.flows.compiler import FlowNodeError
from app.flows.runner import FlowInputError, run_flow, stream_flow
from app.flows.schema import validate_flow
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from tests import flow_fakes
from tests.flow_fakes import TOOLS, EchoModel, flow, news_flow, node

CONFIG = {"configurable": {"user_id": "u1", "model": "chat-model", "sandbox": False}}


def parse(definition):
    parsed, problems = validate_flow(definition, tool_specs(TOOLS))
    assert problems == []
    return parsed


@pytest.fixture(autouse=True)
def clean():
    flow_fakes.tool_calls.clear()


async def test_each_node_gets_what_the_nodes_before_it_produced():
    llm = EchoModel()
    result = await run_flow(parse(news_flow()), {"topic": "Hà Nội"}, CONFIG, llm=llm, tools=TOOLS)
    assert flow_fakes.tool_calls == [{"tool": "web_search", "query": "Hà Nội news", "label": "Flow step s",
                                      "user_id": "u1", "sandbox": False}]
    assert llm.calls[0]["prompt"] == "Summarise: results for Hà Nội news"
    assert result.text == "Summary: [chat-model] Summarise: results for Hà Nội news"
    assert result.outputs["in"] == {"topic": "Hà Nội"}


async def test_parallel_branches_run_together_and_the_join_runs_once_both_are_done():
    gate = asyncio.Event()
    started = []

    class Slow(EchoModel):
        async def _agenerate(self, messages, stop=None, run_manager=None, **kw):   # ainvoke, not streamed
            prompt = messages[-1].content
            started.append(prompt)
            if prompt.startswith("A "):
                await gate.wait()             # A cannot finish before B has started: they run at the same time
            return self._generate(messages, stop, run_manager, **kw)

    async def both_started():
        while len(started) < 2:
            await asyncio.sleep(0.01)

    d = flow([node("in", "input", fields=[{"name": "q"}]),
              node("a", "llm", prompt="A {{in.q}}"), node("a2", "llm", prompt="A2 {{a}}"),
              node("b", "llm", prompt="B {{in.q}}"),
              node("out", "output", template="{{a2}} | {{b}}")],
             [("in", "a"), ("a", "a2"), ("in", "b"), ("a2", "out"), ("b", "out")])
    llm = Slow()
    task = asyncio.ensure_future(run_flow(parse(d), {"q": "x"}, CONFIG, llm=llm, tools=TOOLS))
    try:
        await asyncio.wait_for(both_started(), 5)
        assert sorted(started) == ["A x", "B x"]
    finally:
        gate.set()                            # let the run finish, also when the check above fails
    result = await asyncio.wait_for(task, 5)
    # the output ran once, after the longer branch (a -> a2) too
    assert result.text == "[chat-model] A2 [chat-model] A x | [chat-model] B x"


async def test_an_llm_node_can_name_its_own_model_and_does_not_inherit_the_runs_user_model_settings():
    d = news_flow()
    d["nodes"][2]["config"]["model"] = "qwen/qwen3.5-9b"
    llm = EchoModel()
    config = {"configurable": {**CONFIG["configurable"], "model": "user:abc", "no_fallback": True}}
    await run_flow(parse(d), {"topic": "t"}, config, llm=llm, tools=TOOLS)
    assert llm.calls[0]["model"] == "qwen/qwen3.5-9b"
    assert llm.calls[0]["no_fallback"] is None          # a setting of the run's own endpoint model, not this one


async def test_json_output_is_parsed_even_inside_prose_and_a_fence():
    llm = EchoModel(replies={"Extract": 'Sure:\n```json\n{"city": "Huế", "n": 2}\n```'})
    d = flow([node("in", "input", fields=[{"name": "t"}]),
              node("x", "llm", prompt="Extract {{in.t}}", output="json"),
              node("out", "output", template="{{x.json}}")],
             [("in", "x"), ("x", "out")])
    result = await run_flow(parse(d), {"t": "..."}, CONFIG, llm=llm, tools=TOOLS)
    assert json.loads(result.text) == {"city": "Huế", "n": 2}
    assert "JSON" in llm.calls[0]["system"]


async def test_a_model_that_does_not_answer_json_fails_its_node():
    llm = EchoModel(replies={"Extract": "I cannot do that"})
    d = flow([node("in", "input", fields=[{"name": "t"}]),
              node("x", "llm", prompt="Extract {{in.t}}", output="json"),
              node("out", "output", template="{{x.json}}")],
             [("in", "x"), ("x", "out")])
    with pytest.raises(FlowNodeError) as e:
        await run_flow(parse(d), {"t": "..."}, CONFIG, llm=llm, tools=TOOLS)
    assert e.value.node_id == "x" and "asked for JSON" in e.value.message


async def test_tool_arguments_are_typed_as_the_tool_declares_them():
    d = flow([node("in", "input", fields=[{"name": "text"}, {"name": "k", "type": "number"}]),
              node("t", "tool", tool="top_words", args={"text": "{{in.text}}", "top_k": "2", "lower": "true"}),
              node("out", "output", template="{{t}}")],
             [("in", "t"), ("t", "out")])
    result = await run_flow(parse(d), {"text": "B a C", "k": 3}, CONFIG, llm=EchoModel(), tools=TOOLS)
    assert flow_fakes.tool_calls == [{"tool": "top_words", "text": "B a C", "top_k": 2, "lower": True}]
    assert json.loads(result.text) == ["a", "b"]


async def test_a_failing_tool_stops_the_flow_and_names_the_node():
    d = flow([node("in", "input", fields=[{"name": "q"}]), node("t", "tool", tool="broken", args={"query": "{{in.q}}"}),
              node("out", "output", template="{{t}}")], [("in", "t"), ("t", "out")])
    with pytest.raises(FlowNodeError) as e:
        await run_flow(parse(d), {"q": "x"}, CONFIG, llm=EchoModel(), tools=TOOLS)
    assert e.value.node_id == "t" and "upstream is down" in e.value.message


@pytest.mark.parametrize("inputs, says", [
    ({}, "'topic' is required"),
    ({"topic": "x", "extra": 1}, "no input extra"),
])
async def test_inputs_are_checked_before_anything_runs(inputs, says):
    llm = EchoModel()
    with pytest.raises(FlowInputError, match=says):
        await run_flow(parse(news_flow()), inputs, CONFIG, llm=llm, tools=TOOLS)
    assert llm.calls == [] and flow_fakes.tool_calls == []


def test_inputs_are_typed():
    parsed = parse(flow([node("in", "input", fields=[{"name": "n", "type": "number"}, {"name": "b", "type": "boolean"},
                                                    {"name": "s", "required": False}]),
                         node("out", "output", template="{{in.n}}")], [("in", "out")]))
    assert runner.coerce_inputs(parsed.input_fields, {"n": "2.5", "b": "true"}) == {"n": 2.5, "b": True, "s": ""}
    with pytest.raises(FlowInputError, match="must be a number"):
        runner.coerce_inputs(parsed.input_fields, {"n": "two", "b": False})


async def test_a_streamed_run_reports_each_node_its_tokens_and_the_result():
    events = [e async for e in stream_flow(parse(news_flow()), {"topic": "t"}, CONFIG, llm=EchoModel(), tools=TOOLS)]
    steps = [(e["node"], e["phase"]) for e in events if e["type"] == "node"]
    assert steps == [("in", "start"), ("in", "end"), ("s", "start"), ("s", "end"),
                     ("sum", "start"), ("sum", "end"), ("out", "start"), ("out", "end")]
    s_end = next(e for e in events if e["type"] == "node" and e["node"] == "s" and e["phase"] == "end")
    assert s_end["output"] == {"result": "results for t news"} and s_end["node_type"] == "tool"
    tokens = [e for e in events if e["type"] == "token"]
    assert {e["node"] for e in tokens} == {"sum"}
    assert "".join(e["content"] for e in tokens) == "[chat-model] Summarise: results for t news"
    assert events[-1] == {"type": "result", "text": "Summary: [chat-model] Summarise: results for t news"}


async def test_a_streamed_run_that_fails_ends_with_the_error_and_its_node():
    d = flow([node("in", "input", fields=[{"name": "q"}]), node("t", "tool", tool="broken", args={"query": "{{in.q}}"}),
              node("out", "output", template="{{t}}")], [("in", "t"), ("t", "out")])
    events = [e async for e in stream_flow(parse(d), {"q": "x"}, CONFIG, llm=EchoModel(), tools=TOOLS)]
    assert events[-1]["type"] == "error" and events[-1]["node"] == "t"
    assert not any(e["type"] == "result" for e in events)


async def test_a_run_that_takes_too_long_is_stopped(monkeypatch):
    monkeypatch.setattr(runner.settings, "FLOW_RUN_TIMEOUT_SECONDS", 0.2)

    class Stuck(EchoModel):
        async def _astream(self, messages, stop=None, run_manager=None, **kw):
            await asyncio.sleep(5)
            yield  # pragma: no cover

    events = [e async for e in stream_flow(parse(news_flow()), {"topic": "t"}, CONFIG, llm=Stuck(), tools=TOOLS)]
    assert events[-1]["type"] == "error" and "was stopped" in events[-1]["message"]


async def test_a_flow_run_inside_a_chat_tool_reports_its_steps_under_that_tool():
    """What step 2 (running a flow from chat) builds on: the chat turn's astream_events sees the flow's node events
    and model tokens, with the tool's run id among their parent ids, and the run's configurable reaches the flow."""
    parsed = parse(news_flow())
    llm = EchoModel()

    @tool
    async def flow_news(topic: str) -> str:
        """Run the news flow."""
        return (await run_flow(parsed, {"topic": topic}, {}, llm=llm, tools=TOOLS)).text

    class Caller(BaseChatModel):
        @property
        def _llm_type(self):
            return "caller"

        def bind_tools(self, tools, **kw):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            if any(isinstance(m, ToolMessage) for m in messages):
                return ChatResult(generations=[ChatGeneration(message=AIMessage("done"))])
            return ChatResult(generations=[ChatGeneration(message=AIMessage("", tool_calls=[
                {"name": "flow_news", "id": "c1", "args": {"topic": "Đà Nẵng"}}]))])

    agent = create_agent(model=Caller(), tools=[flow_news])
    tool_run, steps, token_nodes, result = None, [], set(), None
    async for ev in agent.astream_events({"messages": [("user", "go")]},
                                         config={"configurable": {"user_id": "u9", "model": "chat-model"}},
                                         version="v2"):
        under = tool_run is not None and tool_run in ev.get("parent_ids", [])
        if ev["event"] == "on_tool_start" and ev["name"] == "flow_news":
            tool_run = ev["run_id"]
        elif ev["event"] == "on_custom_event" and ev["name"] == "flow_node":
            assert under
            steps.append((ev["data"]["node"], ev["data"]["phase"]))
        elif ev["event"] == "on_chat_model_stream" and under:
            token_nodes.add(ev["metadata"].get("langgraph_node"))
        elif ev["event"] == "on_tool_end" and ev["name"] == "flow_news":
            result = ev["data"]["output"].content
    assert steps[0] == ("in", "start") and steps[-1] == ("out", "end") and len(steps) == 8
    assert token_nodes == {"sum"}
    assert flow_fakes.tool_calls[0]["user_id"] == "u9"       # the chat run's configurable reached the flow's tools
    assert result == "Summary: [chat-model] Summarise: results for Đà Nẵng news"
