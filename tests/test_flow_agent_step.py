"""An LLM step with tools: the model calls them as it decides, within a budget of calls, and always ends with an
answer, even when the budget runs out."""

import pytest
from app.flows.catalog import tool_specs
from app.flows.compiler import FlowNodeError
from app.flows.runner import run_flow, stream_flow
from app.flows.schema import validate_flow
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.config import get_config

from tests import flow_fakes
from tests.flow_fakes import TOOLS, flow, node

CONFIG = {"configurable": {"user_id": "u1", "model": "chat-model", "sandbox": False}}


class ToolCaller(BaseChatModel):
    """Calls `calls` (a list of [name, args] per turn) while it has turns left, then answers with every tool result it
    saw. `obeys`: whether it stops calling tools when told it may not (tool_choice="none")."""

    calls: list = []
    obeys: bool = True
    seen: list = []

    @property
    def _llm_type(self):
        return "tool-caller"

    def bind_tools(self, tools, tool_choice=None, **kw):
        return self.bind(tool_choice=tool_choice)

    def _generate(self, messages, stop=None, run_manager=None, tool_choice=None, **kw):
        self.seen.append({"tool_choice": tool_choice, "model": get_config().get("configurable", {}).get("model")})
        turn = sum(isinstance(m, AIMessage) for m in messages)
        results = [m.content for m in messages if isinstance(m, ToolMessage)]
        if (tool_choice == "none" and self.obeys) or turn >= len(self.calls):
            return ChatResult(generations=[ChatGeneration(message=AIMessage("Found: " + " | ".join(results)))])
        tcs = [{"name": n, "args": a, "id": f"c{turn}-{i}"} for i, (n, a) in enumerate(self.calls[turn])]
        return ChatResult(generations=[ChatGeneration(message=AIMessage("", tool_calls=tcs))])


def agent_flow(tools=("web_search",), max_calls=5, prompt="Find news about {{in.topic}}", **llm):
    return flow([node("in", "input", fields=[{"name": "topic"}]),
                 node("sum", "llm", prompt=prompt, tools=list(tools), max_tool_calls=max_calls, **llm),
                 node("out", "output", template="{{sum}}")],
                [("in", "sum"), ("sum", "out")])


def parse(d):
    parsed, problems = validate_flow(d, tool_specs(TOOLS))
    assert problems == [], problems
    return parsed


@pytest.fixture(autouse=True)
def clean():
    flow_fakes.tool_calls.clear()


async def test_an_llm_step_calls_its_tool_and_answers_with_what_it_found():
    model = ToolCaller(calls=[[["web_search", {"query": "Huế news"}]]])
    result = await run_flow(parse(agent_flow()), {"topic": "Huế"}, CONFIG, llm=model, tools=TOOLS)
    assert result.text == "Found: results for Huế news"
    # run as the user, and labelled for the chat's badge though the model did not give a label
    assert flow_fakes.tool_calls == [{"tool": "web_search", "query": "Huế news", "label": "Step sum · web_search",
                                      "user_id": "u1", "sandbox": False}]
    assert result.outputs["sum"]["tool_calls"] == [{"name": "web_search", "args": {"query": "Huế news"}}]


async def test_the_model_calls_up_to_its_limit_then_gets_a_last_turn_to_answer():
    two_a_turn = [["web_search", {"query": "a"}], ["web_search", {"query": "b"}]]
    model = ToolCaller(calls=[two_a_turn] * 5)
    result = await run_flow(parse(agent_flow(max_calls=3)), {"topic": "x"}, CONFIG, llm=model, tools=TOOLS)
    assert len(flow_fakes.tool_calls) == 3                           # 2, then 1 of the next 2; the 4th is refused
    assert [s["tool_choice"] for s in model.seen] == [None, None, "none"]
    assert result.text.startswith("Found: results for a | results for b | results for a | Not run:")


async def test_a_model_that_keeps_calling_tools_past_its_limit_fails_the_step():
    model = ToolCaller(calls=[[["web_search", {"query": "a"}]]] * 5, obeys=False)
    with pytest.raises(FlowNodeError) as e:
        await run_flow(parse(agent_flow(max_calls=2)), {"topic": "x"}, CONFIG, llm=model, tools=TOOLS)
    assert e.value.node_id == "sum" and "after its limit of 2 calls" in e.value.message


async def test_a_failing_tool_goes_back_to_the_model_which_can_still_answer():
    model = ToolCaller(calls=[[["broken", {"query": "x"}]]])
    result = await run_flow(parse(agent_flow(tools=("broken",))), {"topic": "x"}, CONFIG, llm=model, tools=TOOLS)
    assert result.text == "Found: RuntimeError: upstream is down"


async def test_a_tool_the_step_was_not_given_is_not_run():
    model = ToolCaller(calls=[[["top_words", {"text": "a b", "top_k": 1}]]])
    result = await run_flow(parse(agent_flow()), {"topic": "x"}, CONFIG, llm=model, tools=TOOLS)
    assert flow_fakes.tool_calls == [] and "no tool 'top_words'" in result.text


async def test_every_turn_of_the_step_uses_the_steps_own_model():
    model = ToolCaller(calls=[[["web_search", {"query": "a"}]]])
    await run_flow(parse(agent_flow(model="qwen/qwen3.5-9b")), {"topic": "x"}, CONFIG, llm=model, tools=TOOLS)
    assert [s["model"] for s in model.seen] == ["qwen/qwen3.5-9b", "qwen/qwen3.5-9b"]


async def test_a_test_run_reports_the_tools_the_step_called():
    model = ToolCaller(calls=[[["web_search", {"query": "Huế news"}]]])
    events = [e async for e in stream_flow(parse(agent_flow()), {"topic": "Huế"}, CONFIG, llm=model, tools=TOOLS)]
    assert [e for e in events if e["type"] == "tool"] == [
        {"type": "tool", "node": "sum", "name": "web_search", "input": {"query": "Huế news"}}]
    assert events[-1] == {"type": "result", "text": "Found: results for Huế news"}


def test_a_step_can_only_be_given_tools_a_flow_can_use():
    _, problems = validate_flow(agent_flow(tools=("web_search", "rm_rf")), tool_specs(TOOLS))
    assert [(p.node, p.message) for p in problems] == [("sum", "'rm_rf' is not a tool a flow can use.")]
    _, problems = validate_flow(agent_flow(tools=("web_search", "web_search")), tool_specs(TOOLS))
    assert [p.message for p in problems] == ["A tool is listed twice."]
    _, problems = validate_flow(agent_flow(max_calls=50), tool_specs(TOOLS))
    assert problems and problems[0].node == "sum"


async def test_a_label_the_model_gives_names_the_badge_but_is_not_reported_as_an_argument():
    model = ToolCaller(calls=[[["web_search", {"query": "q", "label": "Searching the news"}]]])
    result = await run_flow(parse(agent_flow()), {"topic": "x"}, CONFIG, llm=model, tools=TOOLS)
    assert flow_fakes.tool_calls[0]["label"] == "Searching the news"
    assert result.outputs["sum"]["tool_calls"] == [{"name": "web_search", "args": {"query": "q"}}]
