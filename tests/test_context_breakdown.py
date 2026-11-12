"""The context panel's breakdown: measured on the request the model really
gets, stored through the graph's own checkpoint, read back as parts that add
up to the gauge."""

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.middleware import ContextBreakdownMiddleware, ToolToggleMiddleware
from app.clients import conversation_client
from app.services import conversation_service


@tool
async def web_search(query: str) -> str:
    """Search the web for a query and return result snippets."""
    return "x"


@tool
async def calc(expr: str) -> str:
    """Evaluate an expression."""
    return "1"


class Plain(BaseChatModel):
    @property
    def _llm_type(self):
        return "plain"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        return ChatResult(generations=[ChatGeneration(message=AIMessage("ok"))])


def _graph():
    return create_agent(
        model=Plain(), tools=[web_search, calc], system_prompt="You are Jarvis. " * 20,
        checkpointer=InMemorySaver(),
        middleware=[ToolToggleMiddleware(), TodoListMiddleware(), ContextBreakdownMiddleware()],
    )


async def _breakdown(**configurable):
    g = _graph()
    cfg = {"configurable": {"thread_id": "t", **configurable}}
    await g.ainvoke({"messages": [("user", "hi")]}, cfg)
    return g, cfg, (await g.aget_state(cfg)).values["context_breakdown"]


async def test_measures_what_the_model_is_really_sent():
    _, _, b = await _breakdown()
    names = {d["name"] for d in b["tool_detail"]}
    assert {"web_search", "calc", "write_todos"} <= names      # middleware-added tool counted
    assert b["tools"] == sum(d["tokens"] for d in b["tool_detail"]) > 0
    # the todo middleware appends its own instructions after the base prompt
    assert b["system"] > len("You are Jarvis. " * 20) // 4


async def test_tools_turned_off_for_the_run_are_not_counted():
    _, _, on = await _breakdown()
    _, _, off = await _breakdown(disabled_tools=["web_search"])
    assert "web_search" not in {d["name"] for d in off["tool_detail"]}
    assert off["tools"] < on["tools"]


async def test_service_returns_parts_that_add_up(monkeypatch):
    g, cfg, b = await _breakdown()

    class Conv:
        user_id = "u"
        context_tokens = b["system"] + b["tools"] + 700

    async def get(_):
        return Conv()

    monkeypatch.setattr(conversation_client, "get_conversation", get)
    out = await conversation_service.get_context_breakdown(g, "t", "u")
    assert out["messages"] == 700
    assert out["system_prompt"] + out["tools"] + out["messages"] == out["total"]


async def test_conversation_without_a_breakdown_gets_the_total_only(monkeypatch):
    g = _graph()

    class Conv:
        user_id = "u"
        context_tokens = 1234

    async def get(_):
        return Conv()

    monkeypatch.setattr(conversation_client, "get_conversation", get)
    out = await conversation_service.get_context_breakdown(g, "never-ran", "u")
    assert out["total"] == 1234
    assert out["system_prompt"] is out["tools"] is out["messages"] is None
    assert out["tool_detail"] == [] and out["deferred"]["tokens"] == 0
