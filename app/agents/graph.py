from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import Annotated
from typing_extensions import TypedDict

from app.agents.llm import build_llm
from app.agents.prompt import SYSTEM_PROMPT
from app.agents.tools import tools


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(checkpointer=None):
    system = SystemMessage(content=SYSTEM_PROMPT)

    def llm_node(state: AgentState, config: RunnableConfig):
        cfg = config.get("configurable", {})
        effort = cfg.get("thinking_effort", "high")
        model = cfg.get("model", None)
        llm_with_tools = build_llm(model).bind_tools(tools)
        response = llm_with_tools.invoke(
            [system] + state["messages"],
            extra_body={"reasoning": {"effort": effort, "exclude": False}},
        )
        return {"messages": [response]}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")

    return graph.compile(checkpointer=checkpointer)
