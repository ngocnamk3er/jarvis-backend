"""Deferred tools: schemas the model is not sent until it asks for them.

Every tool the model is offered costs its whole JSON schema on every call —
`generate_visualization_svg` alone is ~400 tokens, and MCP servers routinely
bring dozens of tools. A deferred tool is instead listed by name and one line
in the system prompt; the model calls `tool_search` when it wants one, the
name goes into the thread's `loaded_tools`, and from the next model call on
the full schema is offered like any other tool. Loaded tools stay loaded for
the rest of the conversation.

Which tools are deferred is decided in graph.py (the DEFERRED_TOOLS setting,
plus every MCP tool by default). Tools are filtered per model call, the way
ToolToggleMiddleware does it: the graph stays bound to the full list, so the
tool node can still run whatever the model calls — this middleware only
refuses a deferred tool that was never loaded, with a message saying how.
"""

import re
from typing import Any, Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from deepagents.middleware._utils import append_to_system_message
from langgraph.types import Command
from typing_extensions import NotRequired

import json

from app.agents.mcp_guard import USER_TOOL_PREFIX
from app.agents.token_estimate import EXTRA_KEY, tokens as _tokens

MAX_LOADED_PER_SEARCH = 5


class ToolSearchState(AgentState):
    loaded_tools: NotRequired[list[str]]


def tool_name(t: Any) -> str:
    if isinstance(t, dict):
        return (t.get("function") or {}).get("name") or t.get("name") or ""
    return getattr(t, "name", "")


def _summary(t: Any) -> str:
    desc = t.get("description") if isinstance(t, dict) else getattr(t, "description", "")
    first = (desc or "").strip().split("\n")[0].strip()
    return first[:160]


# Too common to say anything about what a tool does — left out so that "make a
# csv export" does not also match every description containing "a".
_STOPWORDS = frozenset("the and for with from into that this you your are can use used when then than "
                       "make get not any all has have its out via per".split())


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in _STOPWORDS}


class ToolSearchMiddleware(AgentMiddleware):
    state_schema = ToolSearchState

    def __init__(self, deferred: frozenset[str], extra_catalog: Callable[[], Awaitable[dict[str, str]]] | None = None):
        self.deferred = deferred
        # Deferred tools that differ per user (their own MCP servers): looked
        # up for the current run, never kept in the process-wide catalog below,
        # which every conversation shares.
        self._extra_catalog = extra_catalog
        # name -> one-line description of every deferred tool seen so far.
        # Static for the life of the process, so sharing it across
        # conversations is fine; it is only filled so tool_search can match
        # keywords without needing the tool objects at call time.
        self._catalog: dict[str, str] = {}
        self.tools = [self._make_tool()]

    def _make_tool(self):
        catalog = self._catalog

        extra_catalog = self._extra_catalog

        @tool
        async def tool_search(query: str, runtime: ToolRuntime) -> Command:
            """Load deferred tools so you can call them. The system prompt lists
            them under "Deferred tools" by name — their schemas are not
            available until you load them here.

            Args:
                query: Either `select:name1,name2` to load those tools by exact
                    name, or a few keywords describing what you need (matched
                    against tool names and descriptions; the best few load).
            """
            cat = {**catalog, **(await extra_catalog() if extra_catalog else {})}
            q = query.strip()
            if q.lower().startswith("select:"):
                wanted = [n.strip() for n in q[7:].split(",") if n.strip()]
                chosen = [n for n in wanted if n in cat]
                missing = [n for n in wanted if n not in cat]
            else:
                qw = _words(q)
                scored = sorted(
                    ((len(qw & _words(f"{n} {d}")), n) for n, d in cat.items()), reverse=True
                )
                chosen = [n for score, n in scored if score > 0][:MAX_LOADED_PER_SEARCH]
                missing = []

            already = list(runtime.state.get("loaded_tools") or [])
            loaded = already + [n for n in chosen if n not in already]
            if chosen:
                lines = [f"- `{n}`: {cat[n]}" for n in chosen]
                text = "Loaded — you can call these now:\n" + "\n".join(lines)
            else:
                text = "No deferred tool matched. Deferred tools: " + (", ".join(sorted(cat)) or "none") + "."
            if missing:
                text += f"\nNot found: {', '.join(missing)}."
            return Command(update={
                "loaded_tools": loaded,
                "messages": [ToolMessage(content=text, tool_call_id=runtime.tool_call_id)],
            })

        return tool_search

    def _is_deferred(self, name: str) -> bool:
        return name in self.deferred or name.startswith(USER_TOOL_PREFIX)

    def _apply(self, request: ModelRequest) -> ModelRequest:
        loaded = set(request.state.get("loaded_tools") or [])
        present = [t for t in request.tools if self._is_deferred(tool_name(t))]
        for t in present:
            if tool_name(t) in self.deferred:   # a user's own tools are not remembered across users
                self._catalog[tool_name(t)] = _summary(t)
        hidden = [t for t in present if tool_name(t) not in loaded]

        keep = [
            t for t in request.tools
            if tool_name(t) != "tool_search" or hidden  # nothing left to load: drop the loader too
        ]
        keep = [t for t in keep if t not in hidden]
        if not hidden:
            return request.override(tools=keep)

        listing = "\n".join(f"- `{tool_name(t)}`: {_summary(t)}" for t in hidden)
        text = (
            "## Deferred tools\n"
            "These tools exist but their schemas are not loaded, so you cannot call them yet. "
            "When one would help, call `tool_search` (`select:name` or keywords) first; it becomes "
            "callable on your next step.\n" + listing
        )
        detail = []
        for t in hidden:
            spec = t if isinstance(t, dict) else convert_to_openai_tool(t)
            detail.append({"name": tool_name(t), "tokens": _tokens(json.dumps(spec, default=str))})
        extra = dict(request.state.get(EXTRA_KEY) or {})
        extra["deferred"] = detail
        return request.override(
            tools=keep,
            system_message=append_to_system_message(request.system_message, text),
            state={**request.state, EXTRA_KEY: extra},
        )

    def _blocked(self, request: ToolCallRequest) -> ToolMessage | None:
        name = request.tool_call["name"]
        if self._is_deferred(name) and name not in set((request.state or {}).get("loaded_tools") or []):
            return ToolMessage(
                content=f"'{name}' is a deferred tool and is not loaded yet. Call tool_search with "
                        f"`select:{name}` first, then call it again.",
                tool_call_id=request.tool_call["id"],
                name=name,
                status="error",
            )
        return None

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> Any:
        return handler(self._apply(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> Any:
        return await handler(self._apply(request))

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]) -> Any:
        return self._blocked(request) or handler(request)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[Any]]
    ) -> Any:
        return self._blocked(request) or await handler(request)
