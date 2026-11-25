"""Custom middleware not covered by langchain's built-ins."""

import json
import logging
from typing import Any, Awaitable, Callable

from langchain.agents.middleware.tool_call_limit import ToolCallLimitState
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    hook_config,
)
from langchain_core.callbacks import adispatch_custom_event, dispatch_custom_event
from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.types import Command
from typing_extensions import NotRequired

from app.agents.mcp_guard import USER_TOOL_PREFIX
from app.agents.token_estimate import EXTRA_KEY, tokens as _tokens

logger = logging.getLogger(__name__)


class ToolToggleMiddleware(AgentMiddleware):
    """Per-run tool filtering. Reads `disabled_tools` (a list of tool names)
    from LangGraph's configurable — set by chat_service._make_config from the
    request's `web_search` flag.

    Two layers, both needed:
    - `wrap_model_call` drops the tools from what the model is *offered* each
      turn (the graph stays bound to the full list, so no per-permutation
      rebuild).
    - `wrap_tool_call` rejects a call to a disabled tool with an error
      ToolMessage — a backstop for models that emit a tool call anyway
      despite it not being in the offered schema (deepseek-flash does this
      when the prompt literally says "search the web"). Without this the
      tool node happily executes the hallucinated call, since it's still
      registered.

    Also on the research subagent's middleware list (subagents.py): LangGraph
    merges the root run's configurable into a subagent's own config, so "web
    off" means web off everywhere, not just the main agent.
    """

    @staticmethod
    def _disabled() -> set[str]:
        try:
            from langgraph.config import get_config

            return set(get_config().get("configurable", {}).get("disabled_tools") or [])
        except RuntimeError:
            # No ambient config (e.g. a graph.aget_state() read path) — nothing to filter.
            return set()

    def _apply(self, request: ModelRequest) -> ModelRequest:
        disabled = self._disabled()
        if not disabled or not request.tools:
            return request
        kept = [t for t in request.tools if getattr(t, "name", None) not in disabled]
        return request.override(tools=kept)

    def _reject(self, request: ToolCallRequest) -> ToolMessage | None:
        name = request.tool_call["name"]
        if name in self._disabled():
            return ToolMessage(
                content=f"The '{name}' tool is turned off for this conversation.",
                tool_call_id=request.tool_call["id"],
                name=name,
                status="error",
            )
        return None

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        return handler(self._apply(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        return await handler(self._apply(request))

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]) -> Any:
        return self._reject(request) or handler(request)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[Any]]
    ) -> Any:
        return self._reject(request) or await handler(request)


class ContextTokensState(AgentState):
    context_tokens: NotRequired[int]


class ContextTokensMiddleware(AgentMiddleware):
    """Contributes the `context_tokens` state key so chat_service.py's
    _run_graph can persist the current-context-size gauge into the LangGraph
    checkpoint itself, in addition to Conversation.context_tokens, which now
    lives in jarvis-conversation-service (see
    app/clients/conversation_client.py's set_context_tokens) — verified live
    that a plain scalar key contributed this way survives
    aupdate_state/aget_state with last-writer-wins semantics, same as the
    remote column.

    No hooks — purely a state_schema extension. Kept in both build_graph()
    variants (unlike SummarizationToolMiddleware) since every turn writes
    this, not just compact() runs.
    """

    state_schema = ContextTokensState

class ContextBreakdownState(AgentState):
    context_breakdown: NotRequired[dict]


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b if isinstance(b, str) else str(b.get("text", "")) for b in content)
    return str(content)


class ContextBreakdownMiddleware(AgentMiddleware):
    """Records how much of the model's context is fixed overhead — the system
    prompt, the skills listing and the tool schemas — for the chat's context
    panel, and how much a deferred tool would have cost had it been sent.

    It has to be the LAST middleware in the list: the first one is outermost,
    and each wrap_model_call that runs before this one may still add to the
    system prompt (todo list, sub-agents, skills) or drop tools
    (ToolToggleMiddleware, ToolSearchMiddleware), so only the innermost one
    sees the request exactly as the model gets it. What the outer ones added or
    held back is handed over in `request.state[EXTRA_KEY]`.

    The result goes into the `context_breakdown` state key through the model
    call's own Command, i.e. inside the graph's normal checkpoint write — never
    through graph.aupdate_state while a run is going, which forks the thread's
    checkpoints and loses writes (see _run_graph in chat_service.py).

    Messages are not counted here: the provider's total_tokens is the real
    figure, so messages = total - everything measured here, and the parts
    always add up to the gauge.
    """

    state_schema = ContextBreakdownState

    def __init__(self, mcp_tools: frozenset[str] = frozenset()):
        self.mcp_tools = mcp_tools

    def _kind(self, name: str) -> str:
        return "mcp" if name in self.mcp_tools or name.startswith(USER_TOOL_PREFIX) else "builtin"

    def _measure(self, request: ModelRequest) -> dict | None:
        try:
            extra = request.state.get(EXTRA_KEY) or {}
            skills = extra.get("skills", 0)
            system = max(0, (_tokens(_text(request.system_message.content)) if request.system_message else 0) - skills)
            detail = []
            for t in request.tools or []:
                spec = t if isinstance(t, dict) else convert_to_openai_tool(t)
                name = (spec.get("function") or {}).get("name") or spec.get("name") or "tool"
                detail.append({
                    "name": name,
                    "tokens": _tokens(json.dumps(spec, default=str)),
                    "kind": self._kind(name),
                })
            detail.sort(key=lambda d: -d["tokens"])
            held = extra.get("deferred") or []
            return {
                "system": system,
                "skills": skills,
                "tools": sum(d["tokens"] for d in detail if d["kind"] == "builtin"),
                "mcp_tools": sum(d["tokens"] for d in detail if d["kind"] == "mcp"),
                "tool_detail": detail,
                "deferred": {
                    "tokens": sum(d["tokens"] for d in held),
                    "detail": [{**d, "kind": self._kind(d["name"])} for d in held],
                },
            }
        except Exception:  # a display-only figure must never break a model call
            logger.warning("could not measure context breakdown", exc_info=True)
            return None

    @staticmethod
    def _with(resp: Any, measured: dict | None) -> Any:
        if measured is None or not isinstance(resp, ModelResponse):
            return resp
        return ExtendedModelResponse(model_response=resp, command=Command(update={"context_breakdown": measured}))

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> Any:
        return self._with(handler(request), self._measure(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> Any:
        return self._with(await handler(request), self._measure(request))


# Custom event name dispatched on a soft/hard breach — picked up by
# chat_service.py's _run_graph (as an `on_custom_event` from astream_events)
# and translated into a `tool_limit` SSE event so the frontend gets an explicit
# signal instead of having to infer it from the model's own narration of the
# blocked-call ToolMessage it received. Fires from inside a subagent's own run
# too — astream_events carries it up with parent_ids pointing at the `task`
# run it happened inside, so chat_service.py's existing task_run_id
# resolution nests it under the right subagent badge automatically.
TOOL_CALL_LIMIT_EVENT = "tool_call_limit"


class SoftHardToolCallLimitMiddleware(AgentMiddleware):
    """Two-tier per-run limiter for a single tool: call count resets every run
    (nothing persists thread-wide, unlike `ToolCallLimitMiddleware`'s
    `thread_limit`) — only `soft_limit` and `hard_limit` within the current run.

    - Calls 1..soft_limit: allowed.
    - Calls soft_limit+1..hard_limit: blocked with an error `ToolMessage`, but
      the model keeps going — it can still use other tools or wrap up.
    - A call past hard_limit: blocked, and the model gets one last turn in which
      it may not call any tool (`tool_choice="none"`), so the run ends with an
      answer written from what it already has.

    Until 2026-11-26 a call past hard_limit ended the run right there instead.
    Models call tools in parallel batches, and one batch can cross both limits
    in a single step: two background research tasks went from 10 to 14 searches
    (limits 10/12) and from 20 to 25 (limits 20/24). They ended in that step,
    before the model ever read the soft-limit message, with the stop notice
    where the report should have been — everything they had found was lost.
    Ending there also could not work when the same step called another tool,
    which would have been left without a response.

    If the model calls a tool anyway in that last turn (a provider may ignore
    tool_choice), every call is refused and the run ends there, with
    `stop_notice()` as its last message.

    Either breach also dispatches a `tool_call_limit` custom event (see
    `TOOL_CALL_LIMIT_EVENT`) so the frontend can show it directly, rather than
    relying on the model to narrate the blocked-call `ToolMessage` in its own
    words.
    """

    state_schema = ToolCallLimitState

    def __init__(self, *, tool_name: str, soft_limit: int, hard_limit: int) -> None:
        super().__init__()
        if soft_limit > hard_limit:
            msg = f"soft_limit ({soft_limit}) cannot exceed hard_limit ({hard_limit})"
            raise ValueError(msg)
        self.tool_name = tool_name
        self.soft_limit = soft_limit
        self.hard_limit = hard_limit

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}[{self.tool_name}]"

    def _count(self, state: Any) -> int:
        return (state.get("run_tool_call_count") or {}).get(self.tool_name, 0)

    def _wrapping_up(self, state: Any) -> bool:
        """Past the hard limit: the model's next turn is its last, and it may not call tools."""
        return self._count(state) > self.hard_limit

    def stop_notice(self, count: int) -> str:
        # async_research.py's _LIMIT_STOP matches this text to tell a run that ended here from one with a report.
        return (
            f"'{self.tool_name}' tool call limit reached: hard limit exceeded "
            f"({count}/{self.hard_limit} calls this run). Stopping execution."
        )

    def _refused(self, tool_call: ToolCall, content: str) -> ToolMessage:
        return ToolMessage(content=content, tool_call_id=tool_call["id"], name=tool_call.get("name"), status="error")

    def _evaluate(
        self, state: ToolCallLimitState
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Shared logic for after_model/aafter_model.

        Returns `(state_update, limit_events)` — `limit_events` is a list of
        `tool_call_limit` payloads the caller still needs to dispatch (kept
        separate since dispatching is the only part that differs sync vs async).
        """
        messages = state.get("messages", [])
        at = next((i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], AIMessage)), None)
        if at is None or not messages[at].tool_calls:
            return None, []
        last_ai_message = messages[at]
        # Calls another middleware has already answered (blocked) in this step are not counted or answered again.
        answered = {m.tool_call_id for m in messages[at + 1:] if isinstance(m, ToolMessage)}
        calls = [tc for tc in last_ai_message.tool_calls if tc["id"] not in answered]

        run_counts = dict(state.get("run_tool_call_count") or {})
        count = run_counts.get(self.tool_name, 0)

        if count > self.hard_limit:
            # This was the last turn, offered no tools, and the model called some anyway. Refuse them all — any tool,
            # not just this one, since none may run now — and end the run.
            if not calls:
                return None, []
            return {
                "messages": [
                    *(self._refused(tc, "No more tool calls are possible in this run.") for tc in calls),
                    AIMessage(content=self.stop_notice(count)),
                ],
                "jump_to": "end",
            }, []

        soft_blocked: list[ToolCall] = []
        hard_blocked: list[ToolCall] = []
        for tool_call in calls:
            if tool_call["name"] != self.tool_name:
                continue
            count += 1
            if count > self.hard_limit:
                hard_blocked.append(tool_call)
            elif count > self.soft_limit:
                soft_blocked.append(tool_call)

        run_counts[self.tool_name] = count
        if not soft_blocked and not hard_blocked:
            return {"run_tool_call_count": run_counts}, []

        limit_events: list[dict[str, Any]] = []
        if soft_blocked:
            limit_events.append({
                "tool_name": self.tool_name,
                "level": "soft",
                "count": count,
                "limit": self.soft_limit,
                "blocked": len(soft_blocked),
            })
        if hard_blocked:
            limit_events.append({
                "tool_name": self.tool_name,
                "level": "hard",
                "count": count,
                "limit": self.hard_limit,
                "blocked": len(hard_blocked),
            })
            # The whole step gets the wrap-up message: a soft-limit one saying "you can still use other tools" would be
            # wrong next to it, since the next turn may not call any.
            content = (
                f"Limit reached for '{self.tool_name}' ({self.hard_limit} calls/run). This call was not run, and no "
                "more tool calls are possible. Write your final answer now from what you already have, and say "
                "briefly what you could not cover."
            )
        else:
            content = (
                f"Soft limit reached for '{self.tool_name}' ({self.soft_limit} "
                "calls/run). This call was blocked — you can still use other "
                "tools or wrap up with what you already have."
            )
        return {
            "run_tool_call_count": run_counts,
            "messages": [self._refused(tc, content) for tc in soft_blocked + hard_blocked],
        }, limit_events

    def _last_turn(self, request: ModelRequest) -> ModelRequest:
        # tool_choice rather than dropping the tools: the history holds tool calls, which some providers only accept
        # alongside the tool definitions, and unchanged tools keep the provider's prompt cache valid.
        if request.tools and self._wrapping_up(request.state):
            return request.override(tool_choice="none")
        return request

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        return handler(self._last_turn(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        return await handler(self._last_turn(request))

    @hook_config(can_jump_to=["end"])
    def after_model(self, state: ToolCallLimitState, runtime: Any) -> dict[str, Any] | None:
        result, limit_events = self._evaluate(state)
        for payload in limit_events:
            dispatch_custom_event(TOOL_CALL_LIMIT_EVENT, payload)
        return result

    @hook_config(can_jump_to=["end"])
    async def aafter_model(
        self, state: ToolCallLimitState, runtime: Any
    ) -> dict[str, Any] | None:
        result, limit_events = self._evaluate(state)
        for payload in limit_events:
            await adispatch_custom_event(TOOL_CALL_LIMIT_EVENT, payload)
        return result
