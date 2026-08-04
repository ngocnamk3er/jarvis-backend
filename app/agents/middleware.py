"""Custom middleware not covered by langchain's built-ins."""

from typing import Any

from langchain_core.callbacks import adispatch_custom_event, dispatch_custom_event
from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langchain.agents.middleware.types import AgentMiddleware, hook_config
from langchain.agents.middleware.tool_call_limit import ToolCallLimitState

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
      the model keeps going (same spirit as `ToolCallLimitMiddleware`'s
      `exit_behavior="continue"`) — it can still finish the turn with other
      tools or with the calls that did go through.
    - Any call past hard_limit: also blocked, but additionally ends the run
      right there (`jump_to="end"`, same spirit as `exit_behavior="end"`) —
      a burst that blows past the hard ceiling isn't left for the model to
      keep negotiating around.

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

    def _evaluate(
        self, state: ToolCallLimitState
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Shared logic for after_model/aafter_model.

        Returns `(state_update, limit_events)` — `limit_events` is a list of
        `tool_call_limit` payloads the caller still needs to dispatch (kept
        separate since dispatching is the only part that differs sync vs async).
        """
        messages = state.get("messages", [])
        last_ai_message = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage)), None
        )
        if not last_ai_message or not last_ai_message.tool_calls:
            return None, []

        run_counts = state.get("run_tool_call_count", {}).copy()
        count = run_counts.get(self.tool_name, 0)

        soft_blocked: list[ToolCall] = []
        hard_blocked: list[ToolCall] = []
        for tool_call in last_ai_message.tool_calls:
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

        if hard_blocked:
            # Jumping to "end" leaves any OTHER tool's pending calls in this same
            # AIMessage without a matching ToolMessage, which breaks the next model
            # call (every tool_call must get a response). Only safe to jump straight
            # to "end" when this tool is the only one called this step.
            other_tools = [
                tc for tc in last_ai_message.tool_calls if tc["name"] != self.tool_name
            ]
            if other_tools:
                tool_names = ", ".join({tc["name"] for tc in other_tools})
                msg = (
                    f"Cannot end execution with other tool calls pending. "
                    f"Found calls to: {tool_names}. Hard limit needs this tool to be "
                    "the only one called in that step."
                )
                raise NotImplementedError(msg)

        limit_events: list[dict[str, Any]] = []
        artificial_messages: list[ToolMessage | AIMessage] = [
            ToolMessage(
                content=(
                    f"Soft limit reached for '{self.tool_name}' ({self.soft_limit} "
                    "calls/run). This call was blocked — you can still use other "
                    "tools or wrap up with what you already have."
                ),
                tool_call_id=tool_call["id"],
                name=tool_call.get("name"),
                status="error",
            )
            for tool_call in soft_blocked
        ]
        if soft_blocked:
            limit_events.append({
                "tool_name": self.tool_name,
                "level": "soft",
                "count": count,
                "limit": self.soft_limit,
                "blocked": len(soft_blocked),
            })
        artificial_messages += [
            ToolMessage(
                content=(
                    f"Hard limit reached for '{self.tool_name}' "
                    f"({self.hard_limit} calls/run). Stopping now."
                ),
                tool_call_id=tool_call["id"],
                name=tool_call.get("name"),
                status="error",
            )
            for tool_call in hard_blocked
        ]
        if hard_blocked:
            limit_events.append({
                "tool_name": self.tool_name,
                "level": "hard",
                "count": count,
                "limit": self.hard_limit,
                "blocked": len(hard_blocked),
            })

        if hard_blocked:
            artificial_messages.append(
                AIMessage(
                    content=(
                        f"'{self.tool_name}' tool call limit reached: hard limit "
                        f"exceeded ({count}/{self.hard_limit} calls this run). "
                        "Stopping execution."
                    )
                )
            )
            return {
                "run_tool_call_count": run_counts,
                "jump_to": "end",
                "messages": artificial_messages,
            }, limit_events

        return {
            "run_tool_call_count": run_counts,
            "messages": artificial_messages,
        }, limit_events

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
