import json
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.db import repository
from app.db.connection import get_pool

# ---------------------------------------------------------------------------
# Thinking parser — splits <think>…</think> out of the content stream
# ---------------------------------------------------------------------------


class ThinkingParser:
    """State machine that routes streaming tokens to thinking_token vs token.

    None of the currently configured models (schemas/chat.py AVAILABLE_MODELS)
    actually need this: verified live that all of them report reasoning via
    OpenRouter's dedicated `reasoning` delta field (rescued into
    additional_kwargs by ThinkingChatOpenAI, handled above at the
    `additional_kwargs.get("reasoning")` check) — chunk.content never contains
    <think>/</think> for any of them, so this just passes content through
    unchanged. Kept for a future model that inlines <think>...</think> in
    content instead of using a dedicated field. If that model only emits the
    closing tag with no opening one, use ImplicitThinkingParser instead."""

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self):
        self.in_thinking = False
        self._buf = ""

    def feed(self, text: str) -> list[dict]:
        self._buf += text
        events: list[dict] = []
        tag = self._CLOSE if self.in_thinking else self._OPEN

        while True:
            idx = self._buf.find(tag)
            if idx != -1:
                content = self._buf[:idx]
                if content:
                    events.append(
                        {
                            "type": "thinking_token" if self.in_thinking else "token",
                            "content": content,
                        }
                    )
                self._buf = self._buf[idx + len(tag) :]
                self.in_thinking = not self.in_thinking
                tag = self._CLOSE if self.in_thinking else self._OPEN
            else:
                # Hold back any bytes that could be the start of the tag
                for i in range(1, len(tag)):
                    if self._buf.endswith(tag[:i]):
                        safe = self._buf[:-i]
                        if safe:
                            events.append(
                                {
                                    "type": (
                                        "thinking_token"
                                        if self.in_thinking
                                        else "token"
                                    ),
                                    "content": safe,
                                }
                            )
                        self._buf = self._buf[-i:]
                        return events
                if self._buf:
                    events.append(
                        {
                            "type": "thinking_token" if self.in_thinking else "token",
                            "content": self._buf,
                        }
                    )
                    self._buf = ""
                return events


class ImplicitThinkingParser:
    """State machine for models that start in thinking mode implicitly (no
    opening tag) and signal the end of thinking with a single </think> —
    e.g. some DeepSeek-R1 deployments. Unlike ThinkingParser, there's no
    <think> to wait for: every token is thinking_token until </think> is
    seen once, then everything after is the final answer. Not wired into
    _run_graph() by default — swap in for ThinkingParser if a configured
    model streams reasoning this way.

    reasoning_enabled must reflect whether reasoning was actually requested
    for this turn (e.g. thinking_effort != "none"). If reasoning is off, the
    model streams straight to its answer with no </think> at all — starting
    in_thinking=True in that case would misclassify the entire response as
    thinking_token forever, since the closing tag never arrives.
    """

    _CLOSE = "</think>"

    def __init__(self, reasoning_enabled: bool = True):
        self.in_thinking = reasoning_enabled
        self._buf = ""

    def feed(self, text: str) -> list[dict]:
        self._buf += text
        events: list[dict] = []

        if not self.in_thinking:
            if self._buf:
                events.append({"type": "token", "content": self._buf})
                self._buf = ""
            return events

        idx = self._buf.find(self._CLOSE)
        if idx != -1:
            content = self._buf[:idx]
            if content:
                events.append({"type": "thinking_token", "content": content})
            rest = self._buf[idx + len(self._CLOSE):]
            self.in_thinking = False
            self._buf = ""
            if rest:
                events.append({"type": "token", "content": rest})
            return events

        # Hold back any bytes that could be the start of </think>
        for i in range(1, len(self._CLOSE)):
            if self._buf.endswith(self._CLOSE[:i]):
                safe = self._buf[:-i]
                if safe:
                    events.append({"type": "thinking_token", "content": safe})
                self._buf = self._buf[-i:]
                return events

        if self._buf:
            events.append({"type": "thinking_token", "content": self._buf})
            self._buf = ""
        return events


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


VIZ_TOOLS = {"generate_visualization_svg"}
HIDDEN_TOOLS = {"write_todos"}


class ToolStartEventHandler:
    def handle(self, event: dict, task_run_id: str | None = None) -> list[dict]:
        if event["name"] in VIZ_TOOLS or event["name"] in HIDDEN_TOOLS:
            return []
        raw_input = dict(event["data"].get("input") or {})
        label = raw_input.pop("label", None)
        result = {"type": "tool_start", "name": event["name"], "label": label, "input": raw_input or None, "run_id": event.get("run_id", "")}
        if task_run_id:
            result["task_run_id"] = task_run_id
        return [result]


class ToolEndEventHandler:
    def handle(self, event: dict, task_run_id: str | None = None) -> list[dict]:
        if event["name"] in HIDDEN_TOOLS:
            return []
        raw = event["data"].get("output")
        if isinstance(raw, Command):
            # Tools that return a Command (e.g. the `task` subagent tool) wrap
            # their result in update={"messages": [ToolMessage(...)]} rather
            # than returning content directly.
            msgs = (raw.update or {}).get("messages") or []
            output = msgs[0].content if msgs and hasattr(msgs[0], "content") else str(raw)
        else:
            output = raw.content if hasattr(raw, "content") else str(raw)
        try:
            data = json.loads(output)
            if "__viz__" in data:
                return [{"type": "viz", "format": data["__viz__"], "code": data["code"], "title": data.get("title", "")}]
        except Exception:
            pass
        # Viz tools suppress tool_start/tool_chunk, so FE has no badge yet.
        # Emit a synthetic tool_start first so FE can show the error output.
        events: list[dict] = []
        if event["name"] in VIZ_TOOLS:
            raw_input = dict(event["data"].get("input") or {})
            label = raw_input.pop("label", None)
            start: dict = {"type": "tool_start", "name": event["name"], "label": label, "input": raw_input or None, "run_id": event.get("run_id", "")}
            if task_run_id:
                start["task_run_id"] = task_run_id
            events.append(start)
        end: dict = {"type": "tool_end", "name": event["name"], "output": output, "run_id": event.get("run_id", "")}
        if task_run_id:
            end["task_run_id"] = task_run_id
        events.append(end)
        return events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(thread_id: str, thinking_effort: str = "high", model: str | None = None) -> dict:
    return {
        "configurable": {"thread_id": thread_id, "thinking_effort": thinking_effort, "model": model},
        "recursion_limit": 200,
    }


def _extract_hitl_events(state) -> list[str]:
    """Return serialised hitl_request SSE lines for any pending HITL interrupts."""
    events = []
    for interrupt in getattr(state, "interrupts", ()):
        value = interrupt.value
        if isinstance(value, dict) and "action_requests" in value:
            events.append(json.dumps({
                "type": "hitl_request",
                "actions": value["action_requests"],
                "review_configs": value.get("review_configs", []),
            }))
    return events


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ChatService:
    def __init__(self) -> None:
        self._tool_start = ToolStartEventHandler()
        self._tool_end = ToolEndEventHandler()

    @staticmethod
    def _task_tool_call_id(event: dict) -> str | None:
        """Extract the `task` call's own tool_call_id from its on_tool_end event.

        This is the stable id (matches ToolMessage.tool_call_id in the persisted
        state) used to key a saved subagent trace — unlike run_id, which is a
        fresh UUID every time the graph replays/reloads.
        """
        raw = event["data"].get("output")
        if isinstance(raw, Command):
            msgs = (raw.update or {}).get("messages") or []
            if msgs and hasattr(msgs[0], "tool_call_id"):
                return msgs[0].tool_call_id
        return None

    @staticmethod
    def _usage_event(chunk) -> dict | None:
        # OpenRouter reports usage on a final, contentless chunk per LLM call.
        # A single agent turn may call the model multiple times (tool round-trips,
        # including hidden subagent calls), so the frontend collects one usage
        # entry per call rather than summing — each call's input_tokens already
        # includes all prior calls' context, so summing would double/triple-count
        # the repeated prefix.
        usage = chunk.usage_metadata
        if not usage:
            return None
        return {
            "type": "usage",
            "input_tokens": usage.get("input_tokens", 0) or 0,
            "output_tokens": usage.get("output_tokens", 0) or 0,
            "total_tokens": usage.get("total_tokens", 0) or 0,
        }

    def _handle_token(self, event: dict, parser: ThinkingParser, viz_indexes: set[int]) -> list[dict]:
        chunk = event["data"]["chunk"]
        events: list[dict] = []

        # Some OpenRouter models return reasoning in a dedicated field
        reasoning = chunk.additional_kwargs.get(
            "reasoning"
        ) or chunk.additional_kwargs.get("reasoning_content", "")
        if reasoning:
            events.append({"type": "thinking_token", "content": reasoning})

        # Regular content — route through <think> tag parser
        if chunk.content:
            events.extend(parser.feed(chunk.content))

        usage_event = self._usage_event(chunk)
        if usage_event:
            events.append(usage_event)

        # Tool call chunks — suppress for viz tools (they render as viz blocks, not badges)
        for tc in getattr(chunk, "tool_call_chunks", None) or []:
            name = tc.get("name") or ""
            index = tc.get("index", 0)
            args_delta = tc.get("args", "") or ""
            if name and name in VIZ_TOOLS:
                viz_indexes.add(index)
            if name and name in HIDDEN_TOOLS:
                viz_indexes.add(index)
            if index in viz_indexes:
                continue
            if name or args_delta:
                events.append(
                    {
                        "type": "tool_chunk",
                        "index": index,
                        "name": name,
                        "args_delta": args_delta,
                    }
                )

        return events

    async def _run_graph(self, graph_input, config: dict, graph):
        """Yield SSE lines by streaming graph events, then emit any HITL interrupt."""
        parser = ThinkingParser()
        viz_indexes: set[int] = set()
        hitl_lines: list[str] = []
        thread_id = config["configurable"]["thread_id"]
        # Maps every run_id inside a `task` subagent's execution tree back to
        # the run_id of the specific `task` call it descends from — lets the
        # UI nest each subagent's own tool calls (web_search, web_fetch, ...)
        # under the right "Delegating to sub-agent" badge instead of showing
        # them as flat, unrelated rows. Those tool calls are shown as-is —
        # they're complete, atomic events, safe to stream even with several
        # subagents running in parallel. Their raw model text/thinking tokens
        # are NOT shown: those arrive as many small deltas, and interleave
        # character-by-character into unreadable garbled text when multiple
        # subagents stream concurrently. Token usage is still surfaced either
        # way, for accurate cost accounting.
        subagent_task_root: dict[str, str] = {}
        # Accumulates each task's nested tool_start/tool_end payloads, saved to
        # Postgres once that task finishes (keyed by its tool_call_id) purely
        # so the user can still see them after a reload — never read back into
        # the model's own context on later turns.
        nested_events_by_task: dict[str, list[dict]] = {}
        try:
            async for event in graph.astream_events(graph_input, config=config, version="v2"):
                run_id = event.get("run_id", "")
                parent_ids = event.get("parent_ids") or []
                is_task_start = event["event"] == "on_tool_start" and event.get("name") == "task"
                is_task_end = event["event"] == "on_tool_end" and event.get("name") == "task"
                task_run_id = next((subagent_task_root[pid] for pid in parent_ids if pid in subagent_task_root), None)

                if is_task_start:
                    subagent_task_root[run_id] = run_id
                elif task_run_id:
                    subagent_task_root[run_id] = task_run_id

                results: list[dict] | None = None

                if event["event"] == "on_chat_model_stream":
                    if task_run_id:
                        usage_event = self._usage_event(event["data"]["chunk"])
                        results = [usage_event] if usage_event else None
                    else:
                        results = self._handle_token(event, parser, viz_indexes) or None
                elif event["event"] == "on_tool_start":
                    results = self._tool_start.handle(event, task_run_id=task_run_id)
                    if task_run_id:
                        nested_events_by_task.setdefault(task_run_id, []).extend(results)
                elif event["event"] == "on_tool_end":
                    results = self._tool_end.handle(event, task_run_id=task_run_id)
                    if task_run_id:
                        nested_events_by_task.setdefault(task_run_id, []).extend(results)
                    elif is_task_end:
                        trace = nested_events_by_task.get(run_id)
                        tool_call_id = self._task_tool_call_id(event)
                        if trace and tool_call_id:
                            await repository.save_subagent_trace(get_pool(), thread_id, tool_call_id, trace)

                if results:
                    for data in results:
                        yield f"data: {json.dumps(data)}\n\n"

            # After normal stream completion check for pending HITL interrupt
            state = await graph.aget_state(config)
            hitl_lines = _extract_hitl_events(state)

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        for line in hitl_lines:
            yield f"data: {line}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    async def stream(self, thread_id: str, content: str, graph, thinking_effort: str = "high", model: str | None = None):
        config = _make_config(thread_id, thinking_effort, model)
        async for chunk in self._run_graph(
            {"messages": [HumanMessage(content=content)]}, config, graph
        ):
            yield chunk

    async def resume(self, thread_id: str, decision: str, graph):
        config = _make_config(thread_id)

        # Count pending action_requests so we send exactly N decisions
        n = 1
        state = await graph.aget_state(config)
        for interrupt in getattr(state, "interrupts", ()):
            value = interrupt.value
            if isinstance(value, dict) and "action_requests" in value:
                n = len(value["action_requests"])
                break

        decisions = [{"type": decision}] * n
        async for chunk in self._run_graph(
            Command(resume={"decisions": decisions}), config, graph
        ):
            yield chunk


chat_service = ChatService()
