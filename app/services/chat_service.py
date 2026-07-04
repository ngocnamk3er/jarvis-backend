import json
from langchain_core.messages import HumanMessage
from langgraph.types import Command

# ---------------------------------------------------------------------------
# Thinking parser — splits <think>…</think> out of the content stream
# ---------------------------------------------------------------------------


class ThinkingParser:
    """State machine that routes streaming tokens to thinking_token vs token."""

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


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


VIZ_TOOLS = {"generate_visualization_mermaid", "generate_visualization_svg", "generate_animation", "generate_webapp"}


class ToolStartEventHandler:
    def handle(self, event: dict) -> list[dict]:
        if event["name"] in VIZ_TOOLS:
            return []
        return [{"type": "tool_start", "name": event["name"], "input": event["data"].get("input"), "run_id": event.get("run_id", "")}]


class ToolEndEventHandler:
    def handle(self, event: dict) -> list[dict]:
        raw = event["data"].get("output")
        output = raw.content if hasattr(raw, "content") else str(raw)
        try:
            data = json.loads(output)
            if "__viz__" in data:
                return [{"type": "viz", "format": data["__viz__"], "code": data["code"], "title": data.get("title", "")}]
        except Exception:
            pass
        return [{"type": "tool_end", "name": event["name"], "output": output, "run_id": event.get("run_id", "")}]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(thread_id: str, thinking_effort: str = "high", model: str | None = None) -> dict:
    return {
        "configurable": {"thread_id": thread_id, "thinking_effort": thinking_effort, "model": model},
        "recursion_limit": 50,
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

        # Tool call chunks — suppress for viz tools (they render as viz blocks, not badges)
        for tc in getattr(chunk, "tool_call_chunks", None) or []:
            name = tc.get("name") or ""
            index = tc.get("index", 0)
            args_delta = tc.get("args", "") or ""
            if name and name in VIZ_TOOLS:
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
        try:
            async for event in graph.astream_events(graph_input, config=config, version="v2"):
                results: list[dict] | None = None

                if event["event"] == "on_chat_model_stream":
                    results = self._handle_token(event, parser, viz_indexes) or None
                elif event["event"] == "on_tool_start":
                    results = self._tool_start.handle(event)
                elif event["event"] == "on_tool_end":
                    results = self._tool_end.handle(event)

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
        async for chunk in self._run_graph(
            Command(resume={"decisions": [{"type": decision}]}), config, graph
        ):
            yield chunk


chat_service = ChatService()
