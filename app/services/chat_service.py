import json
from abc import ABC, abstractmethod
from langchain_core.messages import HumanMessage, AIMessage

from app.agents.graph import agent_graph
from app.schemas.chat import Message


# ---------------------------------------------------------------------------
# Event handlers  (return list[dict] so one event can emit N SSE lines)
# ---------------------------------------------------------------------------

class BaseEventHandler(ABC):
    @abstractmethod
    def handle(self, event: dict) -> list[dict] | None: ...


class TokenEventHandler(BaseEventHandler):
    def handle(self, event: dict) -> list[dict] | None:
        chunk = event["data"]["chunk"]

        # Regular text token
        if chunk.content:
            return [{"type": "token", "content": chunk.content}]

        # Streaming tool-call argument chunks
        tool_chunks = getattr(chunk, "tool_call_chunks", None)
        if not tool_chunks:
            return None

        events = []
        for tc in tool_chunks:
            args_delta = tc.get("args", "")
            if args_delta:
                events.append({
                    "type": "tool_chunk",
                    "index": tc.get("index", 0),
                    "name": tc.get("name") or "",
                    "args_delta": args_delta,
                })
        return events or None


class ToolStartEventHandler(BaseEventHandler):
    def handle(self, event: dict) -> list[dict] | None:
        return [{
            "type": "tool_start",
            "name": event["name"],
            "input": event["data"].get("input"),
        }]


class ToolEndEventHandler(BaseEventHandler):
    def handle(self, event: dict) -> list[dict] | None:
        raw = event["data"].get("output")
        output = raw.content if hasattr(raw, "content") else str(raw)
        return [{
            "type": "tool_end",
            "name": event["name"],
            "output": output,
        }]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class EventHandlerFactory:
    def __init__(self) -> None:
        self._handlers: dict[str, BaseEventHandler] = {
            "on_chat_model_stream": TokenEventHandler(),
            "on_tool_start": ToolStartEventHandler(),
            "on_tool_end": ToolEndEventHandler(),
        }

    def get_handler(self, event_type: str) -> BaseEventHandler | None:
        return self._handlers.get(event_type)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

_ROLE_MAP = {"user": HumanMessage, "assistant": AIMessage}


class ChatService:
    def __init__(self) -> None:
        self._factory = EventHandlerFactory()

    def _to_lc_messages(self, messages: list[Message]) -> list:
        return [
            _ROLE_MAP[m.role](content=m.content)
            for m in messages
            if m.role in _ROLE_MAP
        ]

    async def stream(self, messages: list[Message]):
        lc_messages = self._to_lc_messages(messages)
        try:
            async for event in agent_graph.astream_events(
                {"messages": lc_messages},
                version="v2",
            ):
                handler = self._factory.get_handler(event["event"])
                if handler:
                    results = handler.handle(event)
                    if results:
                        for data in results:
                            yield f"data: {json.dumps(data)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"


chat_service = ChatService()
