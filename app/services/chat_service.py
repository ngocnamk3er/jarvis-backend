import json
from abc import ABC, abstractmethod
from langchain_core.messages import HumanMessage


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

class BaseEventHandler(ABC):
    @abstractmethod
    def handle(self, event: dict) -> list[dict] | None: ...


class TokenEventHandler(BaseEventHandler):
    def handle(self, event: dict) -> list[dict] | None:
        chunk = event["data"]["chunk"]

        if chunk.content:
            return [{"type": "token", "content": chunk.content}]

        tool_chunks = getattr(chunk, "tool_call_chunks", None)
        if not tool_chunks:
            return None

        events = []
        for tc in tool_chunks:
            args_delta = tc.get("args", "") or ""
            name = tc.get("name") or ""
            if name or args_delta:
                events.append({
                    "type": "tool_chunk",
                    "index": tc.get("index", 0),
                    "name": name,
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

class ChatService:
    def __init__(self) -> None:
        self._factory = EventHandlerFactory()

    async def stream(self, thread_id: str, content: str, graph):
        config = {"configurable": {"thread_id": thread_id}}
        try:
            async for event in graph.astream_events(
                {"messages": [HumanMessage(content=content)]},
                config=config,
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
