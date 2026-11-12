"""The context-size gauge follows each model call, and survives a Stop."""

import asyncio
import json

from langchain_core.messages import AIMessageChunk

from app.clients import conversation_client
from app.services.chat_service import chat_service


def _model_call(total: int) -> dict:
    chunk = AIMessageChunk(
        content="",
        usage_metadata={"input_tokens": total - 10, "output_tokens": 10, "total_tokens": total},
    )
    return {"event": "on_chat_model_stream", "run_id": f"r{total}", "parent_ids": [], "data": {"chunk": chunk}}


class FakeGraph:
    """Two model calls, then a tool that never returns — the user hits Stop."""

    async def astream_events(self, graph_input, config=None, version=None):
        yield _model_call(1000)
        yield _model_call(2500)
        await asyncio.sleep(3600)


async def test_gauge_updates_per_call_and_is_saved_when_stopped(monkeypatch):
    saved: list[int] = []

    async def set_tokens(thread_id, tokens):
        saved.append(tokens)

    monkeypatch.setattr(conversation_client, "set_context_tokens", set_tokens)

    seen: list[int] = []
    config = {"configurable": {"thread_id": "t-gauge"}}
    async for line in chat_service._run_graph({"messages": []}, config, FakeGraph()):
        ev = json.loads(line.removeprefix("data: "))
        if ev["type"] == "context_tokens":
            seen.append(ev["tokens"])
            if len(seen) == 2:
                # what stop() does first (the sandbox reset after it is not under test)
                chat_service._active_runs["t-gauge"].cancel()

    assert seen == [1000, 2500]   # live, one per model call, before any turn end
    assert saved == [1000, 2500]  # already stored: a reload after Stop reads this
