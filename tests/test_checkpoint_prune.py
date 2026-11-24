"""A chat turn writes one checkpoint, and the thread's older ones are deleted once the turn is over."""

import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from app.core.config import settings
from app.services import chat_service as cs


class FakeSaver:
    def __init__(self, fail=False):
        self.pruned = []
        self.fail = fail

    async def aprune(self, thread_ids, strategy="delete"):
        if self.fail:
            raise RuntimeError("db down")
        self.pruned.append((list(thread_ids), strategy))


class FakeGraph:
    def __init__(self, saver):
        self.checkpointer = saver
        self.seen = {}

    async def astream_events(self, graph_input, config=None, version="v2", **kw):
        self.seen = kw
        if False:
            yield None

    async def aget_state(self, config):
        return SimpleNamespace(values={"messages": [HumanMessage(content="hi"), AIMessage(content="ok", id="m1")]},
                               interrupts=(), next=())

    async def aupdate_state(self, config, values, **kw):
        pass


async def run_turn(graph, monkeypatch, service=None):
    async def no_reset(thread_id):
        pass

    monkeypatch.setattr(cs, "reset_sandbox", no_reset)
    service = service or cs.ChatService()
    return [line async for line in service._run_graph({"messages": []}, {"configurable": {"thread_id": "t1"}}, graph)]


async def test_a_turn_is_written_once_at_its_end_and_older_checkpoints_are_pruned(monkeypatch):
    saver = FakeSaver()
    graph = FakeGraph(saver)
    lines = await run_turn(graph, monkeypatch)
    assert graph.seen["durability"] == "exit"
    assert saver.pruned == [(["t1"], "keep_latest")]
    assert '"type": "done"' in lines[-1]


async def test_pruning_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(settings, "CHECKPOINT_KEEP_LATEST", False)
    saver = FakeSaver()
    await run_turn(FakeGraph(saver), monkeypatch)
    assert saver.pruned == []


async def test_a_failed_prune_does_not_break_the_turn(monkeypatch):
    lines = await run_turn(FakeGraph(FakeSaver(fail=True)), monkeypatch)
    assert '"type": "done"' in lines[-1] and not any('"type": "error"' in l for l in lines)


async def test_a_thread_with_another_run_in_flight_is_left_alone(monkeypatch):
    saver = FakeSaver()
    service = cs.ChatService()
    other = asyncio.ensure_future(asyncio.sleep(60))
    try:
        graph = FakeGraph(saver)

        async def run():
            # the other request registers itself for the thread after this turn's own run is done
            service._active_runs["t1"] = other

        orig = graph.astream_events

        async def astream_events(*a, **kw):
            async for ev in orig(*a, **kw):
                yield ev
            await run()

        graph.astream_events = astream_events
        await run_turn(graph, monkeypatch, service)
        assert saver.pruned == []
    finally:
        other.cancel()
        service._active_runs.pop("t1", None)


async def test_a_graph_without_a_checkpointer_is_not_pruned():
    await cs.prune_thread(SimpleNamespace(checkpointer=None), "t1")
