"""Telling another replica to stop something, and sharing the state of research tasks between replicas.

A replica is stood in for by a RunControl of its own (its own instance id) on the same Redis. The cross-replica checks need
a real Redis: set REDIS_TEST_URL (`docker run --rm -d -p 6390:6379 redis:7-alpine`, REDIS_TEST_URL=redis://localhost:6390/0)."""

import asyncio
import os
import time
from types import SimpleNamespace

import pytest

from app.agents.async_research import ResearchRunner
from app.core import run_control as rc
from app.core.config import settings
from app.core.conversation_lock import ConversationLocks
from app.core.run_control import RunControl
from app.services.chat_service import ChatService

REDIS_URL = os.environ.get("REDIS_TEST_URL", "")
needs_redis = pytest.mark.skipif(not REDIS_URL, reason="set REDIS_TEST_URL to run the cross-replica checks")


async def _clear():
    import redis.asyncio as aioredis

    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        for pattern in ("jarvis:cancel:*", "jarvis:research:*", "jarvis:conv-lock:*"):
            async for key in r.scan_iter(match=pattern):
                await r.delete(key)
    finally:
        await r.aclose()


async def wait_until(condition, timeout=3.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        if condition():
            return True
        await asyncio.sleep(0.02)
    return False


def replica(name: str) -> RunControl:
    rcn = RunControl()
    rcn.instance = name
    return rcn


@pytest.fixture
async def two(monkeypatch):
    """Two replicas, a and b, on a clean Redis; each remembers what it was asked to cancel."""
    monkeypatch.setattr(settings, "REDIS_URL", REDIS_URL)
    monkeypatch.setattr(rc, "POLL_S", 0.1)
    await _clear()
    a, b = replica("pod-a"), replica("pod-b")
    a.asked, b.asked = [], []
    a.runs, b.runs = {}, {}
    yield a, b
    await a.close()
    await b.close()
    await _clear()


async def listen(r: RunControl):
    await r.start(
        on_cancel_run=lambda t: r.asked.append(("run", t)),
        on_cancel_research=lambda t: r.asked.append(("research", t)),
        active_runs=lambda: r.runs,
    )
    await asyncio.sleep(0.15)                                   # subscribed


# ---- stopping what another replica runs ----------------------------------------------------------------------------


@needs_redis
async def test_a_cancel_reaches_the_replica_that_owns_the_run(two):
    a, b = two
    await listen(a)
    assert await b.cancel_remote("run", "thread-1", "pod-a") is True
    assert await wait_until(lambda: a.asked == [("run", "thread-1")])
    assert b.asked == []


@needs_redis
async def test_a_research_cancel_reaches_its_owner(two):
    a, b = two
    await listen(a)
    await b.cancel_remote("research", "task-9", "pod-a")
    assert await wait_until(lambda: a.asked == [("research", "task-9")])


@needs_redis
async def test_a_cancel_the_channel_missed_is_picked_up_from_the_key(two):
    a, b = two
    a.runs = {"thread-2": time.time() - 5}                      # a run that began five seconds ago
    await b.cancel_remote("run", "thread-2", "pod-a")           # a is not listening yet: the message goes nowhere
    await listen(a)
    assert await wait_until(lambda: a.asked == [("run", "thread-2")])


@needs_redis
async def test_a_cancel_aimed_at_an_earlier_run_does_not_stop_the_next_one(two):
    a, b = two
    await b.cancel_remote("run", "thread-3", "pod-a")           # nobody listens, the key waits
    await asyncio.sleep(0.05)
    a.runs = {"thread-3": time.time()}                          # a new run of the same conversation began after it
    await listen(a)
    await asyncio.sleep(0.5)
    assert a.asked == []


@needs_redis
async def test_a_handled_cancel_does_not_fire_twice(two):
    a, b = two
    a.runs = {"thread-4": time.time() - 5}
    await listen(a)
    await b.cancel_remote("run", "thread-4", "pod-a")
    assert await wait_until(lambda: len(a.asked) >= 1)
    await asyncio.sleep(0.5)                                    # several polls later
    assert a.asked == [("run", "thread-4")]


async def test_without_redis_nothing_can_be_asked_of_another_replica(monkeypatch):
    monkeypatch.setattr(settings, "REDIS_URL", "")
    r = replica("pod-x")
    assert await r.cancel_remote("run", "t", "pod-y") is False
    assert await r.research_running() == set()
    assert await r.research_record("task") is None
    await r.track_research("task")                              # does nothing, does not fail
    await r.untrack_research("task", "success")


# ---- who runs the conversation --------------------------------------------------------------------------------------


@needs_redis
async def test_the_conversation_lock_names_the_replica_that_holds_it(monkeypatch):
    monkeypatch.setattr(settings, "REDIS_URL", REDIS_URL)
    await _clear()
    here, elsewhere = ConversationLocks(), ConversationLocks()
    try:
        lease = await here.acquire("t-owner")
        assert lease is not None
        assert await here.owner_of("t-owner") == rc.INSTANCE
        assert await elsewhere.owner_of("t-owner") == rc.INSTANCE      # another process reads the same name off the lock
        assert await elsewhere.owner_of("t-nobody") is None
        lease.release()
    finally:
        await here.close()
        await elsewhere.close()
        await _clear()


async def test_a_stop_for_a_run_this_process_does_not_have_is_sent_to_the_replica_that_does(monkeypatch):
    sent = []

    async def cancel_remote(kind, key, owner):
        sent.append((kind, key, owner))
        return True

    async def owner_of(thread_id):
        return "pod-elsewhere"

    monkeypatch.setattr("app.services.chat_service.run_control", SimpleNamespace(instance="pod-here", cancel_remote=cancel_remote))
    monkeypatch.setattr("app.services.chat_service.conversation_locks", SimpleNamespace(owner_of=owner_of))
    service = ChatService()
    assert await service.stop("thread-r") is True
    assert sent == [("run", "thread-r", "pod-elsewhere")]


async def test_a_stop_for_a_run_nobody_has_does_nothing(monkeypatch):
    async def owner_of(thread_id):
        return None

    monkeypatch.setattr("app.services.chat_service.conversation_locks", SimpleNamespace(owner_of=owner_of))
    assert await ChatService().stop("thread-none") is False


async def test_a_stop_for_a_run_held_by_this_very_process_is_not_sent_anywhere(monkeypatch):
    sent = []

    async def cancel_remote(*a):
        sent.append(a)
        return True

    async def owner_of(thread_id):
        return "pod-here"

    monkeypatch.setattr("app.services.chat_service.run_control", SimpleNamespace(instance="pod-here", cancel_remote=cancel_remote))
    monkeypatch.setattr("app.services.chat_service.conversation_locks", SimpleNamespace(owner_of=owner_of))
    assert await ChatService().stop("thread-mine") is False       # its lock, but no run to cancel any more
    assert sent == []


async def test_a_local_run_is_stopped_here_without_asking_anyone(monkeypatch):
    async def never(*a):
        raise AssertionError("must not ask another replica")

    monkeypatch.setattr("app.services.chat_service.run_control", SimpleNamespace(instance="pod-here", cancel_remote=never))
    cancelled = []

    async def reset(thread_id):
        cancelled.append(thread_id)

    monkeypatch.setattr("app.services.chat_service.reset_sandbox", reset)
    service = ChatService()
    gate = asyncio.Event()
    task = asyncio.ensure_future(gate.wait())
    service._active_runs["thread-l"] = task
    assert await service.stop("thread-l") is True
    await asyncio.sleep(0)
    assert task.cancelled() and cancelled == ["thread-l"]


# ---- shutting down --------------------------------------------------------------------------------------------------


async def test_drain_waits_for_the_runs_and_reports_the_ones_left():
    service = ChatService()
    quick = asyncio.ensure_future(asyncio.sleep(0.1))
    slow = asyncio.ensure_future(asyncio.sleep(5))
    service._active_runs.update({"quick": quick, "slow": slow})
    assert await service.drain(0.4) == 1                         # the quick one finished, the slow one is still going
    slow.cancel()
    assert await service.drain(0.4) == 0


async def test_drain_with_nothing_running_returns_at_once():
    start = time.monotonic()
    assert await ChatService().drain(30) == 0
    assert time.monotonic() - start < 0.5


# ---- research tasks seen from another replica -------------------------------------------------------------------------


@needs_redis
async def test_another_replica_sees_a_research_task_running_and_how_it_ended(two):
    a, b = two
    await a.track_research("task-1")
    record = await b.research_record("task-1")
    assert record == {"status": "running", "error": None, "owner": "pod-a", "alive": True}
    assert "task-1" in await b.research_running()
    await a.untrack_research("task-1", "error", "the search tool failed")
    record = await b.research_record("task-1")
    assert (record["status"], record["error"], record["alive"]) == ("error", "the search tool failed", False)
    assert await b.research_running() == set()


@needs_redis
async def test_a_research_task_whose_replica_died_stops_looking_alive(two, monkeypatch):
    monkeypatch.setattr(rc, "ENTRY_TTL_MS", 300)
    a, b = two
    await a.track_research("task-2")
    assert (await b.research_record("task-2"))["alive"] is True
    await asyncio.sleep(0.5)                                     # nobody renews it: a is gone
    record = await b.research_record("task-2")
    assert record["status"] == "running" and record["alive"] is False


class _Graph:
    """Just enough of a compiled graph for ResearchRunner.inspect: the thread has not run to its end."""

    async def aget_state(self, config):
        return SimpleNamespace(values={}, next=("model",))


@needs_redis
async def test_inspect_reports_a_task_of_another_replica_as_running_not_interrupted(two, monkeypatch):
    a, b = two
    monkeypatch.setattr("app.agents.async_research.run_control", b)
    runner = ResearchRunner()
    runner._graph = _Graph()
    await a.track_research("task-3")
    status, error, _ = await runner.inspect("task-3", "running")
    assert (status, error) == ("running", None)
    await a.untrack_research("task-3", "success")
    status, _, _ = await runner.inspect("task-3", "running")
    assert status == "success"


@needs_redis
async def test_inspect_still_calls_a_task_of_a_dead_replica_interrupted(two, monkeypatch):
    monkeypatch.setattr(rc, "ENTRY_TTL_MS", 300)
    a, b = two
    monkeypatch.setattr("app.agents.async_research.run_control", b)
    runner = ResearchRunner()
    runner._graph = _Graph()
    await a.track_research("task-4")
    await asyncio.sleep(0.5)
    status, _, _ = await runner.inspect("task-4", "running")
    assert status == "interrupted"


@needs_redis
async def test_a_research_task_can_be_cancelled_from_another_replica(two, monkeypatch):
    a, b = two
    monkeypatch.setattr("app.agents.async_research.run_control", b)
    await listen(a)
    await a.track_research("task-5")
    runner = ResearchRunner()
    assert await runner.cancel_anywhere("task-5") is True        # not its own, so it asks the owner
    assert await wait_until(lambda: a.asked == [("research", "task-5")])
    assert await runner.cancel_anywhere("task-unknown") is False


@needs_redis
async def test_the_research_limit_counts_every_replica(two, monkeypatch):
    a, b = two
    monkeypatch.setattr("app.agents.async_research.run_control", b)
    await a.track_research("t1")
    await a.track_research("t2")
    assert await ResearchRunner().global_running() == 2           # nothing runs here, two run on pod-a
