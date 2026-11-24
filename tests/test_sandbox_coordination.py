"""The state the sandbox ceiling shares between backend replicas.

A replica is stood in for by a SandboxCoordination of its own (its own instance id, its own in-process counts) on the same
Redis. The cross-replica tests need a real Redis and run when REDIS_TEST_URL is set
(`docker run --rm -d -p 6390:6379 redis:7-alpine`, REDIS_TEST_URL=redis://localhost:6390/0); the rest run without."""

import asyncio
import os
from types import SimpleNamespace

import pytest

from app.agents.tools import sandbox_manager as sm
from app.core import sandbox_coordination as sc
from app.core.config import settings
from app.core.conversation_lock import ConversationLocks
from app.core.sandbox_coordination import SandboxCoordination
from tests.test_sandbox_capacity import FakeCluster

REDIS_URL = os.environ.get("REDIS_TEST_URL", "")
needs_redis = pytest.mark.skipif(not REDIS_URL, reason="set REDIS_TEST_URL to run the cross-replica checks")


async def _clear_redis():
    import redis.asyncio as aioredis

    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        for pattern in ("jarvis:sbx:*", "jarvis:conv-lock:*"):
            async for key in r.scan_iter(match=pattern):
                await r.delete(key)
    finally:
        await r.aclose()


@pytest.fixture
async def replicas(monkeypatch):
    """Two replicas on one Redis (or none, when the test turns Redis off by itself)."""
    monkeypatch.setattr(settings, "REDIS_URL", REDIS_URL)
    if REDIS_URL:
        await _clear_redis()
    a, b = SandboxCoordination(), SandboxCoordination()
    yield a, b
    await a.close()
    await b.close()
    if REDIS_URL:
        await _clear_redis()


# ---- in one process (no Redis needed) -----------------------------------------------------------------------------


@pytest.fixture
def alone(monkeypatch):
    monkeypatch.setattr(settings, "REDIS_URL", "")
    return SandboxCoordination()


async def test_a_reservation_held_twice_is_kept_until_both_let_go(alone):
    await alone.reserve("c")
    await alone.reserve("c")
    await alone.unreserve("c")
    assert "c" in await alone.reserved()
    await alone.unreserve("c")
    assert "c" not in await alone.reserved()


async def test_in_use_is_counted_per_caller(alone):
    await alone.mark_in_use("c")
    await alone.mark_in_use("c")
    await alone.unmark_in_use("c")
    assert "c" in await alone.in_use()
    await alone.unmark_in_use("c")
    assert "c" not in await alone.in_use()


async def test_letting_go_of_something_never_held_does_nothing(alone):
    await alone.unreserve("never")
    await alone.unmark_in_use("never")
    assert await alone.reserved() == set() and await alone.in_use() == set()


async def test_admissions_in_one_process_never_overlap(alone):
    inside = 0
    worst = 0

    async def decide():
        nonlocal inside, worst
        async with alone.admit():
            inside += 1
            worst = max(worst, inside)
            await asyncio.sleep(0.02)
            inside -= 1

    await asyncio.gather(*[decide() for _ in range(5)])
    assert worst == 1


async def test_redis_down_falls_back_to_this_process(monkeypatch):
    monkeypatch.setattr(settings, "REDIS_URL", "redis://127.0.0.1:1/0")      # nothing listens there
    coord = SandboxCoordination()
    async with coord.admit():
        await coord.reserve("c")
        await coord.mark_in_use("c")
    assert "c" in await coord.reserved() and "c" in await coord.in_use()
    await coord.unreserve("c")
    await coord.unmark_in_use("c")
    assert await coord.reserved() == set()
    await coord.close()


# ---- across replicas (a real Redis) -------------------------------------------------------------------------------


@needs_redis
async def test_another_replica_sees_a_reservation_and_only_its_owner_removes_it(replicas):
    a, b = replicas
    await a.reserve("c")
    assert "c" in await b.reserved()
    await b.unreserve("c")                      # B never reserved it: A's stays
    assert "c" in await a.reserved()
    await a.unreserve("c")
    assert "c" not in await b.reserved()


@needs_redis
async def test_two_replicas_with_the_same_conversation_do_not_remove_each_others_mark(replicas):
    a, b = replicas
    await a.mark_in_use("c")
    await b.mark_in_use("c")
    await b.unmark_in_use("c")
    assert "c" in await b.in_use()              # still A's
    await a.unmark_in_use("c")
    assert "c" not in await b.in_use()


@needs_redis
async def test_the_admission_lock_holds_the_other_replica_back(replicas):
    a, b = replicas
    order: list[str] = []
    a_inside = asyncio.Event()

    async def first():
        async with a.admit():
            order.append("a in")
            a_inside.set()
            await asyncio.sleep(0.3)
            order.append("a out")

    async def second():
        await a_inside.wait()
        async with b.admit():
            order.append("b in")

    await asyncio.gather(first(), second())
    assert order == ["a in", "a out", "b in"]


@needs_redis
async def test_a_dead_holder_does_not_keep_the_admission_lock(replicas, monkeypatch):
    monkeypatch.setattr(sc, "ADMIT_TTL_MS", 300)
    a, b = replicas
    assert await a._take_admit() is not None    # taken and never given back: the replica died
    loop = asyncio.get_running_loop()
    start = loop.time()
    async with b.admit():
        waited = loop.time() - start
    assert 0.2 < waited < 2


@needs_redis
async def test_what_a_crashed_replica_held_lapses_by_itself(replicas, monkeypatch):
    monkeypatch.setattr(sc, "ENTRY_TTL_MS", 300)
    a, b = replicas
    await a.reserve("r")
    await a.mark_in_use("u")
    assert "r" in await b.reserved() and "u" in await b.in_use()
    a._beat.cancel()                            # the replica is gone: nothing renews its entries
    await asyncio.sleep(0.5)
    assert "r" not in await b.reserved() and "u" not in await b.in_use()


@needs_redis
async def test_a_live_replica_renews_what_it_holds(replicas, monkeypatch):
    monkeypatch.setattr(sc, "ENTRY_TTL_MS", 600)
    monkeypatch.setattr(sc, "BEAT_S", 0.15)
    a, b = replicas
    await a.reserve("r")
    await a.mark_in_use("u")
    await asyncio.sleep(1.2)                    # twice the TTL
    assert "r" in await b.reserved() and "u" in await b.in_use()
    await a.unreserve("r")
    await a.unmark_in_use("u")
    assert await b.reserved() == set() and await b.in_use() == set()


@needs_redis
async def test_a_conversation_run_by_another_replica_is_listed_as_running():
    import redis.asyncio as aioredis  # noqa: F401 (imported to fail early when the package is missing)

    settings.REDIS_URL = REDIS_URL
    await _clear_redis()
    a, b = ConversationLocks(), ConversationLocks()
    try:
        lease = await a.acquire("t-elsewhere")
        assert lease is not None
        assert "t-elsewhere" not in b.running_threads()                 # this process does not hold it
        assert "t-elsewhere" in await b.all_running_threads()           # but it is running
        lease.release()
        await asyncio.sleep(0.1)
        assert "t-elsewhere" not in await b.all_running_threads()
    finally:
        await a.close()
        await b.close()
        await _clear_redis()


# ---- the ceiling itself, decided by two replicas --------------------------------------------------------------------


@pytest.fixture
def ceiling(monkeypatch):
    """A cluster at 2 of 3 slots, both recently used (so neither may be paused), and a local stand-in for each replica."""
    c = FakeCluster()
    c.add("a", 0.5)
    c.add("b", 0.5)
    monkeypatch.setattr(sm, "_get_client", lambda: SimpleNamespace(k8s_helper=SimpleNamespace(custom_objects_api=c)))
    monkeypatch.setattr(settings, "AGENTSANDBOX_MAX_PODS", 3)
    monkeypatch.setattr(settings, "AGENTSANDBOX_MAX_PODS_PER_USER", 0)
    monkeypatch.setattr(settings, "AGENTSANDBOX_ADMIT_WAIT_SECONDS", 0)
    monkeypatch.setattr(settings, "AGENTSANDBOX_EVICT_MIN_IDLE_SECONDS", 120)
    monkeypatch.setattr(sm, "_owners", {})

    async def no_owner(label):
        return None

    monkeypatch.setattr(sm, "_owner_of", no_owner)
    monkeypatch.setattr(sm, "conversation_locks", ConversationLocks())
    return c


async def _try_to_get_in(coord, label, hold=0.3):
    try:
        async with sm._admission(label, coord):
            await asyncio.sleep(hold)
        return "in"
    except sm.SandboxCapacityError:
        return "refused"


@needs_redis
async def test_two_replicas_asking_for_the_last_slot_at_once_do_not_both_get_it(ceiling, replicas):
    a, b = replicas
    outcome = await asyncio.gather(_try_to_get_in(a, "x"), _try_to_get_in(b, "y"))
    assert sorted(outcome) == ["in", "refused"]


async def test_without_the_shared_state_two_replicas_both_take_the_last_slot(ceiling, monkeypatch):
    """What happened before: each replica counted only its own reservations. Kept as the control for the test above."""
    monkeypatch.setattr(settings, "REDIS_URL", "")
    a, b = SandboxCoordination(), SandboxCoordination()
    outcome = await asyncio.gather(_try_to_get_in(a, "x"), _try_to_get_in(b, "y"))
    assert outcome == ["in", "in"]


@needs_redis
async def test_a_sandbox_in_use_on_another_replica_is_not_paused(ceiling, replicas):
    a, b = replicas
    ceiling.add("c", 10)                                      # the idlest of three
    ceiling.add("old", 600, running=False)                    # does not count
    await a.mark_in_use("c")                                  # replica A is running a command in it
    ceiling.sandboxes["sb-a"]["spec"]["shutdownTime"] = _ts_minutes(15)   # a is the next idlest
    outcome = await _try_to_get_in(b, "new", hold=0)
    assert outcome == "in"
    assert ceiling.paused == ["a"]                            # not "c"


async def test_without_the_shared_state_a_sandbox_in_use_elsewhere_is_paused(ceiling, monkeypatch):
    monkeypatch.setattr(settings, "REDIS_URL", "")
    a, b = SandboxCoordination(), SandboxCoordination()
    ceiling.add("c", 10)
    await a.mark_in_use("c")                                  # invisible to B
    ceiling.sandboxes["sb-a"]["spec"]["shutdownTime"] = _ts_minutes(5)
    await _try_to_get_in(b, "new", hold=0)
    assert ceiling.paused == ["c"]


def _ts_minutes(minutes_ago: float) -> str:
    from tests.test_sandbox_capacity import _ts

    return _ts(minutes_ago)
