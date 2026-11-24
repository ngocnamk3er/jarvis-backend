"""One run at a time per conversation: the lock, its lease, and how the chat endpoints use it.

Runs against the in-process backend; the same checks run against a real Redis when REDIS_TEST_URL is set
(`docker run --rm -p 6390:6379 redis:7-alpine`, REDIS_TEST_URL=redis://localhost:6390/0)."""

import asyncio
import os

import pytest
from fastapi import HTTPException

from app.api.v1.endpoints import chat as chat_endpoint
from app.core import conversation_lock as cl
from app.core.config import settings
from app.services.chat_service import ChatService

REDIS_URL = os.environ.get("REDIS_TEST_URL", "")


@pytest.fixture(params=["memory"] + (["redis"] if REDIS_URL else []))
async def locks(request, monkeypatch):
    monkeypatch.setattr(settings, "REDIS_URL", REDIS_URL if request.param == "redis" else "")
    lk = cl.ConversationLocks()
    lk.ttl_ms = 300
    yield lk
    await lk.close()


async def _settle():
    await asyncio.sleep(0.05)


async def test_second_acquire_on_the_same_conversation_is_refused(locks):
    a = await locks.acquire("t-same")
    assert a is not None
    assert await locks.acquire("t-same") is None
    a.release()
    await _settle()


async def test_other_conversations_are_not_affected(locks):
    a = await locks.acquire("t-one")
    b = await locks.acquire("t-two")
    assert a is not None and b is not None
    a.release()
    b.release()
    await _settle()


async def test_release_frees_the_conversation(locks):
    a = await locks.acquire("t-free")
    a.release()
    await _settle()
    b = await locks.acquire("t-free")
    assert b is not None
    b.release()
    await _settle()


async def test_lease_runs_out_by_itself_without_a_heartbeat(locks):
    """A response that never starts (client left before the first byte) must not hold the conversation."""
    a = await locks.acquire("t-ttl")
    assert a is not None
    await asyncio.sleep(0.5)
    b = await locks.acquire("t-ttl")
    assert b is not None
    b.release()
    await _settle()


async def test_heartbeat_keeps_the_lease_alive(locks):
    a = await locks.acquire("t-beat")
    a.start_heartbeat()
    await asyncio.sleep(0.9)                       # three TTLs
    assert await locks.acquire("t-beat") is None
    a.release()
    await _settle()
    assert (b := await locks.acquire("t-beat")) is not None
    b.release()
    await _settle()


async def test_an_expired_holder_cannot_release_the_new_holders_lock(locks):
    old = await locks.acquire("t-owner")
    await asyncio.sleep(0.5)                       # old expired, never renewed
    new = await locks.acquire("t-owner")
    assert new is not None
    old.release()                                  # must not delete new's key
    await _settle()
    assert await locks.acquire("t-owner") is None
    new.release()
    await _settle()


async def test_release_waits_for_the_graph_run(locks):
    """The browser can leave mid-run and the run carries on; the conversation stays locked until it is done."""
    lease = await locks.acquire("t-run")
    lease.start_heartbeat()
    done = asyncio.Event()

    async def run():
        await done.wait()

    task = asyncio.ensure_future(run())
    locks.hold_until("t-run", task)
    lease.release()                                # the response ended
    await asyncio.sleep(0.6)                       # past a TTL: only the heartbeat holds it
    assert await locks.acquire("t-run") is None
    done.set()
    await task
    await _settle()
    again = await locks.acquire("t-run")
    assert again is not None
    again.release()
    await _settle()


async def test_guarded_releases_when_the_stream_ends_and_when_it_is_abandoned(locks, monkeypatch):
    monkeypatch.setattr(cl, "conversation_locks", locks)

    async def three():
        for i in range(3):
            yield i

    lease = await locks.acquire("t-guard")
    assert [c async for c in ChatService.guarded(lease, three())] == [0, 1, 2]
    await _settle()
    lease = await locks.acquire("t-guard")
    assert lease is not None

    gen = ChatService.guarded(lease, three())      # the client disconnects after the first chunk
    assert await gen.__anext__() == 0
    await gen.aclose()
    await _settle()
    again = await locks.acquire("t-guard")
    assert again is not None
    again.release()
    await _settle()


async def test_redis_down_falls_back_to_the_in_process_lock(monkeypatch):
    monkeypatch.setattr(settings, "REDIS_URL", "redis://127.0.0.1:1/0")      # nothing listens there
    lk = cl.ConversationLocks()
    a = await lk.acquire("t-down")
    assert a is not None                                                      # chat still works
    assert await lk.acquire("t-down") is None                                 # and two runs are still kept apart
    a.release()
    await _settle()
    await lk.close()


async def test_endpoint_answers_409_while_the_conversation_is_running(monkeypatch):
    lk = cl.ConversationLocks()
    monkeypatch.setattr(chat_endpoint, "conversation_locks", lk)
    first = await chat_endpoint._lock_conversation("t-409")
    with pytest.raises(HTTPException) as e:
        await chat_endpoint._lock_conversation("t-409")
    assert e.value.status_code == 409
    first.release()
    await _settle()
    (await chat_endpoint._lock_conversation("t-409")).release()
    await _settle()
