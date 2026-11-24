"""One run at a time per conversation.

Two requests that run the graph on the same thread at once each start from the same checkpoint and each write
their own at the end, so one of the two turns is lost (seen: two requests, two answers streamed, two messages
left instead of four). The send button is only disabled in the browser, so a second tab or a direct API call
got through. `acquire()` is the server-side gate: the chat endpoints take the lock before they start streaming
and answer 409 when somebody else holds it.

The lock is a Redis key (`SET key token NX PX ttl`) so it holds across backend replicas, but:

* it is a *lease*: a short TTL, renewed by a heartbeat while the run is alive, so a crashed backend never leaves a
  conversation locked for long (at most one TTL);
* it is released only by its owner (compare the token, in Lua), so a lease that expired and was taken by another
  request is never deleted by the old holder;
* it is held until the graph task has finished, not just until the response ends: a browser that goes away
  mid-stream does not stop the run (see ChatService._run_graph), and the next request must wait for it;
* with no `REDIS_URL`, or when Redis is unreachable, it falls back to an in-process lock. That is still correct
  for one replica, which is how the backend runs today (chat_service keeps its in-flight runs in a per-process
  dict, see backend.yaml), so Redis being down degrades the lock to per-process instead of failing every chat.
"""
from __future__ import annotations

import asyncio
import logging
import secrets
import time
from typing import Awaitable, Protocol

from app.core.config import settings
from app.core.run_control import INSTANCE

logger = logging.getLogger(__name__)

KEY_PREFIX = "jarvis:conv-lock:"

_RENEW = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('pexpire', KEYS[1], ARGV[2]) else return 0 end"
_RELEASE = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"


class _Backend(Protocol):
    async def holder(self, key: str) -> str | None: ...
    async def acquire(self, key: str, token: str, ttl_ms: int) -> bool: ...
    async def renew(self, key: str, token: str, ttl_ms: int) -> bool: ...
    async def release(self, key: str, token: str) -> None: ...


class MemoryBackend:
    """The same lease semantics inside one process."""

    def __init__(self) -> None:
        self._held: dict[str, tuple[str, float]] = {}

    async def holder(self, key: str) -> str | None:
        current = self._held.get(key)
        return current[0] if current and current[1] > time.monotonic() else None

    async def acquire(self, key: str, token: str, ttl_ms: int) -> bool:
        now = time.monotonic()
        current = self._held.get(key)
        if current and current[1] > now:
            return False
        self._held[key] = (token, now + ttl_ms / 1000)
        return True

    async def renew(self, key: str, token: str, ttl_ms: int) -> bool:
        current = self._held.get(key)
        if not current or current[0] != token or current[1] <= time.monotonic():
            return False
        self._held[key] = (token, time.monotonic() + ttl_ms / 1000)
        return True

    async def release(self, key: str, token: str) -> None:
        current = self._held.get(key)
        if current and current[0] == token:
            del self._held[key]


class RedisBackend:
    def __init__(self, url: str) -> None:
        import redis.asyncio as aioredis  # imported here: the memory backend works without the package

        # Short timeouts: a hung Redis must turn into the fallback quickly, not into a hung chat request.
        self._r = aioredis.from_url(url, socket_timeout=2, socket_connect_timeout=2, decode_responses=True)
        self._renew = self._r.register_script(_RENEW)
        self._release = self._r.register_script(_RELEASE)

    async def holder(self, key: str) -> str | None:
        return await self._r.get(key)

    async def acquire(self, key: str, token: str, ttl_ms: int) -> bool:
        return bool(await self._r.set(key, token, nx=True, px=ttl_ms))

    async def renew(self, key: str, token: str, ttl_ms: int) -> bool:
        return bool(await self._renew(keys=[key], args=[token, ttl_ms]))

    async def release(self, key: str, token: str) -> None:
        await self._release(keys=[key], args=[token])

    async def locked_threads(self) -> set[str]:
        """Every conversation that has a lease right now, whichever replica holds it."""
        threads: set[str] = set()
        async for key in self._r.scan_iter(match=KEY_PREFIX + "*", count=200):
            threads.add(key[len(KEY_PREFIX):])
        return threads

    async def close(self) -> None:
        await self._r.aclose()


class Lease:
    """A held lock. Create with `ConversationLocks.acquire`; give it back with `release()`."""

    def __init__(self, locks: "ConversationLocks", thread_id: str, backend: _Backend, token: str) -> None:
        self._locks = locks
        self.thread_id = thread_id
        self._backend = backend
        self._key = KEY_PREFIX + thread_id
        self._token = token
        self._heartbeat: asyncio.Task | None = None
        self._until: list[asyncio.Future] = []
        self._finishing = False

    def start_heartbeat(self) -> None:
        """Keep the lease alive. Not started at acquire time: a response that is never iterated (the client left
        before the first byte) must not hold the conversation, so until the stream really starts the TTL runs out
        by itself."""
        if self._heartbeat is None:
            self._heartbeat = asyncio.ensure_future(self._beat())

    def hold_until(self, task: Awaitable) -> None:
        """Do not release before `task` (the graph run) is done, even if `release()` is called earlier."""
        self._until.append(asyncio.ensure_future(task))

    async def _beat(self) -> None:
        interval = self._locks.ttl_ms / 3000
        while True:
            await asyncio.sleep(interval)
            try:
                if await self._backend.renew(self._key, self._token, self._locks.ttl_ms):
                    continue
                # The key is gone: Redis restarted, or the lease ran out while this process was stalled. Take it
                # back if nobody else did, so the run stays protected.
                if await self._backend.acquire(self._key, self._token, self._locks.ttl_ms):
                    logger.warning("conversation lock for %s had expired and was re-taken", self.thread_id)
                else:
                    logger.error("conversation lock for %s was lost to another request", self.thread_id)
                    return
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — a failed renewal is retried on the next tick
                logger.warning("could not renew the conversation lock for %s", self.thread_id, exc_info=True)

    def release(self) -> None:
        """Give the lock back once every task registered with `hold_until` is done. Returns at once."""
        if self._finishing:
            return
        self._finishing = True
        asyncio.ensure_future(self._finish())

    async def _finish(self) -> None:
        pending = [t for t in self._until if not t.done()]
        if pending:
            await asyncio.wait(pending)
        if self._heartbeat is not None:
            self._heartbeat.cancel()
        try:
            await self._backend.release(self._key, self._token)
        except Exception:  # noqa: BLE001 — the TTL frees it anyway
            logger.warning("could not release the conversation lock for %s", self.thread_id, exc_info=True)
        self._locks._forget(self)


class ConversationLocks:
    def __init__(self) -> None:
        self.ttl_ms = 30_000
        self._memory = MemoryBackend()
        self._redis: RedisBackend | None = None
        self._redis_url: str | None = None
        self._held: dict[str, Lease] = {}

    def _primary(self) -> _Backend:
        url = settings.REDIS_URL
        if not url:
            return self._memory
        if self._redis is None or self._redis_url != url:
            self._redis, self._redis_url = RedisBackend(url), url
        return self._redis

    async def acquire(self, thread_id: str) -> Lease | None:
        """A lease on the conversation, or None when another request is running on it."""
        # The token starts with this process's instance id, so the lock says who is running the conversation
        # (owner_of): that is how a Stop that reaches another replica finds the one to ask.
        token = f"{INSTANCE}:{secrets.token_hex(16)}"
        backend = self._primary()
        try:
            got = await backend.acquire(KEY_PREFIX + thread_id, token, self.ttl_ms)
        except Exception:  # noqa: BLE001 — Redis down: still keep two runs in this process apart
            logger.warning("Redis unavailable for the conversation lock; using the in-process lock", exc_info=True)
            backend = self._memory
            got = await backend.acquire(KEY_PREFIX + thread_id, token, self.ttl_ms)
        if not got:
            return None
        lease = Lease(self, thread_id, backend, token)
        self._held[thread_id] = lease
        return lease

    async def owner_of(self, thread_id: str) -> str | None:
        """The instance id of the process that holds the conversation's lock, or None when nobody holds it."""
        try:
            token = await self._primary().holder(KEY_PREFIX + thread_id)
        except Exception:  # noqa: BLE001
            logger.warning("could not read the conversation lock of %s", thread_id, exc_info=True)
            return None
        return token.split(":", 1)[0] if token and ":" in token else None

    def hold_until(self, thread_id: str, task: Awaitable) -> None:
        """Called by the run itself: the lease for this thread (if this process holds one) outlives `task`."""
        lease = self._held.get(thread_id)
        if lease is not None:
            lease.hold_until(task)

    def running_threads(self) -> set[str]:
        """Conversations that have a run going in this process."""
        return set(self._held)

    async def all_running_threads(self) -> set[str]:
        """Conversations that have a run going in any backend process: what the sandbox ceiling must not pause.

        This process's own, plus every lease in Redis (a lease is held from the start of a request until its run is
        done, which is exactly "a run is going"). Redis failing leaves this process's own, as before."""
        threads = set(self._held)
        backend = self._primary()
        if isinstance(backend, RedisBackend):
            try:
                threads |= await backend.locked_threads()
            except Exception:  # noqa: BLE001
                logger.warning("could not list the conversation locks in Redis", exc_info=True)
        return threads

    def _forget(self, lease: Lease) -> None:
        if self._held.get(lease.thread_id) is lease:
            del self._held[lease.thread_id]

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.close()
            self._redis = None


conversation_locks = ConversationLocks()
