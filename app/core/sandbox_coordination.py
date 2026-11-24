"""What the sandbox ceiling has to share between backend replicas.

The ceiling (sandbox_manager._admission) decides from three things that used to live in one process's memory:

* an **admission lock**: one admission decision at a time, so two requests do not both see the one free slot;
* **reservations**: conversations admitted whose pod is not yet visible in the cluster, so the next decision counts
  them;
* **in use**: conversations with a sandbox call in flight, whose pod must not be paused under them.

With two backend replicas each one only saw its own, so both could take the last slot and either could pause a sandbox
the other was running a command in. Here the three live in Redis when REDIS_URL is set, and in this process otherwise
(one replica, or Redis not configured), the same way the conversation lock does:

* the admission lock is `SET key token NX PX ttl`, released only by its owner. Its TTL frees it if the holder dies;
* reservations and in-use marks are sorted sets, member `"<instance>|<label>"` and score the moment it lapses
  (taken from Redis's own clock, not this machine's). A background beat renews what this process still holds, so
  whatever a crashed replica held is gone within ENTRY_TTL_MS. The instance prefix keeps one replica from deleting
  another's mark for the same conversation;
* every Redis call is best-effort: a failure is logged (once in a while, not per call) and that decision is made on
  what this process knows, which is what happened before this module existed. A Redis outage must not stop sandboxes.

Which conversations have a *run* going is not kept here: it is every conversation lock in Redis (see
ConversationLocks.all_running_threads).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import secrets
import time

from app.core.config import settings
from app.core.redis_scripts import ADD_MEMBERS, LIVE_MEMBERS

logger = logging.getLogger(__name__)

KEY_ADMIT = "jarvis:sbx:admit"
KEY_RESERVED = "jarvis:sbx:reserved"
KEY_INUSE = "jarvis:sbx:inuse"

ADMIT_TTL_MS = 20_000  # an admission decision lists the cluster and may pause one sandbox: seconds, not minutes
ADMIT_WAIT_S = 25.0  # longer than the TTL, so a lock whose holder died is always free by the time we give up
ENTRY_TTL_MS = 30_000  # how long a reservation / in-use mark outlives the process that stopped renewing it
BEAT_S = 10.0  # renewal interval, a third of ENTRY_TTL_MS
_WARN_EVERY_S = 30.0

_ADD = ADD_MEMBERS
_MEMBERS = LIVE_MEMBERS
_RELEASE = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"


class SandboxCoordination:
    def __init__(self) -> None:
        self.instance = secrets.token_hex(4)  # which backend process wrote an entry
        self._local_admit = asyncio.Lock()
        self._reserved: dict[str, int] = {}  # label -> holders in this process
        self._in_use: dict[str, int] = {}
        self._r = None
        self._url: str | None = None
        self._beat: asyncio.Task | None = None
        self._last_warn = 0.0

    # ---- Redis plumbing ------------------------------------------------------------------------------------------

    def _client(self):
        """The Redis client, or None when REDIS_URL is not set (everything then stays in this process)."""
        url = settings.REDIS_URL
        if not url:
            return None
        if self._r is None or self._url != url:
            import redis.asyncio as aioredis  # imported here: the in-process mode works without the package

            # Short timeouts: a hung Redis must turn into the in-process fallback quickly, not into a hung request.
            self._r = aioredis.from_url(url, socket_timeout=2, socket_connect_timeout=2, decode_responses=True)
            self._url = url
            self._add = self._r.register_script(_ADD)
            self._members = self._r.register_script(_MEMBERS)
            self._release = self._r.register_script(_RELEASE)
        return self._r

    def _warn(self, what: str) -> None:
        now = time.monotonic()
        if now - self._last_warn >= _WARN_EVERY_S:
            self._last_warn = now
            logger.warning("Redis unavailable for %s; deciding on what this process knows", what, exc_info=True)

    def _member(self, label: str) -> str:
        return f"{self.instance}|{label}"

    async def _publish(self, key: str, labels: list[str]) -> None:
        if not labels or self._client() is None:
            return
        try:
            await self._add(keys=[key], args=[ENTRY_TTL_MS, *[self._member(x) for x in labels]])
        except Exception:  # noqa: BLE001
            self._warn("sandbox coordination")
        self._ensure_beat()

    async def _retract(self, key: str, label: str) -> None:
        r = self._client()
        if r is None:
            return
        try:
            await r.zrem(key, self._member(label))
        except Exception:  # noqa: BLE001 — it lapses by itself within ENTRY_TTL_MS
            self._warn("sandbox coordination")

    async def _labels(self, key: str) -> set[str]:
        """Every label marked under `key` by any replica."""
        if self._client() is None:
            return set()
        try:
            members = await self._members(keys=[key])
        except Exception:  # noqa: BLE001
            self._warn("sandbox coordination")
            return set()
        return {m.split("|", 1)[1] for m in members if "|" in m}

    # ---- keep what this process holds alive ----------------------------------------------------------------------

    def _ensure_beat(self) -> None:
        if self._beat is None or self._beat.done():
            self._beat = asyncio.ensure_future(self._beat_loop())

    async def _beat_loop(self) -> None:
        while self._reserved or self._in_use:
            await asyncio.sleep(BEAT_S)
            for key, held in ((KEY_RESERVED, self._reserved), (KEY_INUSE, self._in_use)):
                if held:
                    await self._publish_quietly(key, list(held))

    async def _publish_quietly(self, key: str, labels: list[str]) -> None:
        if self._client() is None:
            return
        try:
            await self._add(keys=[key], args=[ENTRY_TTL_MS, *[self._member(x) for x in labels]])
        except Exception:  # noqa: BLE001
            self._warn("sandbox coordination")

    # ---- the admission lock --------------------------------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def admit(self):
        """One admission decision at a time: within this process always, across replicas when Redis is there."""
        async with self._local_admit:
            token = await self._take_admit()
            try:
                yield
            finally:
                if token:
                    await self._give_admit(token)

    async def _take_admit(self) -> str | None:
        r = self._client()
        if r is None:
            return None
        token = secrets.token_hex(8)
        deadline = time.monotonic() + ADMIT_WAIT_S
        while True:
            try:
                if await r.set(KEY_ADMIT, token, nx=True, px=ADMIT_TTL_MS):
                    return token
            except Exception:  # noqa: BLE001
                self._warn("the sandbox admission lock")
                return None
            if time.monotonic() >= deadline:
                logger.error("could not take the sandbox admission lock in %ss; deciding without it", ADMIT_WAIT_S)
                return None
            await asyncio.sleep(0.03 + random.random() * 0.05)

    async def _give_admit(self, token: str) -> None:
        try:
            await self._release(keys=[KEY_ADMIT], args=[token])
        except Exception:  # noqa: BLE001 — the TTL frees it
            self._warn("the sandbox admission lock")

    # ---- reservations ---------------------------------------------------------------------------------------------

    async def reserve(self, label: str) -> None:
        """Count conversation `label` as holding a slot, here and for the other replicas."""
        self._reserved[label] = self._reserved.get(label, 0) + 1
        if self._reserved[label] == 1:
            await self._publish(KEY_RESERVED, [label])

    async def unreserve(self, label: str) -> None:
        n = self._reserved.get(label, 0) - 1
        if n > 0:
            self._reserved[label] = n
            return
        self._reserved.pop(label, None)
        await self._retract(KEY_RESERVED, label)

    async def reserved(self) -> set[str]:
        return set(self._reserved) | await self._labels(KEY_RESERVED)

    # ---- in use ---------------------------------------------------------------------------------------------------

    async def mark_in_use(self, label: str) -> None:
        """A sandbox call is starting on `label`: its pod must not be paused until the matching unmark."""
        self._in_use[label] = self._in_use.get(label, 0) + 1
        if self._in_use[label] == 1:
            await self._publish(KEY_INUSE, [label])

    async def unmark_in_use(self, label: str) -> None:
        # The local count first and synchronously, so a cancelled caller still leaves it right; the Redis entry
        # lapses by itself if the call below is cut short.
        n = self._in_use.get(label, 0) - 1
        if n > 0:
            self._in_use[label] = n
            return
        self._in_use.pop(label, None)
        await self._retract(KEY_INUSE, label)

    async def in_use(self) -> set[str]:
        return set(self._in_use) | await self._labels(KEY_INUSE)

    async def close(self) -> None:
        if self._beat is not None:
            self._beat.cancel()
            self._beat = None
        if self._r is not None:
            with contextlib.suppress(Exception):
                await self._r.aclose()
            self._r = None


coordination = SandboxCoordination()
