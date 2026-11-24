"""Talking to the other backend replicas: who is running what, and telling the one that runs it to stop.

A chat run and a background research task live in the memory of the process that started them. With more than one
replica a request can land on another one, which then cannot stop them and cannot tell whether a research task is still
going. This module is the small piece of shared state that fixes that, without moving any run anywhere:

* Every process has an `INSTANCE` id (the pod name plus a suffix). It is put in front of the token of every
  conversation lock, so the lock itself says which process is running the conversation (ConversationLocks.owner_of).
* To stop something another process runs, `cancel_remote` writes a short-lived cancel key and then publishes a message on
  that process's channel (`jarvis:ctl:<instance>`). The owner listens on it from start-up and acts at once. The key is the
  safety net: a Pub/Sub message is gone if the owner's connection happened to be down, so the owner also looks for cancel
  keys of what it runs every few seconds.
* Research tasks are registered here while they run (a sorted set renewed by their owner, so a crashed owner's entries
  lapse by themselves) and their final status is kept after, so another process can answer "is it still going" and
  "how did it end" correctly instead of calling a task it does not know "interrupted".

Everything is best-effort and does nothing without REDIS_URL (one replica needs none of it).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import os
import secrets
import socket
import time
from collections.abc import Awaitable, Callable

from app.core.config import settings
from app.core.redis_scripts import ADD_MEMBERS, LIVE_MEMBERS

logger = logging.getLogger(__name__)

# The pod name (the hostname in Kubernetes) with the process id and a suffix, so a restart is a different instance.
INSTANCE = f"{socket.gethostname()}-{os.getpid()}-{secrets.token_hex(3)}"

KEY_RESEARCH_RUNNING = "jarvis:research:running"
CANCEL_TTL_S = 30  # a cancel key nobody picked up is not left around to cancel something later
RESEARCH_RECORD_TTL_S = 86_400
ENTRY_TTL_MS = 30_000  # how long a running research task stays listed after its owner stops renewing
BEAT_S = 10.0
POLL_S = 3.0  # how often the owner looks for cancel keys it may have missed
_WARN_EVERY_S = 30.0


def channel_of(instance: str) -> str:
    return f"jarvis:ctl:{instance}"


def cancel_key(kind: str, key: str) -> str:
    return f"jarvis:cancel:{kind}:{key}"


def research_key(task_id: str) -> str:
    return f"jarvis:research:{task_id}"


async def _call(handler: Callable[[str], Awaitable | None], arg: str) -> None:
    result = handler(arg)
    if inspect.isawaitable(result):
        await result


class RunControl:
    def __init__(self) -> None:
        self.instance = INSTANCE
        self._r = None
        self._url: str | None = None
        self._research: set[str] = set()  # research tasks running in this process
        self._tasks: list[asyncio.Task] = []
        self._on_cancel_run: Callable | None = None
        self._on_cancel_research: Callable | None = None
        self._active_runs: Callable[[], dict[str, float]] = dict
        self._last_warn = 0.0

    # ---- plumbing ----------------------------------------------------------------------------------------------------

    def _client(self):
        """The Redis client, or None when REDIS_URL is not set."""
        url = settings.REDIS_URL
        if not url:
            return None
        if self._r is None or self._url != url:
            import redis.asyncio as aioredis  # imported here: one replica without Redis does not need the package

            self._r = aioredis.from_url(url, socket_timeout=2, socket_connect_timeout=2, decode_responses=True)
            self._url = url
            self._add = self._r.register_script(ADD_MEMBERS)
            self._live = self._r.register_script(LIVE_MEMBERS)
        return self._r

    def _warn(self, what: str) -> None:
        now = time.monotonic()
        if now - self._last_warn >= _WARN_EVERY_S:
            self._last_warn = now
            logger.warning("Redis unavailable for %s", what, exc_info=True)

    # ---- listening for cancel requests (this process is the owner) ---------------------------------------------------

    async def start(
        self,
        on_cancel_run: Callable[[str], Awaitable | None],
        on_cancel_research: Callable[[str], Awaitable | None],
        active_runs: Callable[[], dict[str, float]],
    ) -> None:
        """Start listening. `on_cancel_run(thread_id)` / `on_cancel_research(task_id)` stop what this process runs.
        `active_runs()` is {thread_id: when the run started} for the runs going here."""
        if self._client() is None:
            return
        self._on_cancel_run, self._on_cancel_research, self._active_runs = on_cancel_run, on_cancel_research, active_runs
        self._tasks = [asyncio.ensure_future(self._listen()), asyncio.ensure_future(self._housekeeping())]

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks = []
        if self._r is not None:
            with contextlib.suppress(Exception):
                await self._r.aclose()
            self._r = None

    async def _listen(self) -> None:
        import redis.asyncio as aioredis

        while True:
            # Its own connection, without a read timeout: a quiet channel is not a hung Redis.
            conn = aioredis.from_url(settings.REDIS_URL, socket_connect_timeout=2, health_check_interval=15, decode_responses=True)
            pubsub = conn.pubsub()
            try:
                await pubsub.subscribe(channel_of(self.instance))
                async for message in pubsub.listen():
                    if message.get("type") == "message":
                        await self._handle(message["data"])
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — reconnect; what was missed meanwhile the polling below picks up
                self._warn("the control channel")
                await asyncio.sleep(1)
            finally:
                with contextlib.suppress(Exception):
                    await pubsub.aclose()
                with contextlib.suppress(Exception):
                    await conn.aclose()

    async def _handle(self, raw: str) -> None:
        try:
            message = json.loads(raw)
            kind, key = message["kind"], message["key"]
        except (ValueError, KeyError, TypeError):
            return
        r = self._client()
        with contextlib.suppress(Exception):
            await r.delete(cancel_key(kind, key))  # handled here, so the safety net must not fire again
        handler = self._on_cancel_run if kind == "run" else self._on_cancel_research if kind == "research" else None
        if handler is not None:
            await _call(handler, key)

    async def _housekeeping(self) -> None:
        """Renew what this process has registered, and look for cancel keys the channel did not deliver."""
        last_renewal = 0.0
        while True:
            await asyncio.sleep(POLL_S)
            try:
                await self._poll_cancel_keys()
                if time.monotonic() - last_renewal >= BEAT_S:
                    last_renewal = time.monotonic()
                    if self._research:
                        await self._add(keys=[KEY_RESEARCH_RUNNING], args=[ENTRY_TTL_MS, *self._research])
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                self._warn("renewing the research registry")

    async def _poll_cancel_keys(self) -> None:
        r = self._client()
        runs = self._active_runs()
        wanted = [("run", t) for t in runs] + [("research", t) for t in self._research]
        if not wanted:
            return
        values = await r.mget([cancel_key(kind, key) for kind, key in wanted])
        for (kind, key), value in zip(wanted, values):
            if value is None:
                continue
            await r.delete(cancel_key(kind, key))
            if kind == "run":
                try:
                    asked_at = float(value)
                except ValueError:
                    asked_at = 0.0
                if asked_at < runs[key]:
                    continue  # asked before this run began: meant for an earlier one, which has since ended
                await _call(self._on_cancel_run, key)
            else:
                await _call(self._on_cancel_research, key)

    # ---- asking another process to stop something ------------------------------------------------------------------

    async def cancel_remote(self, kind: str, key: str, owner: str) -> bool:
        """Ask process `owner` to stop the run (kind "run", key = thread id) or research task (kind "research") it
        has. True when the request was put out; there is no way to know here that it arrived."""
        r = self._client()
        if r is None:
            return False
        try:
            await r.set(cancel_key(kind, key), str(time.time()), ex=CANCEL_TTL_S)  # first: the safety net
            await r.publish(channel_of(owner), json.dumps({"kind": kind, "key": key}))
            return True
        except Exception:  # noqa: BLE001
            self._warn("a cancel request")
            return False

    # ---- research tasks ---------------------------------------------------------------------------------------------

    async def track_research(self, task_id: str) -> None:
        """A research task is starting in this process: list it as running, with this process as its owner."""
        self._research.add(task_id)
        r = self._client()
        if r is None:
            return
        try:
            await self._add(keys=[KEY_RESEARCH_RUNNING], args=[ENTRY_TTL_MS, task_id])
            await r.hset(research_key(task_id), mapping={"owner": self.instance, "status": "running", "error": ""})
            await r.expire(research_key(task_id), RESEARCH_RECORD_TTL_S)
        except Exception:  # noqa: BLE001
            self._warn("the research registry")

    async def untrack_research(self, task_id: str, status: str, error: str | None = None) -> None:
        """It ended: take it off the running list and keep how it ended."""
        self._research.discard(task_id)
        r = self._client()
        if r is None:
            return
        try:
            await r.zrem(KEY_RESEARCH_RUNNING, task_id)
            await r.hset(research_key(task_id), mapping={"status": status, "error": error or ""})
            await r.expire(research_key(task_id), RESEARCH_RECORD_TTL_S)
        except Exception:  # noqa: BLE001
            self._warn("the research registry")

    async def research_running(self) -> set[str]:
        """Ids of the research tasks running in any process."""
        if self._client() is None:
            return set()
        try:
            return set(await self._live(keys=[KEY_RESEARCH_RUNNING]))
        except Exception:  # noqa: BLE001
            self._warn("the research registry")
            return set()

    async def research_record(self, task_id: str) -> dict | None:
        """What is known about a task from any process: {"status", "error", "owner", "alive"}; None if nothing.
        `alive` is whether its owner is still renewing it."""
        r = self._client()
        if r is None:
            return None
        try:
            record = await r.hgetall(research_key(task_id))
            if not record:
                return None
            alive = task_id in await self._live(keys=[KEY_RESEARCH_RUNNING])
        except Exception:  # noqa: BLE001
            self._warn("the research registry")
            return None
        return {"status": record.get("status", ""), "error": record.get("error") or None, "owner": record.get("owner"), "alive": alive}


run_control = RunControl()
