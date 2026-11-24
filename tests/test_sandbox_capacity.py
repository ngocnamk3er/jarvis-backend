"""The ceiling on sandbox pods: who gets paused to make room, who never does, and what happens when nothing can be.

Runs with the shared state (admission lock, reservations, in-use marks) in this process; the same checks run with it in a
real Redis when REDIS_TEST_URL is set (`docker run --rm -d -p 6390:6379 redis:7-alpine`,
REDIS_TEST_URL=redis://localhost:6390/0), which is how it is kept between backend replicas."""

import asyncio
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.agents.tools import sandbox_manager as sm
from app.core.config import settings
from app.core.conversation_lock import ConversationLocks
from app.core.sandbox_coordination import SandboxCoordination

REDIS_URL = os.environ.get("REDIS_TEST_URL", "")

NOW = datetime.now(timezone.utc)


def _ts(minutes_ago: float) -> str:
    """shutdownTime of a sandbox last used `minutes_ago` minutes ago (deadline = last use + TTL)."""
    last_use = NOW - timedelta(minutes=minutes_ago)
    return (last_use + timedelta(seconds=settings.AGENTSANDBOX_TTL_SECONDS)).strftime("%Y-%m-%dT%H:%M:%SZ")


class FakeCluster:
    """Just enough of the custom-objects API: claims and sandboxes in memory, pause = patch shutdownTime to now."""

    def __init__(self):
        self.claims: dict[str, dict] = {}
        self.sandboxes: dict[str, dict] = {}
        self.paused: list[str] = []

    def add(self, thread: str, idle_minutes: float, running: bool = True, user: str | None = None):
        reason = None if running else "SandboxExpired"
        cond = {"type": "Ready", "status": "True" if running else "False"}
        if reason:
            cond["reason"] = reason
        if not running:
            idle_minutes = 600                                    # expired long ago
        self.claims[f"claim-{thread}"] = {
            "metadata": {"name": f"claim-{thread}", "labels": {sm._THREAD_LABEL: thread, **({sm._USER_LABEL: user} if user else {})}},
            "status": {"conditions": [cond], "sandbox": {"name": f"sb-{thread}"}},
        }
        self.sandboxes[f"sb-{thread}"] = {"metadata": {"name": f"sb-{thread}"}, "spec": {"shutdownTime": _ts(idle_minutes)}}

    async def list_namespaced_custom_object(self, namespace, group, version, plural):
        items = self.claims if plural == "sandboxclaims" else self.sandboxes
        return {"items": list(items.values())}

    async def patch_namespaced_custom_object(self, namespace, name, body, group, version, plural, **kw):
        assert plural == "sandboxes"
        thread = name.removeprefix("sb-")
        self.paused.append(thread)
        claim = self.claims[f"claim-{thread}"]
        claim["status"]["conditions"] = [{"type": "Ready", "status": "False", "reason": "SandboxExpired"}]
        self.sandboxes[name]["spec"]["shutdownTime"] = NOW.strftime("%Y-%m-%dT%H:%M:%SZ")


async def _clear_redis():
    import redis.asyncio as aioredis

    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        for pattern in ("jarvis:sbx:*", "jarvis:conv-lock:*"):
            async for key in r.scan_iter(match=pattern):
                await r.delete(key)
    finally:
        await r.aclose()


@pytest.fixture(params=["memory"] + (["redis"] if REDIS_URL else []))
async def cluster(request, monkeypatch):
    monkeypatch.setattr(settings, "REDIS_URL", REDIS_URL if request.param == "redis" else "")
    if request.param == "redis":
        await _clear_redis()
    c = FakeCluster()
    client = SimpleNamespace(k8s_helper=SimpleNamespace(custom_objects_api=c))
    monkeypatch.setattr(sm, "_get_client", lambda: client)
    monkeypatch.setattr(settings, "AGENTSANDBOX_MAX_PODS", 3)
    monkeypatch.setattr(settings, "AGENTSANDBOX_MAX_PODS_PER_USER", 0)      # the per-user tests turn it on
    monkeypatch.setattr(sm, "_owners", {})
    monkeypatch.setattr(settings, "AGENTSANDBOX_ADMIT_WAIT_SECONDS", 0)
    monkeypatch.setattr(settings, "AGENTSANDBOX_EVICT_MIN_IDLE_SECONDS", 120)
    coord = SandboxCoordination()
    monkeypatch.setattr(sm, "coordination", coord)
    c.coord = coord
    c.owners = {}

    async def owner_of(label):
        return c.owners.get(label)

    monkeypatch.setattr(sm, "_owner_of", owner_of)
    locks = ConversationLocks()
    monkeypatch.setattr(sm, "conversation_locks", locks)
    c.locks = locks
    yield c
    await coord.close()
    await locks.close()
    if request.param == "redis":
        await _clear_redis()


async def test_below_the_ceiling_nothing_is_paused(cluster):
    cluster.add("a", 20)
    cluster.add("b", 20)
    async with sm._admission("new"):
        pass
    assert cluster.paused == []


async def test_paused_sandboxes_do_not_count(cluster):
    for t in "abcdef":
        cluster.add(t, 600, running=False)
        cluster.sandboxes[f"sb-{t}"]["spec"]["shutdownTime"] = _ts(600)      # deadline long past
    async with sm._admission("new"):
        pass
    assert cluster.paused == []


async def test_a_just_revived_sandbox_counts_even_while_its_claim_still_says_expired(cluster):
    """The claim's Ready condition lags the Sandbox by seconds; a request in that window must see the pod."""
    for t in "abd":
        cluster.add(t, 1)
    for t in "abd":
        cluster.claims[f"claim-{t}"]["status"]["conditions"] = [
            {"type": "Ready", "status": "False", "reason": "SandboxExpired"}
        ]                                   # stale: the Sandbox deadline is in the future
        await cluster.locks.acquire(t)
    with pytest.raises(sm.SandboxCapacityError):
        async with sm._admission("c"):
            pass


async def test_at_the_ceiling_the_idlest_sandbox_is_paused(cluster):
    cluster.add("a", 10)
    cluster.add("b", 25)          # idlest
    cluster.add("c", 20)
    async with sm._admission("new"):
        pass
    assert cluster.paused == ["b"]


async def test_a_conversation_with_a_run_going_is_never_paused(cluster):
    cluster.add("a", 10)
    cluster.add("b", 25)          # idlest, but a run holds it
    cluster.add("c", 20)
    lease = await cluster.locks.acquire("b")
    async with sm._admission("new"):
        pass
    assert cluster.paused == ["c"]            # the next idlest
    lease.release()
    await asyncio.sleep(0.05)


async def test_a_call_in_flight_is_never_paused(cluster):
    cluster.add("a", 10)
    cluster.add("b", 25)
    cluster.add("c", 20)
    async with sm._using("b"):
        async with sm._admission("new"):
            pass
    assert cluster.paused == ["c"]


async def test_a_sandbox_used_a_moment_ago_is_not_paused(cluster):
    for t in "abc":
        cluster.add(t, 0.5)       # all used 30 s ago: between two tool calls of a turn
    with pytest.raises(sm.SandboxCapacityError):
        async with sm._admission("new"):
            pass
    assert cluster.paused == []


async def test_when_nothing_can_be_paused_the_request_fails_with_a_clear_error(cluster):
    for t in "abc":
        cluster.add(t, 25)
    for t in "abc":
        await cluster.locks.acquire(t)        # every conversation is mid-run
    with pytest.raises(sm.SandboxCapacityError) as e:
        async with sm._admission("new"):
            pass
    assert "3 sandboxes" in str(e.value)
    assert cluster.paused == []


async def test_it_waits_for_a_slot_that_frees_up(cluster, monkeypatch):
    monkeypatch.setattr(settings, "AGENTSANDBOX_ADMIT_WAIT_SECONDS", 10)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(sm.asyncio, "sleep", lambda s: real_sleep(0.01))
    for t in "abc":
        cluster.add(t, 25)
    leases = [await cluster.locks.acquire(t) for t in "abc"]

    async def finish_one():
        await real_sleep(0.1)
        leases[0].release()                   # the run of "a" ends, so it can be paused

    asyncio.ensure_future(finish_one())
    async with sm._admission("new"):
        pass
    assert cluster.paused == ["a"]


async def test_two_requests_do_not_share_the_one_free_slot(cluster):
    cluster.add("a", 25)
    cluster.add("b", 25)          # two pods of three: one free slot
    for t in "ab":
        await cluster.locks.acquire(t)        # neither can be paused

    admitted, refused = [], []

    async def go(label):
        try:
            async with sm._admission(label):
                admitted.append(label)
                await asyncio.sleep(0.1)       # the pod takes a while to appear
        except sm.SandboxCapacityError:
            refused.append(label)

    await asyncio.gather(go("x"), go("y"))
    assert len(admitted) == 1 and len(refused) == 1


async def test_ceiling_zero_means_no_ceiling(cluster, monkeypatch):
    monkeypatch.setattr(settings, "AGENTSANDBOX_MAX_PODS", 0)
    for t in "abcdefghij":
        cluster.add(t, 25)
    async with sm._admission("new"):
        pass
    assert cluster.paused == []


async def test_a_conversation_that_already_has_a_pod_is_not_blocked_by_the_ceiling(cluster):
    for t in "abc":
        cluster.add(t, 25)
    for t in "abc":
        await cluster.locks.acquire(t)
    async with sm._admission("a"):            # already one of the three
        pass


async def test_exec_bash_reports_capacity_as_an_error_not_a_crash(monkeypatch):
    async def boom(thread_id):
        raise sm.SandboxCapacityError("All 3 sandboxes are in use by other conversations right now.")

    monkeypatch.setattr(sm, "_get_or_create_sandbox", boom)
    result = await sm.exec_bash("t", "ls")
    assert result["timed_out"] is True and "3 sandboxes" in result["error"]


async def test_a_conversation_paused_for_someone_else_counts_again_once_it_is_revived(cluster):
    """Regression: a paused-then-revived sandbox used to stay "free" for a while, letting a third conversation in."""
    for t in "abc":
        cluster.add(t, 25)
    async with sm._admission("d"):            # pauses the idlest ("a" has the oldest use below)
        pass
    paused = cluster.paused[0]
    cluster.sandboxes[f"sb-{paused}"]["spec"]["shutdownTime"] = _ts(0)       # revived: deadline in the future
    cluster.claims[f"claim-{paused}"]["status"]["conditions"] = [{"type": "Ready", "status": "True"}]
    for t in "abc":
        await cluster.locks.acquire(t)        # every run is going again
    with pytest.raises(sm.SandboxCapacityError):
        async with sm._admission("e"):
            pass


async def test_the_user_who_has_been_away_longest_loses_a_sandbox_first(cluster):
    cluster.add("a1", 28, user="ann")          # ann's idlest, but ann is also using a2 right now
    cluster.add("a2", 5, user="ann")
    cluster.add("b1", 25, user="bob")          # bob has been away longer than ann
    async with sm._admission("new"):
        pass
    assert cluster.paused == ["b1"]


async def test_a_user_with_many_old_sandboxes_is_paused_one_at_a_time_idlest_first(cluster):
    cluster.add("b1", 24, user="bob")
    cluster.add("b2", 29, user="bob")          # bob's idlest
    cluster.add("a1", 10, user="ann")
    async with sm._admission("new"):
        pass
    assert cluster.paused == ["b2"]


async def test_a_user_running_something_keeps_all_their_sandboxes_ahead_of_others(cluster):
    cluster.add("a1", 29, user="ann")          # very old, but ann has a run going in a2
    cluster.add("a2", 26, user="ann")
    cluster.add("b1", 20, user="bob")
    await cluster.locks.acquire("a2")
    async with sm._admission("new"):
        pass
    assert cluster.paused == ["b1"]


async def test_the_requesting_user_counts_as_active_and_goes_last(cluster):
    cluster.owners["new"] = "ann"
    cluster.add("a1", 29, user="ann")          # ann is asking for a new sandbox right now
    cluster.add("b1", 24, user="bob")
    cluster.add("b2", 20, user="bob")
    async with sm._admission("new"):
        pass
    assert cluster.paused == ["b1"]


async def test_the_requester_loses_their_own_sandbox_when_nobody_else_can_give_one(cluster):
    cluster.owners["new"] = "ann"
    cluster.add("a1", 29, user="ann")
    cluster.add("b1", 24, user="bob")
    cluster.add("b2", 20, user="bob")
    for t in ("b1", "b2"):
        await cluster.locks.acquire(t)
    async with sm._admission("new"):
        pass
    assert cluster.paused == ["a1"]


async def test_claims_without_a_user_label_are_each_their_own_user(cluster):
    cluster.add("old1", 29)
    cluster.add("old2", 25)
    cluster.add("x", 10, user="ann")
    async with sm._admission("new"):
        pass
    assert cluster.paused == ["old1"]


# ---- the per-user limit ----------------------------------------------------------------------------------------


@pytest.fixture
def per_user(cluster, monkeypatch):
    monkeypatch.setattr(settings, "AGENTSANDBOX_MAX_PODS", 10)
    monkeypatch.setattr(settings, "AGENTSANDBOX_MAX_PODS_PER_USER", 2)
    cluster.owners["new"] = "ann"
    return cluster


async def test_under_the_per_user_limit_nothing_is_paused(per_user):
    per_user.add("a1", 20, user="ann")
    per_user.add("b1", 25, user="bob")
    async with sm._admission("new"):
        pass
    assert per_user.paused == []


async def test_at_the_per_user_limit_the_users_own_idlest_is_paused_not_anyone_elses(per_user):
    per_user.add("a1", 10, user="ann")
    per_user.add("a2", 25, user="ann")          # ann's idlest
    per_user.add("b1", 29, user="bob")          # idler still, but bob is under his limit and the ceiling is far off
    async with sm._admission("new"):
        pass
    assert per_user.paused == ["a2"]


async def test_another_users_pods_do_not_count_against_the_limit(per_user):
    for t in ("b1", "b2", "b3", "b4"):
        per_user.add(t, 20, user="bob")
    per_user.add("a1", 20, user="ann")
    async with sm._admission("new"):
        pass
    assert per_user.paused == []


async def test_at_the_per_user_limit_a_busy_sandbox_is_not_paused_and_the_request_fails_with_a_clear_error(per_user):
    per_user.add("a1", 10, user="ann")
    per_user.add("a2", 25, user="ann")
    await per_user.locks.acquire("a2")
    per_user.add("b1", 29, user="bob")
    # a1 is the only one of ann's that can go, but it was used 10 minutes ago, which is past EVICT_MIN_IDLE: it goes
    async with sm._admission("new"):
        pass
    assert per_user.paused == ["a1"]
    per_user.paused.clear()
    per_user.add("a3", 25, user="ann")
    per_user.owners["newer"] = "ann"
    await per_user.locks.acquire("a3")
    await per_user.locks.acquire("a1")
    per_user.add("a1", 25, user="ann")          # revived and busy again
    with pytest.raises(sm.SandboxCapacityError) as e:
        async with sm._admission("newer"):
            pass
    assert "2 sandboxes" in str(e.value) and "You already have" in str(e.value)
    assert "b1" not in per_user.paused          # never takes bob's to satisfy ann's own limit


async def test_the_shared_ceiling_still_applies_on_top_of_the_per_user_limit(per_user, monkeypatch):
    monkeypatch.setattr(settings, "AGENTSANDBOX_MAX_PODS", 3)
    per_user.add("b1", 28, user="bob")          # bob has been away longest
    per_user.add("c1", 15, user="cat")
    per_user.add("a1", 5, user="ann")           # ann has one: under her limit, but the shared ceiling is reached
    async with sm._admission("new"):
        pass
    assert per_user.paused == ["b1"]


async def test_per_user_limit_zero_means_no_per_user_limit(per_user, monkeypatch):
    monkeypatch.setattr(settings, "AGENTSANDBOX_MAX_PODS_PER_USER", 0)
    for t in ("a1", "a2", "a3", "a4"):
        per_user.add(t, 20, user="ann")
    async with sm._admission("new"):
        pass
    assert per_user.paused == []


async def test_when_the_owner_is_unknown_only_the_shared_ceiling_applies(per_user):
    per_user.owners.clear()                      # the conversation service could not say who "new" belongs to
    for t in ("a1", "a2", "a3"):
        per_user.add(t, 20, user="ann")
    async with sm._admission("new"):
        pass
    assert per_user.paused == []
