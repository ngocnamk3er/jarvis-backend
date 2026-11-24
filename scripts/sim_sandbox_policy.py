"""Simulate the sandbox admission policy on a synthetic workload: old (global idlest) vs new (per-user + longest-away).

No LLM and no cluster. It runs the REAL `sandbox_manager._admission` / `_pause_idlest` against an in-memory cluster
and a simulated clock, so what is measured is the shipped code, not a re-implementation of it.

  old = no `jarvis-user` label and no per-user limit: every conversation is its own "user", so the pause order is
        "the single idlest conversation", which is what the code did before.
  new = claims carry the user label, AGENTSANDBOX_MAX_PODS_PER_USER applies, and the shared ceiling pauses a sandbox
        of the user who has been away longest.

Both see the same generated workload per seed (paired comparison).

    .venv/bin/python scripts/sim_sandbox_policy.py            # tables to stdout
    .venv/bin/python scripts/sim_sandbox_policy.py --seeds 40

Model (all numbers are assumptions, change them below): "normal" users have 1-2 conversations and send a message every
~2.5 min; "heavy" users (25 %) keep 6-10 conversations open and fire runs across them every ~1 min. A run lasts tens of
seconds, and while it runs it calls the sandbox every 8-25 s. A request that finds no slot is counted as DENIED at once
(the real system waits up to 45 s first, so denial is somewhat pessimistic for both policies alike).
"""

import argparse
import asyncio
import heapq
import math
import random
import statistics
import sys
from datetime import datetime as _real_datetime
from datetime import timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.tools import sandbox_manager as sm  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.core.conversation_lock import ConversationLocks  # noqa: E402

_REAL_PAUSE = sm._pause_idlest


async def _never_pause(*args, **kwargs):
    return None


START = _real_datetime(2026, 1, 1, tzinfo=timezone.utc)
HORIZON_MIN = 180
SAMPLE_EVERY_S = 30
REGRET_WINDOW_S = 600           # a pause counts as a mistake if its conversation is needed again within this long
# Who is on the system. "mixed": a quarter of users keep many conversations open and use them in turn. "hog": a few
# users fan out over many conversations at once (parallel research, many tabs), the case the per-user limit is for.
PROFILES = {
    "mixed": dict(heavy_share=0.25, heavy_convs=(6, 10), heavy_gap=60.0, heavy_median=90.0, heavy_use=0.7),
    "hog": dict(heavy_share=0.10, heavy_convs=(12, 20), heavy_gap=15.0, heavy_median=120.0, heavy_use=0.95),
}


class _Clock:
    t = START


class _FakeDatetime(_real_datetime):
    @classmethod
    def now(cls, tz=None):
        return _Clock.t


def _fmt(t) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


class Cluster:
    """Claims + sandboxes in memory; pausing a sandbox = patching its shutdownTime to now."""

    def __init__(self):
        self.claims, self.sandboxes = {}, {}
        self.forced: list[tuple[float, str]] = []           # (sim seconds, thread) of every pause made to make room

    async def list_namespaced_custom_object(self, namespace, group, version, plural):
        return {"items": list((self.claims if plural == "sandboxclaims" else self.sandboxes).values())}

    async def patch_namespaced_custom_object(self, namespace, name, body, group, version, plural, **kw):
        self.sandboxes[name]["spec"]["shutdownTime"] = body["spec"]["shutdownTime"]
        self.forced.append(((_Clock.t - START).total_seconds(), name.removeprefix("sb-")))

    def deadline(self, thread):
        sb = self.sandboxes.get(f"sb-{thread}")
        return sm._parse_time(sb["spec"]["shutdownTime"]) if sb else None

    def pods(self) -> int:
        now = _Clock.t
        return sum(1 for s in self.sandboxes.values() if sm._parse_time(s["spec"]["shutdownTime"]) > now)


# ---- workload ------------------------------------------------------------------------------------------------------


def make_workload(rng: random.Random, n_users: int, profile: dict):
    """-> (runs, heavy): runs = [(t_start_s, user, conv, duration_s, [call offsets])], heavy = set of heavy users."""
    heavy = {u for u in range(n_users) if rng.random() < profile["heavy_share"]}
    runs = []
    for u in range(n_users):
        is_heavy = u in heavy
        n_conv = rng.randint(*profile["heavy_convs"]) if is_heavy else rng.randint(1, 2)
        t = rng.uniform(0, 40 * 60)
        end = min(t + rng.uniform(60, 120) * 60, HORIZON_MIN * 60)
        gap = profile["heavy_gap"] if is_heavy else 150.0
        median = profile["heavy_median"] if is_heavy else 50.0
        use_p = profile["heavy_use"] if is_heavy else 0.6
        conv, busy_until = 0, {}
        while t < end:
            if is_heavy or (n_conv > 1 and rng.random() < 0.2):
                conv = rng.randrange(n_conv)
            dur = min(max(10.0, rng.lognormvariate(math.log(median), 0.6)), 600.0)
            if busy_until.get(conv, -1) <= t:
                calls = []
                if rng.random() < use_p:
                    off = rng.uniform(3, 15)
                    while off < dur:
                        calls.append(off)
                        off += rng.uniform(8, 25)
                    calls = calls or [min(5.0, dur)]
                runs.append((t, u, conv, dur, calls))
                busy_until[conv] = t + dur
            t += dur + rng.expovariate(1 / gap)
    return runs, heavy


# ---- one simulation ------------------------------------------------------------------------------------------------


async def simulate(runs, heavy, new_policy: bool, max_pods: int, per_user: int, ttl_s: int, cap_only: bool = False):
    sm.datetime = _FakeDatetime
    cluster = Cluster()
    client = SimpleNamespace(k8s_helper=SimpleNamespace(custom_objects_api=cluster))
    locks = ConversationLocks()
    sm._get_client = lambda: client
    sm.conversation_locks = locks
    sm._reserved, sm._in_use, sm._owners, sm._admit_lock = set(), {}, {}, asyncio.Lock()
    settings.AGENTSANDBOX_MAX_PODS = max_pods
    settings.AGENTSANDBOX_MAX_PODS_PER_USER = per_user if new_policy else 0
    settings.AGENTSANDBOX_TTL_SECONDS = ttl_s
    settings.AGENTSANDBOX_EVICT_MIN_IDLE_SECONDS = 120
    settings.AGENTSANDBOX_ADMIT_WAIT_SECONDS = 0

    async def owner_of(label):
        return label.split("c")[0] if new_policy else None

    sm._owner_of = owner_of
    # cap_only: the ceiling alone. Nothing is ever paused to make room; a request waits for a pod to expire on its own.
    sm._pause_idlest = _never_pause if cap_only else _REAL_PAUSE

    m = SimpleNamespace(
        runs=0, runs_sbx=0, denied=0, denied_normal=0, denied_heavy=0, sbx_normal=0, sbx_heavy=0,
        cold=0, regret=0, samples=[], forced_on_normal=0,
    )
    events, seq = [], 0

    def push(t, kind, *payload):
        nonlocal seq
        seq += 1
        heapq.heappush(events, (t, seq, kind, payload))

    for r in runs:
        push(r[0], "start", *r)
    for s in range(0, HORIZON_MIN * 60 + 1, SAMPLE_EVERY_S):
        push(s, "sample")
    leases, denied_runs = {}, set()
    last_forced = {}

    async def use(thread, user):
        """Mirror of _get_or_create_sandbox + _touch_expiry: what one sandbox call does to the cluster."""
        label = sm._label_value(thread)
        now = _Clock.t
        deadline = cluster.deadline(thread)
        if deadline is not None and deadline > now:
            cluster.sandboxes[f"sb-{thread}"]["spec"]["shutdownTime"] = _fmt(now + timedelta(seconds=ttl_s))
            return
        n_forced = len(cluster.forced)
        async with sm._admission(label):
            if deadline is None:
                owner = await sm._owner_of(label)
                cluster.claims[f"claim-{thread}"] = {
                    "metadata": {"name": f"claim-{thread}",
                                 "labels": {sm._THREAD_LABEL: label, **({sm._USER_LABEL: owner} if owner else {})}},
                    "status": {"conditions": [{"type": "Ready", "status": "True"}], "sandbox": {"name": f"sb-{thread}"}},
                }
                cluster.sandboxes[f"sb-{thread}"] = {"metadata": {"name": f"sb-{thread}"}, "spec": {}}
            cluster.sandboxes[f"sb-{thread}"]["spec"]["shutdownTime"] = _fmt(now + timedelta(seconds=ttl_s))
        for t_forced, victim in cluster.forced[n_forced:]:
            last_forced[victim] = t_forced
            if int(victim.split("c")[0][1:]) not in heavy:
                m.forced_on_normal += 1
        m.cold += 1
        t_forced = last_forced.get(thread)
        if t_forced is not None and (now - START).total_seconds() - t_forced < REGRET_WINDOW_S:
            m.regret += 1

    while events:
        t, _, kind, payload = heapq.heappop(events)
        _Clock.t = START + timedelta(seconds=t)
        if kind == "sample":
            m.samples.append(cluster.pods())
        elif kind == "start":
            _, u, c, dur, calls = payload
            thread = f"u{u}c{c}"
            m.runs += 1
            m.runs_sbx += bool(calls)
            if u in heavy:
                m.sbx_heavy += bool(calls)
            else:
                m.sbx_normal += bool(calls)
            leases[thread] = await locks.acquire(thread)
            for off in calls:
                push(t + off, "call", thread, u)
            push(t + dur, "end", thread)
        elif kind == "call":
            thread, u = payload
            if thread in denied_runs:
                continue
            try:
                await use(thread, u)
            except sm.SandboxCapacityError:
                denied_runs.add(thread)
                m.denied += 1
                if u in heavy:
                    m.denied_heavy += 1
                else:
                    m.denied_normal += 1
        elif kind == "end":
            (thread,) = payload
            leases.pop(thread).release()
            denied_runs.discard(thread)
            await asyncio.sleep(0)
    for lease in leases.values():
        lease.release()
    forced = len(cluster.forced)
    return {
        "denied%": 100 * m.denied / max(1, m.runs_sbx),
        "denied% normal": 100 * m.denied_normal / max(1, m.sbx_normal),
        "denied% heavy": 100 * m.denied_heavy / max(1, m.sbx_heavy),
        "avg pods": statistics.mean(m.samples),
        "p95 pods": sorted(m.samples)[int(0.95 * len(m.samples))],
        "pod-min": sum(m.samples) * SAMPLE_EVERY_S / 60,
        "cold starts": m.cold,
        "forced pauses": forced,
        "forced on normal %": 100 * m.forced_on_normal / max(1, forced),
        "regret pauses": m.regret,
    }


def run_scenario(seeds, n_users, new_policy, max_pods=10, per_user=3, ttl_s=1800, profile="mixed", cap_only=False):
    rows = []
    for seed in seeds:
        runs, heavy = make_workload(random.Random(seed), n_users, PROFILES[profile])
        rows.append(asyncio.run(simulate(runs, heavy, new_policy, max_pods, per_user, ttl_s, cap_only)))
    return {k: (statistics.mean(r[k] for r in rows), statistics.pstdev(r[k] for r in rows)) for k in rows[0]}


def table(title, rows, cols):
    print(f"\n### {title}\n")
    print("| policy | " + " | ".join(cols) + " |")
    print("|---|" + "---:|" * len(cols))
    for name, res in rows:
        print(f"| {name} | " + " | ".join(f"{res[c][0]:.1f} ±{res[c][1]:.1f}" for c in cols) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--cap-only", action="store_true", help="compare the ceiling alone with the two pausing policies")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    if args.cap_only:
        return cap_only_comparison(seeds)
    cols = ["denied%", "denied% normal", "denied% heavy", "avg pods", "forced pauses",
            "forced on normal %", "regret pauses", "cold starts"]
    print(f"{args.seeds} seeds per cell, {HORIZON_MIN} simulated minutes, mean ±sd")

    def compare(title, n, cap, profile):
        rows = [("old", run_scenario(seeds, n, False, cap, profile=profile))]
        rows += [(f"new, {k}/user", run_scenario(seeds, n, True, cap, per_user=k, profile=profile)) for k in (2, 3)]
        table(title, rows, cols)

    for n in (20, 45):
        compare(f"[mixed] {n} users, ceiling 10", n, 10, "mixed")
    for cap in (15, 20, 30):
        compare(f"[mixed] 30 users, ceiling {cap}", 30, cap, "mixed")
    for cap in (10, 15):
        compare(f"[hog] 30 users (10% fan out over 12-20 conversations), ceiling {cap}", 30, cap, "hog")

    rows = []
    for ttl in (1800, 900, 600, 300):
        rows.append((f"old, ttl {ttl // 60}m", run_scenario(seeds, 30, False, 15, ttl_s=ttl)))
        rows.append((f"new 3/user, ttl {ttl // 60}m", run_scenario(seeds, 30, True, 15, 3, ttl_s=ttl)))
    table("TTL trade-off, 30 users, ceiling 15 (RAM held vs cold starts)", rows,
          ["avg pods", "pod-min", "cold starts", "denied%", "regret pauses"])


def cap_only_comparison(seeds):
    cols = ["denied%", "denied% normal", "avg pods", "pod-min", "cold starts", "forced pauses", "regret pauses"]
    print(f"{len(seeds)} seeds per cell, {HORIZON_MIN} simulated minutes, mean ±sd")
    for n, cap in ((20, 10), (30, 10), (30, 15), (30, 20)):
        rows = [
            ("ceiling only", run_scenario(seeds, n, False, cap, cap_only=True)),
            ("ceiling + pause idlest (old)", run_scenario(seeds, n, False, cap)),
            ("ceiling + per-user 3 (new)", run_scenario(seeds, n, True, cap, 3)),
        ]
        table(f"[mixed] {n} users, ceiling {cap}", rows, cols)
    rows = []
    for ttl in (1800, 600):
        rows.append((f"ceiling only, ttl {ttl // 60}m", run_scenario(seeds, 30, False, 15, ttl_s=ttl, cap_only=True)))
        rows.append((f"pause idlest, ttl {ttl // 60}m", run_scenario(seeds, 30, False, 15, ttl_s=ttl)))
    table("TTL, 30 users, ceiling 15", rows, cols)


if __name__ == "__main__":
    main()
