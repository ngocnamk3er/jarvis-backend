"""Client for kubernetes-sigs/agent-sandbox — talks to agent-sandbox's
controller (CRDs `SandboxClaim`/`Sandbox` in `settings.AGENTSANDBOX_NAMESPACE`)
to give each conversation its own sandbox pod, running agent-sandbox's own
stock `python-runtime-sandbox` image. See SANDBOX-SETUP.md in this repo for
the full deployment.

Written against `k8s-agent-sandbox`'s *actual* installed behavior, verified
live against a real cluster while building this — not the docs site, which
is stale/inconsistent in several places (its getting-started guide never
mentions the SandboxWarmPool that `create_sandbox()` requires, for one).

Design decisions, and why — each verified live, not just read off docs:

- **`SandboxDirectConnectionConfig` through the router, not
  `SandboxInClusterConnectionConfig`.** Going pod-direct (the client
  resolving a pod IP and talking straight to :8888) skips the only place
  any authorization can happen: sandbox pods serve `/execute`,
  `/download/<path>` and friends to anyone who can reach them, by design,
  trusting NetworkPolicy + the router to be the only things that can. That
  is not hypothetical — from inside one sandbox pod, a plain HTTP call to
  another live sandbox pod's IP read its files and ran commands in it.
  NetworkPolicy is not enforced on this cluster, so routing through the
  router is what is left. Note this does **not** close that pod-to-pod
  path (nothing in the router can — that traffic never reaches it); it is
  a known, accepted gap. See SANDBOX-SETUP.md's "Known limitation".

- **The router runs agent-sandbox's own published image, not a self-built
  one.** An earlier pass here built the Go router from the `v1.0.2` source
  with `--authz-mode=tokenreview --cache-enabled=true`, on two beliefs
  about the published image that were both tested on 2026-09-15 and turned
  out wrong. First, the 502s blamed on a missing pod-IP cache: the SDK
  resolves the pod IP itself from the Sandbox CR's `.status.podIPs` (see
  `async_sandbox.get_pod_ip` — it reads the *Sandbox*, so the existing
  `sandboxes: get` grant is enough, no `pods` RBAC needed) and sends it as
  `X-Sandbox-Pod-IP`, so the router only forwards. That is also why
  upstream's own YAML ships no ServiceAccount at all. Second, that the
  image had no auth: it does, via `ALLOW_UNAUTHENTICATED_ROUTER` plus a
  shared `ROUTER_AUTH_TOKEN`, and with it enabled an unauthenticated call
  gets a real 401. The deployment currently sets that flag to `"true"`
  (no token required) — a scope decision, not a limitation of the image.

  Worth keeping straight, since the tokenreview setup was once described
  here as the fix for cross-conversation access: it never was.
  `tokenreview` only *authenticates* the caller, and jarvis-backend uses
  one ServiceAccount token for every conversation, so it never
  distinguished one conversation from another. Per-conversation isolation
  comes entirely from `_get_or_create_sandbox()` below giving each
  `thread_id` its own claim, hence its own pod — verified live with two
  conversations where B could neither list nor read A's files.

- **A Kubernetes label carries the `thread_id -> claim_name` mapping,
  not a database.** `create_sandbox()` always generates its own random
  `sandbox-claim-<uuid8>` claim name (see `async_sandbox_client.py`,
  not overridable) — so reattaching to the same sandbox for a later
  `bash` call in the same conversation can't just recompute the claim
  name from `thread_id`. Labeling the claim with the (sanitized)
  `thread_id` at creation and looking it up via
  `list_all_sandboxes(label_selector=...)` avoids needing new state
  anywhere else — Kubernetes is already the source of truth for "does
  this conversation have a sandbox".

- **Command timeouts surface as an exception here, not a `timed_out`
  field.** `k8s_agent_sandbox`'s `CommandExecutor.run(command, timeout)`
  has no graceful timeout result — `timeout` is just the HTTP client's
  request timeout, and blowing it raises (wrapped as `SandboxRequestError`).
  Caught here and translated to a `{"timed_out": True, ...}` shape so
  `bash.py` doesn't need to special-case this module.

`read_file()` doesn't catch anything itself — a 404 (missing file) or 400
(path is a directory) from the sandbox runtime's `/download/<path>`
surfaces as `k8s_agent_sandbox.exceptions.SandboxRequestError` (the SDK
wraps every non-2xx response in this, `.status_code` set from the real
HTTP status — confirmed by reading `async_connector.py`'s `send_request()`
directly, not assumed). **Not** `httpx.HTTPStatusError` — an earlier version of this docstring
claimed that, which was never actually true for `AsyncSandboxClient`, and
left `present_file.py`/`chat.py`'s `/sandbox-file` endpoint catching the
wrong exception type from the moment the cutover went live until this got
read carefully (2026-09-15) — no one had hit a missing file yet. Both are
fixed to catch `SandboxRequestError` now. Verified against the 404 case
live; the *directory* (400) case still hasn't been exercised through this
path.
"""

import asyncio
import contextlib
import logging
import re
import shlex
from datetime import datetime, timedelta, timezone

from k8s_agent_sandbox import AsyncSandboxClient
from k8s_agent_sandbox.exceptions import SandboxRequestError
from k8s_agent_sandbox.models import SandboxDirectConnectionConfig

from app.clients import conversation_client
from app.core.config import settings
from app.core.conversation_lock import conversation_locks
from app.core.sandbox_coordination import SandboxCoordination, coordination

logger = logging.getLogger(__name__)

_client: AsyncSandboxClient | None = None

_THREAD_LABEL = "jarvis-thread"
# Whose conversation the claim is, so that making room can pause the sandboxes of whoever has been away longest
# rather than just the single idlest conversation. Claims created before this label existed have none, and each of
# those counts as a user of its own.
_USER_LABEL = "jarvis-user"
_CLAIM_READY_TIMEOUT = 180  # generous first-claim cold start; adopting a warm pod is much faster
# In-cluster DNS for the agent-sandbox-system sandbox-router Service.
_ROUTER_URL = "http://sandbox-router-svc.agent-sandbox-system.svc.cluster.local:8080"

# The two CRDs the expiry patches below target — the SDK's helper can get and
# delete these but not update them.
_SANDBOX_API = {"group": "agents.x-k8s.io", "version": "v1beta1", "plural": "sandboxes"}
_CLAIM_API = {
    "group": "extensions.agents.x-k8s.io",
    "version": "v1beta1",
    "plural": "sandboxclaims",
}
# The client offers json-patch and merge-patch and picks the first, so a plain
# dict body is rejected as a malformed JSON Patch operation list without this.
_MERGE_PATCH = "application/merge-patch+json"


def init_client() -> None:
    global _client
    _client = AsyncSandboxClient(
        connection_config=SandboxDirectConnectionConfig(api_url=_ROUTER_URL, server_port=8888),
        # The SDK defaults this to True, registering an atexit hook that
        # deletes every sandbox the process is tracking. That made sense while
        # nothing else reclaimed them, but now the expiry set in
        # _touch_expiry() does — from the cluster, so it also covers sandboxes
        # a previous process created and this one has never heard of. Leaving
        # the hook on would turn every backend restart, including a routine
        # rollout, into data loss for every conversation currently in flight:
        # deleting the claim deletes the Sandbox and its volume, where letting
        # it expire would only have paused it.
        cleanup=False,
    )


async def close_client() -> None:
    if _client is not None:
        await _client.delete_all()
        await _client.close()


def _get_client() -> AsyncSandboxClient:
    if _client is None:
        raise RuntimeError("sandbox client not initialised")
    return _client


def get_thread_id(config) -> str:
    return config.get("configurable", {}).get("thread_id", "default")


def has_sandbox(config) -> bool:
    """Whether this run has a sandbox behind it. False only for a run that sets
    `sandbox: False` in its configurable (the background research task, which
    has no `bash`), so tools that would park a big result in the sandbox know to
    cut it instead."""
    return config.get("configurable", {}).get("sandbox", True)


def normalize_workspace_path(path: str) -> str:
    """Fold the ways the agent refers to a file down to one relative name.

    The model freely writes `report.docx`, `./report.docx` and
    `/workspace/report.docx` for the same file. `/workspace` is not a real
    path on the stock runtime (its working directory is the image's own), but
    the prefix still shows up in model output, so strip it along with a
    leading `./`; leave anything that would escape (a real absolute path, a
    `..`) for the caller's guard to reject.
    """
    name = path.strip()
    for prefix in ("/workspace/", "./"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _label_value(thread_id: str) -> str:
    """A Kubernetes label value must be <=63 chars, matching
    `(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?`. thread_id is a plain
    UUID today (fits as-is) — this still guards against a future thread_id
    shape that wouldn't."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "-", thread_id)[:63]
    return safe.strip("-_.") or "unknown"


class SandboxCapacityError(SandboxRequestError):
    """Every sandbox slot is taken by a conversation that is busy. A SandboxRequestError so the callers that already
    turn those into "sandbox unavailable" (the tools, the upload / download endpoints) need no new branch."""


# Reasons a claim reports once its pod is gone (expired = paused, the volume stays) or finished.
_NO_POD = {"SandboxExpired", "PodSucceeded", "PodFailed"}

# The admission lock, the reservations (conversations admitted whose pod is not yet visible in the cluster) and the
# conversations with a sandbox call in flight are shared between backend replicas: see core/sandbox_coordination.py.
# thread label -> user label, so the conversation service is asked once per conversation, not once per eviction.
_owners: dict[str, str] = {}


@contextlib.asynccontextmanager
async def _using(thread_id: str):
    """A sandbox call is in flight on this conversation: its pod must not be paused until the block ends, on any replica."""
    label = _label_value(thread_id)
    await coordination.mark_in_use(label)
    try:
        yield
    finally:
        await coordination.unmark_in_use(label)


async def _owner_of(label: str) -> str | None:
    """The label of the user who owns conversation `label`, or None when that cannot be told.

    Best-effort on purpose: a sandbox must still be had when the conversation service is down, and an unknown owner
    only means this conversation is judged on its own, as before."""
    if label in _owners:
        return _owners[label]
    try:
        conv = await conversation_client.get_conversation(label)
    except Exception:
        logger.warning("could not look up the owner of conversation %s", label, exc_info=True)
        return None
    if conv is None or not conv.user_id:
        return None
    _owners[label] = _label_value(conv.user_id)
    return _owners[label]


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


async def _list_objects(api: dict, namespace: str) -> list[dict]:
    helper = _get_client().k8s_helper
    result = await helper.custom_objects_api.list_namespaced_custom_object(namespace=namespace, **api)
    return result.get("items", [])


def _has_pod(claim: dict, sandbox: dict | None) -> bool:
    """Whether this conversation's sandbox has a pod (or is getting one).

    The Sandbox's own `shutdownTime` decides, because it is what this process writes: reviving pushes it into the
    future, pausing sets it to now. The claim's Ready condition only follows a few seconds later, and judging by it
    alone counted a sandbox that had just been revived as still paused, so a request arriving in that window was let
    in over the ceiling."""
    conditions = (claim.get("status") or {}).get("conditions") or []
    ready = next((c for c in conditions if c.get("type") == "Ready"), None)
    if ready and ready.get("status") != "True" and ready.get("reason") in {"PodSucceeded", "PodFailed"}:
        return False
    deadline = _parse_time(((sandbox or {}).get("spec") or {}).get("shutdownTime"))
    if deadline is not None:
        return deadline > datetime.now(timezone.utc)
    return not (ready and ready.get("status") != "True" and ready.get("reason") in _NO_POD)


async def _pause_idlest(
    namespace: str,
    label: str,
    claims: list[dict],
    sandboxes: dict[str, dict],
    running: set[str],
    requester: str | None = None,
    only_user: str | None = None,
    coord: SandboxCoordination | None = None,
) -> str | None:
    """Pause a sandbox of the user who has been away longest; the label of the conversation it was, or None.

    A user is as recent as their most recent sandbox: someone using one conversation right now keeps all their other
    sandboxes out of the running, however old those are. Among the users, the one gone longest loses first, and of
    their sandboxes the idlest. So a user cannot hold slots just by having opened many conversations and then
    wandered off, nor lose one to somebody else's older but less abandoned tab. The requester counts as active now.
    `only_user` limits the choice to that user's sandboxes: the per-user limit makes room from the user's own.

    Pausing is what the idle timeout does anyway: the pod goes, the volume (and so the files) stays, and the next
    message revives it. Only a sandbox no run is using qualifies: its conversation has no run going, no call in flight
    here, and it has been quiet for EVICT_MIN_IDLE, so a conversation between two tool calls of one turn keeps its pod.
    """
    busy = {_label_value(t) for t in await conversation_locks.all_running_threads()} | await (coord or coordination).in_use() | {label}
    now = datetime.now(timezone.utc)
    ttl = timedelta(seconds=settings.AGENTSANDBOX_TTL_SECONDS)
    activity: dict[str, datetime] = {requester: now} if requester else {}
    candidates = []
    for claim in claims:
        labels = claim["metadata"].get("labels") or {}
        lbl = labels.get(_THREAD_LABEL)
        if lbl not in running:
            continue
        user = labels.get(_USER_LABEL) or f"thread:{lbl}"
        sandbox = sandboxes.get(((claim.get("status") or {}).get("sandbox") or {}).get("name", ""))
        deadline = _parse_time(((sandbox or {}).get("spec") or {}).get("shutdownTime"))
        if deadline is None or lbl in busy:
            # never used, or in use right now: it cannot be paused, and it shows its user is here
            if lbl in busy:
                activity[user] = now
            continue
        last_use = deadline - ttl
        activity[user] = max(activity.get(user, last_use), last_use)
        if only_user is not None and user != only_user:
            continue
        if (now - last_use).total_seconds() < settings.AGENTSANDBOX_EVICT_MIN_IDLE_SECONDS:
            continue
        candidates.append((user, last_use, sandbox["metadata"]["name"], lbl))
    if not candidates:
        return None
    user, _, name, victim = min(candidates, key=lambda c: (activity[c[0]], c[1]))
    await _get_client().k8s_helper.custom_objects_api.patch_namespaced_custom_object(
        namespace=namespace,
        name=name,
        body={"spec": {"shutdownTime": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "shutdownPolicy": "Retain"}},
        _content_type=_MERGE_PATCH,
        **_SANDBOX_API,
    )
    logger.warning("sandbox ceiling reached: paused a sandbox of the longest-away user (conversation %s) for %s", victim, label)
    return victim


@contextlib.asynccontextmanager
async def _admission(label: str, coord: SandboxCoordination | None = None):
    """Hold a slot for conversation `label` while its pod is being created or revived.

    Two limits, checked in this order. A user may have AGENTSANDBOX_MAX_PODS_PER_USER pods: past it the user's own
    idlest sandbox is paused. And AGENTSANDBOX_MAX_PODS pods may exist at all: past it a sandbox of the user who has
    been away longest is paused (see _pause_idlest). If nothing can be paused, waits (polling) up to
    AGENTSANDBOX_ADMIT_WAIT_SECONDS for something to free up, then raises SandboxCapacityError. Counted from the
    claims in the cluster, so it survives a backend restart; the lock and the reservation only keep concurrent
    requests from sharing one free slot, and they are shared between backend replicas (core/sandbox_coordination.py).
    `coord` is only for tests that stand in for two replicas.
    """
    coord = coord or coordination
    cap = settings.AGENTSANDBOX_MAX_PODS
    user_cap = settings.AGENTSANDBOX_MAX_PODS_PER_USER
    if cap <= 0 and user_cap <= 0:
        yield
        return
    namespace = settings.AGENTSANDBOX_NAMESPACE
    requester = await _owner_of(label)            # outside the lock: it may be a network call
    loop = asyncio.get_running_loop()
    give_up = loop.time() + settings.AGENTSANDBOX_ADMIT_WAIT_SECONDS
    while True:
        why = ""
        async with coord.admit():
            claims = await _list_objects(_CLAIM_API, namespace)
            sandboxes = {s["metadata"]["name"]: s for s in await _list_objects(_SANDBOX_API, namespace)}
            owner_of_label = {
                lbl: (c["metadata"].get("labels") or {}).get(_USER_LABEL)
                for c in claims
                if (lbl := (c["metadata"].get("labels") or {}).get(_THREAD_LABEL))
            }
            running = {
                lbl
                for c in claims
                if (lbl := (c["metadata"].get("labels") or {}).get(_THREAD_LABEL))
                and _has_pod(c, sandboxes.get(((c.get("status") or {}).get("sandbox") or {}).get("name", "")))
            }
            if label in running:
                await coord.reserve(label)
                break
            others = (running | await coord.reserved()) - {label}
            mine = {lbl for lbl in others if (owner_of_label.get(lbl) or _owners.get(lbl)) == requester} if requester else set()
            if user_cap > 0 and len(mine) >= user_cap:
                victim = await _pause_idlest(namespace, label, claims, sandboxes, running, requester, only_user=requester, coord=coord)
                if victim:
                    continue      # its Sandbox deadline is now in the past, so the next count already sees the slot
                why = f"You already have {user_cap} sandboxes in use by other conversations right now."
            elif cap > 0 and len(others) >= cap:
                victim = await _pause_idlest(namespace, label, claims, sandboxes, running, requester, coord=coord)
                if victim:
                    continue
                why = f"All {cap} sandboxes are in use by other conversations right now."
            else:
                await coord.reserve(label)
                break
        if loop.time() >= give_up:
            logger.warning("sandbox limit: no slot for %s after %ss (%s)", label, settings.AGENTSANDBOX_ADMIT_WAIT_SECONDS, why)
            raise SandboxCapacityError(f"{why} Try again in a minute.")
        await asyncio.sleep(2)
    try:
        yield
    finally:
        await coord.unreserve(label)


async def _touch_expiry(claim_name: str, namespace: str, label: str) -> None:
    """Push a claim's sandbox expiry out to now + TTL, reviving it if lapsed.

    Called before every use, which is what turns a fixed deadline into an idle
    timeout — the clock only runs while the conversation is quiet.

    Two deadlines get pushed, on two different objects, because `Retain` means
    something different at each level and only one of them spares the volume:

    - On the **Sandbox** (short): expiry deletes the Pod but `Retain` keeps the
      Sandbox and its volume, so extending it later brings the same filesystem
      back in a couple of seconds. A pause.
    - On the **claim** (long): expiry with `Delete` cascades — claim, Sandbox,
      volume. The only thing that ever reclaims disk, since the pause above
      never does.

    Note `Retain` on the *claim* would not do the same favour: it keeps the
    claim object but still deletes the Sandbox beneath it, taking the volume.
    That asymmetry is also why the SDK's `shutdown_after_seconds` is no use
    here — it writes claim-level lifecycle with `Delete`.

    This has to run *before* `get_sandbox()`, not after: the SDK treats
    `SandboxExpired` as terminal and raises `SandboxNotFoundError` rather than
    handing back an object to revive. The Sandbox's name is read from the
    claim's `status.sandbox.name`, which survives expiry.

    Best-effort — a sandbox that can't be re-armed keeps its old deadline,
    which beats refusing to run the user's command.
    """
    helper = _get_client().k8s_helper
    now = datetime.now(timezone.utc)

    def _at(seconds: int) -> str:
        return (now + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        claim = await helper.get_sandbox_claim(claim_name, namespace)
        status = (claim or {}).get("status") or {}
        name = (status.get("sandbox") or {}).get("name")
        if not name:
            return  # not bound to a sandbox yet; creation will set the deadline

        expired = any(
            c.get("type") == "Ready" and c.get("reason") == "SandboxExpired"
            for c in status.get("conditions") or []
        )
        # Pushing the deadline of an expired sandbox is what brings its pod back, so that is the moment it needs a slot.
        async with _admission(label) if expired else contextlib.nullcontext():
            await helper.custom_objects_api.patch_namespaced_custom_object(
                namespace=namespace,
                name=name,
                body={
                    "spec": {
                        "shutdownTime": _at(settings.AGENTSANDBOX_TTL_SECONDS),
                        "shutdownPolicy": "Retain",
                    }
                },
                _content_type=_MERGE_PATCH,
                **_SANDBOX_API,
            )
            await helper.custom_objects_api.patch_namespaced_custom_object(
                namespace=namespace,
                name=claim_name,
                body={
                    "spec": {
                        "lifecycle": {
                            "shutdownTime": _at(settings.AGENTSANDBOX_MAX_IDLE_SECONDS),
                            "shutdownPolicy": "Delete",
                        }
                    }
                },
                _content_type=_MERGE_PATCH,
                **_CLAIM_API,
            )
            if expired:
                await helper.wait_for_sandbox_ready(name, namespace, _CLAIM_READY_TIMEOUT)
    except SandboxCapacityError:
        raise   # not a hiccup to log and swallow: the caller must not run in a sandbox that is not there
    except Exception:
        logger.warning("could not extend expiry for claim %s", claim_name, exc_info=True)


# One creation at a time per conversation. The model may ask for several bash calls in one step (seen in 27 saved
# steps, 11 of them the first bash step of their conversation); each looks for the conversation's sandbox, finds none,
# and creates one, so the conversation ended up with two or more sandboxes that do not share files. A conversation has
# one run at a time (conversation_locks), so every call for it is in this process and a local lock is enough.
_creating: dict[str, tuple[asyncio.Lock, int]] = {}  # label -> (lock, callers using it); dropped when the last leaves


@contextlib.asynccontextmanager
async def _one_creation_at_a_time(label: str):
    lock, users = _creating.get(label) or (asyncio.Lock(), 0)
    _creating[label] = (lock, users + 1)
    try:
        async with lock:
            yield
    finally:
        lock, users = _creating[label]
        if users <= 1:
            del _creating[label]
        else:
            _creating[label] = (lock, users - 1)


async def _find_sandbox(client, namespace: str, label: str):
    existing = await client.list_all_sandboxes(namespace, label_selector=f"{_THREAD_LABEL}={label}")
    if not existing:
        return None
    await _touch_expiry(existing[0], namespace, label)
    return await client.get_sandbox(existing[0], namespace)


async def _get_or_create_sandbox(thread_id: str):
    client = _get_client()
    namespace = settings.AGENTSANDBOX_NAMESPACE
    label = _label_value(thread_id)
    if (found := await _find_sandbox(client, namespace, label)) is not None:
        return found                      # the usual case: no lock, calls of one conversation share the sandbox freely

    async with _one_creation_at_a_time(label):
        # Looked again inside the lock: a call that was here first has made it by now.
        if (found := await _find_sandbox(client, namespace, label)) is not None:
            return found
        owner = await _owner_of(label)
        async with _admission(label):
            sandbox = await client.create_sandbox(
                warmpool=settings.AGENTSANDBOX_WARMPOOL,
                namespace=namespace,
                sandbox_ready_timeout=_CLAIM_READY_TIMEOUT,
                labels={_THREAD_LABEL: label, **({_USER_LABEL: owner} if owner else {})},
            )
            # claim_name, not sandbox_id — the latter names the Sandbox, which is a
            # different object and would silently find no claim here.
            await _touch_expiry(sandbox.claim_name, namespace, label)
        return sandbox


async def exec_bash(thread_id: str, command: str) -> dict:
    """Returns {stdout, stderr, exit_code, timed_out}; plus `error` when no sandbox could be had at all."""
    async with _using(thread_id):
        try:
            return await _exec_bash(thread_id, command)
        except SandboxCapacityError as e:
            return {"stdout": "", "stderr": str(e), "exit_code": None, "timed_out": True, "error": str(e)}


async def _exec_bash(thread_id: str, command: str) -> dict:
    sandbox = await _get_or_create_sandbox(thread_id)
    # Some sandbox server images run the command via shlex.split() + subprocess
    # directly, not a real shell — confirmed live 2026-09-15 against
    # agent-sandbox's own stock python-runtime-sandbox (its main.py does
    # exactly this): `&&`, `|`, `>`, and heredocs all silently misbehave
    # (e.g. "pwd && ls -la ~" makes pwd receive "-la" as a bogus flag, since
    # shlex.split just tokenizes on whitespace/quotes with zero shell
    # semantics). Wrapping as `bash -c '<command>'` makes that same
    # shlex.split produce exactly ['bash', '-c', '<command>'], so bash itself
    # parses the shell syntax. Harmless against a runtime that already uses a
    # shell — it just adds one nested `bash -c` layer, not a behavior change.
    wrapped = "bash -c " + shlex.quote(command)
    try:
        result = await sandbox.commands.run(wrapped, timeout=300)
    except SandboxRequestError as e:
        # The SDK has no graceful timeout result (see module docstring) —
        # only ever raises. A real connectivity failure looks the same to
        # the caller; bash.py already treats "Error: sandbox unavailable"
        # and "timed out" the same way (both just stop the turn), so
        # collapsing them here doesn't lose anything actionable.
        return {"stdout": "", "stderr": str(e), "exit_code": None, "timed_out": True}
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.exit_code,
        "timed_out": False,
    }


async def read_file(thread_id: str, name: str) -> tuple[bytes, str, str]:
    """Returns (bytes, mime_type, filename).

    The runtime's /download response doesn't report a mime type — falls back
    to guessing from the extension client-side. Fine for present_file's current use (it
    already has the filename from the tool call), worth revisiting if that
    changes.
    """
    import mimetypes

    async with _using(thread_id):
        sandbox = await _get_or_create_sandbox(thread_id)
        content = await sandbox.files.read(normalize_workspace_path(name))
    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    return content, mime, name.rsplit("/", 1)[-1]


async def write_file(thread_id: str, name: str, content: bytes) -> None:
    """Put `content` into the conversation's sandbox as `name`.

    One binary-safe call, unlike sandbox_save.save_and_stub's
    base64-through-`bash -c` path. That one exists to write model-facing
    *text* and has to fit through the command channel in 60k chunks; this
    carries whole files, where the same route would turn a 5MB spreadsheet
    into ~7MB of base64 split across a hundred execs.
    """
    async with _using(thread_id):
        sandbox = await _get_or_create_sandbox(thread_id)
        await sandbox.files.write(normalize_workspace_path(name), content)


async def reset(thread_id: str) -> None:
    """Tear the conversation's sandbox down (deletes its claim + pod)."""
    client = _get_client()
    namespace = settings.AGENTSANDBOX_NAMESPACE
    label = _label_value(thread_id)
    try:
        existing = await client.list_all_sandboxes(
            namespace, label_selector=f"{_THREAD_LABEL}={label}"
        )
        for claim_name in existing:
            await client.delete_sandbox(claim_name, namespace)
    except Exception:
        # Best-effort cleanup — a sandbox hiccup must not fail the stop/delete.
        pass
