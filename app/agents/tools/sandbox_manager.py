"""Client for kubernetes-sigs/agent-sandbox — talks to agent-sandbox's
controller (CRDs `SandboxClaim`/`Sandbox` in `settings.AGENTSANDBOX_NAMESPACE`)
to give each conversation its own sandbox pod, built from jarvis-sandbox's
own toolchain image via `Dockerfile.agentsandbox` (see jarvis-sandbox repo's
AGENTSANDBOX-MIGRATION.md for the full history — Phase 1 install, Phase 2
build-out, and the 2026-09-14 cutover from jarvis-sandbox's own
orchestrator, which this replaced and which no longer runs anywhere).

Written against `k8s-agent-sandbox`'s *actual* installed behavior, verified
live against a real cluster while building this — not the docs site, which
is stale/inconsistent in several places (see the migration doc).

Design decisions, and why — each verified live, not just read off docs:

- **`SandboxDirectConnectionConfig` through the router, not
  `SandboxInClusterConnectionConfig` — reversed 2026-09-15, on purpose,
  for security, not convenience.** An earlier pass here used
  `SandboxInClusterConnectionConfig` (client resolves the pod IP itself,
  bypassing the router) because the *deployed* router image
  (`sandbox-router:latest-main`, a perpetually-rebuilt staging tag) turned
  out to be an old Python build with no pod-IP cache — router-mode 502'd
  every time. That was true, but going pod-direct has a much bigger cost
  than the 502s it avoided: **agent-sandbox's authorization model lives
  entirely in the router** (`authz.Authorizer`, see sandbox-router's own
  README) — individual sandbox pods (`agentsandbox_server.py`, and
  agent-sandbox's own stock `python-runtime-sandbox`) are deliberately
  unauthenticated, trusting NetworkPolicy + the router to be the only
  thing that can reach them. Bypassing the router bypasses that
  authorization entirely. Confirmed live 2026-09-15: from inside one
  sandbox pod, `curl`ing another live sandbox pod's IP directly on
  :8888 — no token, no claim, nothing — successfully read its files
  *and* ran arbitrary commands in it. Any conversation could reach any
  other conversation's sandbox, full read/write/exec, by just knowing
  (or scanning for) its pod IP. This is *worse* than the router's 502s,
  not better. Fix: build the router from the `v1.0.2` source tag instead
  (the Go rewrite has the pod-IP cache the staging image lacks) and route
  through it with `--authz-mode=tokenreview`, using jarvis-backend's own
  ServiceAccount token as the bearer credential (see `_auth_headers()`
  below) — the same RBAC already applied for direct k8s-API access
  (`sandboxclaims`/`sandboxes`) doubles as router authentication, no new
  secret to provision.

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
(path is a directory) from `agentsandbox_server.py`'s `/download/<path>`
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

import re
import shlex

from k8s_agent_sandbox import AsyncSandboxClient
from k8s_agent_sandbox.exceptions import SandboxRequestError
from k8s_agent_sandbox.models import SandboxDirectConnectionConfig

from app.core.config import settings

_client: AsyncSandboxClient | None = None

_THREAD_LABEL = "jarvis-thread"
_CLAIM_READY_TIMEOUT = 180  # generous first-claim cold start; adopting a warm pod is much faster
# In-cluster DNS for the agent-sandbox-system sandbox-router Service.
_ROUTER_URL = "http://sandbox-router-svc.agent-sandbox-system.svc.cluster.local:8080"
# The standard path every pod gets its ServiceAccount token projected to —
# same identity the RBAC for sandboxclaims/sandboxes was granted to (see
# AGENTSANDBOX-MIGRATION.md). Read fresh on every call, not cached: this is
# a *projected* token (kubelet auto-rotates it, ~1h by default per
# ServiceAccountTokenExpiration), so a value cached at startup would
# eventually be rejected by the router's TokenReview check.
_SA_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"


def _auth_headers() -> dict[str, str]:
    try:
        with open(_SA_TOKEN_PATH) as f:
            token = f.read().strip()
    except OSError:
        # Local dev / no ServiceAccount mounted — let the router's own
        # --authz-tokenreview-require-token setting decide whether a
        # tokenless request is allowed; don't crash the whole call over it.
        return {}
    return {"Authorization": f"Bearer {token}"}


def init_client() -> None:
    global _client
    _client = AsyncSandboxClient(
        connection_config=SandboxDirectConnectionConfig(api_url=_ROUTER_URL, server_port=8888),
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


def normalize_workspace_path(path: str) -> str:
    """Fold the ways the agent refers to a file in its workspace down to one
    relative name.

    `/workspace` is the working directory, so the model freely writes
    `report.docx`, `./report.docx` and `/workspace/report.docx` for the same
    file (the bash tool tells it absolute-under-/workspace is fine). Strip the
    workspace prefix / leading `./`; leave anything that would escape (a real
    absolute path, a `..`) for the caller's guard to reject.
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


async def _get_or_create_sandbox(thread_id: str):
    client = _get_client()
    namespace = settings.AGENTSANDBOX_NAMESPACE
    label = _label_value(thread_id)
    existing = await client.list_all_sandboxes(namespace, label_selector=f"{_THREAD_LABEL}={label}")
    if existing:
        sandbox = await client.get_sandbox(existing[0], namespace)
    else:
        sandbox = await client.create_sandbox(
            warmpool=settings.AGENTSANDBOX_WARMPOOL,
            namespace=namespace,
            sandbox_ready_timeout=_CLAIM_READY_TIMEOUT,
            labels={_THREAD_LABEL: label},
        )
    # Claim/lookup above talks to the k8s API directly (create_sandbox,
    # list_all_sandboxes, get_sandbox) — this only covers the HTTP path
    # (commands.run / files.read/write) that actually goes to the router.
    # No first-class SDK API for this (SandboxDirectConnectionConfig has no
    # headers field) — sandbox.connector is a plain instance attribute
    # (commands and files share the one connector, see async_sandbox.py),
    # so this reaches in and sets it directly.
    sandbox.connector.client.headers.update(_auth_headers())
    return sandbox


async def exec_bash(thread_id: str, command: str) -> dict:
    """Returns {stdout, stderr, exit_code, timed_out}."""
    sandbox = await _get_or_create_sandbox(thread_id)
    # Some sandbox server images run the command via shlex.split() + subprocess
    # directly, not a real shell — confirmed live 2026-09-15 against
    # agent-sandbox's own stock python-runtime-sandbox (its main.py does
    # exactly this): `&&`, `|`, `>`, and heredocs all silently misbehave
    # (e.g. "pwd && ls -la ~" makes pwd receive "-la" as a bogus flag, since
    # shlex.split just tokenizes on whitespace/quotes with zero shell
    # semantics). Wrapping as `bash -c '<command>'` makes that same
    # shlex.split produce exactly ['bash', '-c', '<command>'], so bash itself
    # parses the shell syntax. Harmless on jarvis's own agentsandbox_server.py,
    # which already runs `bash -c <command>` server-side — this just adds one
    # extra (nested) bash -c layer there, not a behavior change.
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

    Mime type isn't reported by agentsandbox_server.py's /download response
    headers the way the old orchestrator's did — falls back to guessing from
    the extension client-side. Fine for present_file's current use (it
    already has the filename from the tool call), worth revisiting if that
    changes.
    """
    import mimetypes

    sandbox = await _get_or_create_sandbox(thread_id)
    content = await sandbox.files.read(normalize_workspace_path(name))
    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    return content, mime, name.rsplit("/", 1)[-1]


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
