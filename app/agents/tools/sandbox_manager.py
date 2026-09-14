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

- **`SandboxInClusterConnectionConfig`, not the router.** Tried routing
  through `sandbox-router-svc` instead (`SandboxDirectConnectionConfig`) —
  reasonable-looking on paper, since every sandbox this module touches
  *is* claimed through `create_sandbox()`, not addressed by a guessed
  name. Verified live anyway, from a real jarvis-backend pod, with a
  genuinely SDK-claimed sandbox and the exact headers the SDK sends
  (`X-Sandbox-ID`/`-Namespace`/`-Port`) — still 502'd:
  `{"detail":"Could not connect to the backend sandbox: <pod>"}`. Root
  cause, found by reading the *actual running* router's logs, not just
  its source: the deployed image (`sandbox-router:latest-main`, a
  perpetually-rebuilt staging tag) turned out to be an old Python build
  that falls straight to `<id>.<namespace>.svc.cluster.local` DNS — which
  never resolves (no per-Sandbox Service exists) — with no informer-backed
  pod-IP cache at all. The *source* at tag `v1.0.2` (matching the
  installed controller) is a Go rewrite that does have that cache and
  would very likely make router-mode work — but that image isn't what's
  deployed. Revisit router mode once that's actually running; until then
  `SandboxInClusterConnectionConfig` (client resolves the pod IP itself
  from the Sandbox's own status, bypassing the router entirely) is the
  one confirmed to work.

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

Known gap, not yet resolved: `agentsandbox_server.py`'s `/download/<path>`
returns a real 404 (via `FileNotFoundError`) for a missing file, which
`httpx`/`requests` surfaces as an `HTTPStatusError` — `read_file()` below
re-raises that, matching this module's documented contract ("Raises
httpx.HTTPStatusError on 404"). Not yet verified against the *directory*
(400) case through this path — do that before relying on it for
`present_file`'s existing error handling.
"""

import re

from k8s_agent_sandbox import AsyncSandboxClient
from k8s_agent_sandbox.exceptions import SandboxRequestError
from k8s_agent_sandbox.models import SandboxInClusterConnectionConfig

from app.core.config import settings

_client: AsyncSandboxClient | None = None

_THREAD_LABEL = "jarvis-thread"
_CLAIM_READY_TIMEOUT = 180  # generous first-claim cold start; adopting a warm pod is much faster


def init_client() -> None:
    global _client
    _client = AsyncSandboxClient(
        connection_config=SandboxInClusterConnectionConfig(server_port=8888),
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
        return await client.get_sandbox(existing[0], namespace)
    return await client.create_sandbox(
        warmpool=settings.AGENTSANDBOX_WARMPOOL,
        namespace=namespace,
        sandbox_ready_timeout=_CLAIM_READY_TIMEOUT,
        labels={_THREAD_LABEL: label},
    )


async def exec_bash(thread_id: str, command: str) -> dict:
    """Returns {stdout, stderr, exit_code, timed_out}."""
    sandbox = await _get_or_create_sandbox(thread_id)
    try:
        result = await sandbox.commands.run(command, timeout=300)
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
