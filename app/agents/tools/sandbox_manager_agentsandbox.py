"""HTTP-equivalent client for kubernetes-sigs/agent-sandbox — an alternate
implementation of sandbox_manager.py's public surface (exec_bash, read_file,
reset, get_thread_id, normalize_workspace_path), talking to agent-sandbox's
controller instead of jarvis-sandbox's own orchestrator.

Not wired into any tool yet — see AGENTSANDBOX-MIGRATION.md (jarvis-sandbox
repo) Phase 2 step B/F for the swap-in plan and everything this still needs
before that (RBAC, NetworkPolicy, the read_file 404-vs-empty-dir gap noted
below). Written against `k8s-agent-sandbox`'s *actual* installed behavior,
verified live against a real cluster while building this — not the docs
site, which is stale/inconsistent in several places (see the migration doc).

Design decisions, and why — each verified live, not just read off docs:

- **`SandboxInClusterConnectionConfig`, not Tunnel or a hand-rolled router
  call.** Tunnel mode shells out to `kubectl port-forward` per sandbox —
  fine for a laptop, wrong for a long-running backend service (extra
  process per sandbox, needs `kubectl` + a kubeconfig baked into the
  image). Hand-rolling direct HTTP calls to sandbox-router with
  `X-Sandbox-ID`/`X-Sandbox-Namespace` headers (the obvious-looking
  alternative) was tried and verified NOT to work for a sandbox that was
  never claimed through the SDK: the router could route to a sandbox
  claimed via `create_sandbox()` (proxied straight to its pod IP) but
  returned 502 for the exact same pod addressed by name directly — it
  tried `<name>.<namespace>.svc.cluster.local`, which doesn't resolve
  (agent-sandbox doesn't create a per-Sandbox Service). Conclusion: the
  router's routing table is populated by the *claim* lifecycle, not by a
  Sandbox merely existing — so claiming has to go through the real client,
  which then makes `SandboxInClusterConnectionConfig` (client resolves the
  pod IP itself, from the Sandbox's own status, bypassing the router
  entirely) the natural production-grade choice, not a hack.

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
  field.** The current jarvis-sandbox agent catches its own timeout
  server-side and returns `{"timed_out": true, ...}` gracefully.
  `k8s_agent_sandbox`'s `CommandExecutor.run(command, timeout)` has no
  such thing — `timeout` is just the HTTP client's request timeout, and
  blowing it raises (wrapped as `SandboxRequestError`). Caught here and
  translated to the same `{"timed_out": True, ...}` shape `bash.py`
  already expects, so nothing above this module needs to change.

Known gap, not yet resolved: `agentsandbox_server.py`'s `/download/<path>`
returns a real 404 (via `FileNotFoundError`) for a missing file, which
`httpx`/`requests` surfaces as an `HTTPStatusError` — same shape
`read_file()` below already re-raises, matching the current module's
documented contract ("Raises httpx.HTTPStatusError on 404"). Not yet
verified against the *directory* (400) case through this path — do that
before relying on it for `present_file`'s existing error handling.
"""

import re

from app.core.config import settings

try:
    from k8s_agent_sandbox import AsyncSandboxClient
    from k8s_agent_sandbox.exceptions import SandboxRequestError
    from k8s_agent_sandbox.models import SandboxInClusterConnectionConfig
except ImportError as e:  # pragma: no cover - only hit if the optional dep isn't installed
    raise ImportError(
        "sandbox_manager_agentsandbox requires the k8s-agent-sandbox[async] "
        "package (pip install 'k8s-agent-sandbox[async]') — not part of "
        "jarvis-backend's regular requirements.txt yet, see "
        "AGENTSANDBOX-MIGRATION.md step B before wiring this in for real."
    ) from e

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


def _get_client() -> AsyncSandboxClient:
    if _client is None:
        raise RuntimeError("sandbox client not initialised")
    return _client


def get_thread_id(config) -> str:
    return config.get("configurable", {}).get("thread_id", "default")


def normalize_workspace_path(path: str) -> str:
    """Same folding rules as sandbox_manager.py — kept identical so
    present_file / the bash tool's docstring don't need to change."""
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
    """Returns {stdout, stderr, exit_code, timed_out} — same shape as
    sandbox_manager.py's exec_bash, so bash.py needs no changes."""
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

    Unlike sandbox_manager.py's version, mime type isn't reported by
    agentsandbox_server.py's /download response headers the same way (no
    Content-Disposition filename echo) — falls back to guessing from the
    extension client-side. Fine for present_file's current use (it already
    has the filename from the tool call), worth revisiting if that changes.
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
        # Best-effort cleanup — a sandbox hiccup must not fail the stop/delete,
        # matching sandbox_manager.py's reset().
        pass
