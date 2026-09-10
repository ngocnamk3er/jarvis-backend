"""HTTP client for jarvis-sandbox.

We talk to the sandbox *orchestrator* (Service `sandbox`, internal-only behind
X-Internal-Api-Key). The API is unchanged — `/sandbox/exec`, `/sandbox/read`,
`/sandbox/reset`, all keyed by thread_id — but under the hood each conversation
now gets its own dedicated agent pod from a warm pool, not a shared container.
Isolation is the k8s pod boundary (own namespaces, non-root, dropped caps,
seccomp, a NetworkPolicy that blocks the rest of the cluster); the pod is
deleted on reset / idle GC. Still no session state for us to track.

Replaced OpenSandbox, which broke on this host: its nested bwrap/userns
isolation stopped working once the kernel set
apparmor_restrict_unprivileged_userns=1 (Ubuntu 24.04+ default).
"""
import httpx

from app.core.config import settings

_client: httpx.AsyncClient | None = None


def init_client() -> None:
    global _client
    _client = httpx.AsyncClient(
        base_url=f"{settings.SANDBOX_SERVICE_URL}{settings.API_PREFIX}",
        headers={"X-Internal-Api-Key": settings.INTERNAL_API_KEY},
        # Longer than the 300s command timeout so the server's own timeout
        # (which returns a clean {timed_out: true}) always wins the race.
        # The extra headroom also covers the first call of a conversation,
        # when the orchestrator may spin up a fresh agent pod.
        timeout=360.0,
    )


async def close_client() -> None:
    if _client is not None:
        await _client.aclose()


def _get_client() -> httpx.AsyncClient:
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


async def exec_bash(thread_id: str, command: str) -> dict:
    """Returns {stdout, stderr, exit_code, timed_out}."""
    resp = await _get_client().post(
        "/sandbox/exec", json={"thread_id": thread_id, "command": command}
    )
    resp.raise_for_status()
    return resp.json()


async def read_file(thread_id: str, name: str) -> tuple[bytes, str, str]:
    """Returns (bytes, mime_type, filename). Raises httpx.HTTPStatusError on
    404 (not found) / 400 (dir, too large, path escape)."""
    resp = await _get_client().get("/sandbox/read", params={"thread_id": thread_id, "name": name})
    resp.raise_for_status()
    mime = resp.headers.get("content-type", "application/octet-stream")
    disposition = resp.headers.get("content-disposition", "")
    filename = name
    if 'filename="' in disposition:
        filename = disposition.split('filename="', 1)[1].split('"', 1)[0]
    return resp.content, mime, filename


async def reset(thread_id: str) -> None:
    """Tear the conversation's sandbox down (deletes its pod) — on /chat/stop
    and conversation delete."""
    try:
        resp = await _get_client().post("/sandbox/reset", json={"thread_id": thread_id})
        resp.raise_for_status()
    except httpx.HTTPError:
        # Best-effort cleanup — a sandbox hiccup must not fail the stop/delete.
        pass
