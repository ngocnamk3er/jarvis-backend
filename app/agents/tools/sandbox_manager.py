"""Dispatches every sandbox call to whichever backend `settings.SANDBOX_BACKEND`
names — `"legacy"` (jarvis-sandbox's own orchestrator, default) or
`"agentsandbox"` (kubernetes-sigs/agent-sandbox, see
`sandbox_manager_agentsandbox.py` / AGENTSANDBOX-MIGRATION.md in jarvis-sandbox).

Every call site (bash.py, present_file.py, files.py, web_search.py,
web_fetch.py, sandbox_save.py, chat_service.py, conversation_service.py,
chat.py, main.py) imports from here and never needs to know which backend is
live. Flipping `SANDBOX_BACKEND` — a ConfigMap value, no image rebuild — is
the entire cutover switch, and flipping it back is the entire rollback.

The `agentsandbox` module is imported lazily, only when actually selected —
it hard-requires the optional `k8s-agent-sandbox` package (raises ImportError
at import time if missing, see that module's docstring) and talks to CRDs
jarvis-backend doesn't have RBAC for by default. With the default
`SANDBOX_BACKEND=legacy`, none of that is ever touched, so a deploy that
hasn't picked up the new dependency yet still starts up and runs exactly as
before this dispatcher existed.
"""

from app.agents.tools import sandbox_manager_legacy as _legacy
from app.core.config import settings


def _backend():
    if settings.SANDBOX_BACKEND == "agentsandbox":
        from app.agents.tools import sandbox_manager_agentsandbox as _agentsandbox

        return _agentsandbox
    return _legacy


def init_client() -> None:
    _backend().init_client()


async def close_client() -> None:
    await _backend().close_client()


def get_thread_id(config) -> str:
    return _backend().get_thread_id(config)


def normalize_workspace_path(path: str) -> str:
    return _backend().normalize_workspace_path(path)


async def exec_bash(thread_id: str, command: str) -> dict:
    """Returns {stdout, stderr, exit_code, timed_out}."""
    return await _backend().exec_bash(thread_id, command)


async def read_file(thread_id: str, name: str) -> tuple[bytes, str, str]:
    """Returns (bytes, mime_type, filename)."""
    return await _backend().read_file(thread_id, name)


async def reset(thread_id: str) -> None:
    await _backend().reset(thread_id)
