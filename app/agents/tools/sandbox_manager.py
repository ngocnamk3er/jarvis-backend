import asyncio
import time
from dataclasses import dataclass, field
from datetime import timedelta

from opensandbox import Sandbox
from opensandbox.config import ConnectionConfig
from opensandbox.exceptions.sandbox import SandboxException
from opensandbox.models.execd import Execution
from opensandbox.models.isolated import (
    CreateIsolatedSessionRequest,
    IsolatedRunOpts,
    IsolatedWorkspaceSpec,
)
from opensandbox.services.isolated import IsolationSession

from app.core.config import settings
from app.db import repository

_WORKSPACE = "/workspace"
_COMMAND_TIMEOUT = timedelta(seconds=300)
_SESSION_IDLE_TIMEOUT_SECONDS = 3600
# One shared pod hosts every conversation's isolation session (OpenSandbox's
# recommended pattern — see docs/guides/isolation-sessions.md). It's created
# with resources sized for many concurrent sessions, not the single-
# conversation default (1 CPU / 2Gi) the old one-pod-per-conversation model
# used.
_HOST_RESOURCE = {"cpu": "2", "memory": "4Gi"}
# Isolation sessions need bwrap to create user/mount namespaces inside the
# sandbox container, which the default pod securityContext blocks (verified
# live: capabilities() reported "Operation not permitted" on both the
# setpriv and userns paths). This extension flag — read directly from the
# installed opensandbox_server package's k8s provider
# (services/k8s/provider_common.py, isolation_enabled branch) — makes the
# server add the SYS_ADMIN capability, Unconfined seccomp/AppArmor, and the
# isolation-upper volume bwrap needs. Without it, host.isolation.create()
# fails with a 503.
_ISOLATION_EXTENSIONS = {"bootstrap.execd.isolation": "enable"}


@dataclass
class _Entry:
    session: IsolationSession
    last_used: float = field(default_factory=time.monotonic)


# Per-conversation isolation-session handles, cached in-process for the
# lifetime of this backend instance. The underlying session_id is also
# persisted to Postgres (Conversation.sandbox_session_id) so it can be
# reattached after a restart — see ensure_session() below.
_sessions: dict[str, _Entry] = {}

_host: Sandbox | None = None
_host_lock = asyncio.Lock()


def get_thread_id(config) -> str:
    return config.get("configurable", {}).get("thread_id", "default")


def _connection_config() -> ConnectionConfig:
    return ConnectionConfig(
        domain=settings.SANDBOX_SERVER_DOMAIN,
        api_key=settings.SANDBOX_API_KEY or None,
        request_timeout=timedelta(seconds=60),
        # This backend can't reach a sandbox pod's ClusterIP directly — only
        # the OpenSandbox server (itself in-cluster) can. use_server_proxy
        # routes all sandbox traffic (health checks, commands, isolation
        # session calls) through the server instead.
        use_server_proxy=True,
    )


async def _ensure_host() -> Sandbox:
    global _host
    if _host is not None:
        return _host

    async with _host_lock:
        if _host is not None:
            return _host

        # timeout=None means "manual cleanup" — the server must NOT auto-kill
        # this pod after a fixed duration like the old per-conversation pods
        # did (timedelta(hours=1)), since every conversation's session lives
        # on it for as long as the backend process runs.
        host = await Sandbox.create(
            settings.SANDBOX_IMAGE,
            connection_config=_connection_config(),
            timeout=None,
            resource=_HOST_RESOURCE,
            extensions=_ISOLATION_EXTENSIONS,
        )
        await host.commands.run(f"mkdir -p {_WORKSPACE}")
        _host = host
        return host


async def ensure_session(thread_id: str) -> IsolationSession:
    entry = _sessions.get(thread_id)
    if entry is not None:
        entry.last_used = time.monotonic()
        return entry.session

    host = await _ensure_host()
    workspace_path = f"{_WORKSPACE}/{thread_id}"

    conversation = await repository.get_conversation(thread_id)
    saved_session_id = conversation.sandbox_session_id if conversation else None

    session: IsolationSession | None = None
    if saved_session_id:
        try:
            session = await host.isolation.attach(saved_session_id)
        except SandboxException:
            # Session expired/destroyed server-side (idle timeout, host pod
            # recreated, etc.) — fall through and create a fresh one.
            session = None

    if session is None:
        await host.commands.run(f"mkdir -p {workspace_path}")
        session = await host.isolation.create(
            CreateIsolatedSessionRequest(
                workspace=IsolatedWorkspaceSpec(path=workspace_path, mode="rw"),
                idle_timeout_seconds=_SESSION_IDLE_TIMEOUT_SECONDS,
            )
        )
        await repository.set_sandbox_session_id(thread_id, session.session_id)

    _sessions[thread_id] = _Entry(session=session)
    return session


async def exec_bash_in_sandbox(thread_id: str, command: str) -> Execution:
    session = await ensure_session(thread_id)
    return await session.run(
        command,
        opts=IsolatedRunOpts(timeout_seconds=int(_COMMAND_TIMEOUT.total_seconds())),
    )


async def stop_sandbox(thread_id: str) -> None:
    entry = _sessions.pop(thread_id, None)
    session = entry.session if entry is not None else None

    if session is None:
        # No live handle in this process (e.g. after a restart) — best-effort
        # reattach-then-delete using the persisted ID so cleanup still works.
        conversation = await repository.get_conversation(thread_id)
        saved_session_id = conversation.sandbox_session_id if conversation else None
        if saved_session_id is None:
            return
        try:
            host = await _ensure_host()
            session = await host.isolation.attach(saved_session_id)
        except SandboxException:
            session = None

    if session is not None:
        try:
            await session.delete()
        except Exception:
            pass

    await repository.set_sandbox_session_id(thread_id, None)


async def cleanup_expired_sandboxes(ttl_minutes: int = 30) -> int:
    """Evict sessions idle longer than ttl_minutes from this process's cache.
    idle_timeout_seconds set at session creation is a server-side backstop
    for the same purpose in case this process dies before its next sweep."""
    cutoff = time.monotonic() - ttl_minutes * 60
    expired = [tid for tid, entry in _sessions.items() if entry.last_used < cutoff]
    for thread_id in expired:
        await stop_sandbox(thread_id)
    return len(expired)
