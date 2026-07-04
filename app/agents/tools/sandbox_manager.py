import re
import subprocess
from pathlib import Path

from app.core.config import settings

_DOCKER_IMAGE = "jarvis-sandbox"
_UV_CACHE_VOLUME = "jarvis-uv-cache"
_SANDBOX_MOUNT = "/sandbox"

# Virtual paths the agent sees — never exposed as real paths
VIRTUAL_WORKSPACE = "/workspace"
VIRTUAL_OUTPUT = "/output"
VIRTUAL_UPLOAD = "/upload"
VIRTUAL_VENV = "/venv"

# Match virtual paths only when they appear as a root path expression
# (preceded by quote, whitespace, (, =, [, comma) to avoid replacing subpaths like /workspace/output
_VIRTUAL_PATH_RE = re.compile(r'(?:^|(?<=["\'\s(=,\[]))(/workspace|/output|/upload)\b')


def _container_name(thread_id: str) -> str:
    return f"jarvis-sandbox-{thread_id}"


def get_thread_id(config) -> str:
    return config.get("configurable", {}).get("thread_id", "default")


def resolve_virtual_path(virtual_path: str, thread_id: str) -> Path:
    """Translate a virtual path (/workspace, /output, /upload) to a real host path."""
    base = Path(settings.SANDBOX_DATA_DIR) / thread_id
    vp = virtual_path.strip()
    if vp.startswith(VIRTUAL_WORKSPACE):
        return base / "workspace" / vp[len(VIRTUAL_WORKSPACE):].lstrip("/")
    if vp.startswith(VIRTUAL_OUTPUT):
        return base / "output" / vp[len(VIRTUAL_OUTPUT):].lstrip("/")
    if vp.startswith(VIRTUAL_UPLOAD):
        return base / "upload" / vp[len(VIRTUAL_UPLOAD):].lstrip("/")
    return base / "workspace" / vp.lstrip("/")


def replace_virtual_paths(code: str, _conversation_id: str) -> str:
    """Replace /output, /workspace, /upload with real container paths before execution."""
    def _sub(m: re.Match) -> str:
        prefix = m.group(0)[: m.start(1) - m.start()]
        virtual = m.group(1)
        name = virtual.lstrip("/")
        return prefix + f"{_SANDBOX_MOUNT}/{name}"

    return _VIRTUAL_PATH_RE.sub(_sub, code)


def mask_real_paths(text: str, _conversation_id: str) -> str:
    """Replace container paths with virtual paths so agent never sees internal structure."""
    text = text.replace(f"{_SANDBOX_MOUNT}/workspace", VIRTUAL_WORKSPACE)
    text = text.replace(f"{_SANDBOX_MOUNT}/output", VIRTUAL_OUTPUT)
    text = text.replace(f"{_SANDBOX_MOUNT}/upload", VIRTUAL_UPLOAD)
    text = text.replace(f"{_SANDBOX_MOUNT}/.venv", VIRTUAL_VENV)
    return text


def _prepare_host_dirs(conversation_id: str) -> None:
    base = Path(settings.SANDBOX_DATA_DIR) / conversation_id
    for folder in ("workspace", "output", "upload"):
        (base / folder).mkdir(parents=True, exist_ok=True)


def ensure_venv(thread_id: str) -> None:
    """Create a per-conversation uv venv if it doesn't exist yet."""
    # pyvenv.cfg is a plain file (not a symlink) — reliable existence check
    pyvenv_cfg = Path(settings.SANDBOX_DATA_DIR) / thread_id / ".venv" / "pyvenv.cfg"
    if pyvenv_cfg.exists():
        return
    try:
        subprocess.run(
            [
                "docker", "exec",
                "-e", "UV_LINK_MODE=copy",
                _container_name(thread_id),
                "uv", "venv", "--seed", "--system-site-packages",
                f"{_SANDBOX_MOUNT}/.venv",
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        # Race condition: another concurrent call may have created the venv just before us
        if not pyvenv_cfg.exists():
            raise


def ensure_container_running(thread_id: str) -> str:
    container = _container_name(thread_id)
    inspect = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}}", container],
        capture_output=True,
        text=True,
    )
    if inspect.returncode == 0:
        status = inspect.stdout.strip()
        if status == "running":
            return container
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)

    host_dir = str(Path(settings.SANDBOX_DATA_DIR) / thread_id)
    try:
        subprocess.run(
            [
                "docker", "run", "-d",
                "--name", container,
                "--memory", "512m",
                "--cpus", "1.0",
                "--security-opt", "no-new-privileges:true",
                "--tmpfs", "/tmp:size=4g,exec",
                "-v", f"{_UV_CACHE_VOLUME}:/root/.cache/uv",
                "-v", f"{host_dir}:{_SANDBOX_MOUNT}",
                _DOCKER_IMAGE,
                "sleep", "infinity",
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        check = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", container],
            capture_output=True,
            text=True,
        )
        if check.stdout.strip() != "running":
            raise

    return container


def exec_bash_in_sandbox(
    conversation_id: str, command: str, timeout: int = 60
) -> subprocess.CompletedProcess:
    """Execute a bash command in the per-conversation container."""
    _prepare_host_dirs(conversation_id)
    ensure_container_running(conversation_id)
    ensure_venv(conversation_id)
    real_command = replace_virtual_paths(command, conversation_id)

    env_prefix = (
        f"export PATH={_SANDBOX_MOUNT}/.venv/bin:$PATH "
        f"VIRTUAL_ENV={_SANDBOX_MOUNT}/.venv "
        f"WORKSPACE={_SANDBOX_MOUNT}/workspace "
        f"OUTPUT={_SANDBOX_MOUNT}/output "
        f"UPLOAD={_SANDBOX_MOUNT}/upload && "
    )

    return subprocess.run(
        [
            "docker", "exec",
            "-w", f"{_SANDBOX_MOUNT}/workspace",
            _container_name(conversation_id),
            "bash", "-c",
            env_prefix + real_command,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def exec_in_sandbox(
    conversation_id: str, code: str, timeout: int = 300
) -> subprocess.CompletedProcess:
    """Execute Python code in the per-conversation container."""
    _prepare_host_dirs(conversation_id)
    ensure_container_running(conversation_id)
    ensure_venv(conversation_id)
    real_code = replace_virtual_paths(code, conversation_id)

    return subprocess.run(
        [
            "docker", "exec",
            "-w", f"{_SANDBOX_MOUNT}/workspace",
            "-i",
            _container_name(conversation_id),
            f"{_SANDBOX_MOUNT}/.venv/bin/python",
            "-",
        ],
        input=real_code,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
