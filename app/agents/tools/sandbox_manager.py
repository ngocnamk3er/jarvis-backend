import re
import subprocess
from pathlib import Path

from app.core.config import settings

_DOCKER_IMAGE = "jarvis-sandbox"
_PIP_CACHE_VOLUME = "jarvis-pip-cache"
_CONTAINER_NAME = "jarvis-sandbox"
_SANDBOX_MOUNT = "/sandbox"

# Virtual paths the agent sees — never exposed as real paths
VIRTUAL_WORKSPACE = "/workspace"
VIRTUAL_OUTPUT = "/output"
VIRTUAL_UPLOAD = "/upload"

# Match virtual paths only when they appear as a root path expression
# (preceded by quote, whitespace, (, =, [, comma) to avoid replacing subpaths like /workspace/output
_VIRTUAL_PATH_RE = re.compile(
    r'(?:^|(?<=["\'\s(=,\[]))(/workspace|/output|/upload)\b'
)


def _real_base(conversation_id: str) -> str:
    return f"{_SANDBOX_MOUNT}/{conversation_id}"


def replace_virtual_paths(code: str, conversation_id: str) -> str:
    """Replace /output, /workspace, /upload with real conv-scoped paths before execution."""
    base = _real_base(conversation_id)

    def _sub(m: re.Match) -> str:
        prefix = m.group(0)[: m.start(1) - m.start()]  # leading delimiter if any
        virtual = m.group(1)
        name = virtual.lstrip("/")
        return prefix + f"{base}/{name}"

    return _VIRTUAL_PATH_RE.sub(_sub, code)


def mask_real_paths(text: str, conversation_id: str) -> str:
    """Replace real conv-scoped paths with virtual paths in output so agent never sees conv_id."""
    base = _real_base(conversation_id)
    text = text.replace(f"{base}/workspace", VIRTUAL_WORKSPACE)
    text = text.replace(f"{base}/output", VIRTUAL_OUTPUT)
    text = text.replace(f"{base}/upload", VIRTUAL_UPLOAD)
    return text


def _prepare_host_dirs(conversation_id: str) -> None:
    base = Path(settings.SANDBOX_DATA_DIR) / conversation_id
    for folder in ("workspace", "output", "upload"):
        (base / folder).mkdir(parents=True, exist_ok=True)


def ensure_container_running() -> str:
    inspect = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}}", _CONTAINER_NAME],
        capture_output=True,
        text=True,
    )
    if inspect.returncode == 0:
        status = inspect.stdout.strip()
        if status == "running":
            return _CONTAINER_NAME
        subprocess.run(["docker", "rm", "-f", _CONTAINER_NAME], capture_output=True)

    try:
        subprocess.run(
            [
                "docker", "run", "-d",
                "--name", _CONTAINER_NAME,
                "--memory", "512m",
                "--cpus", "1.0",
                "--security-opt", "no-new-privileges:true",
                "--tmpfs", "/tmp:size=4g,exec",
                "-v", f"{_PIP_CACHE_VOLUME}:/root/.cache/pip",
                "-v", f"{settings.SANDBOX_DATA_DIR}:{_SANDBOX_MOUNT}",
                _DOCKER_IMAGE,
                "sleep", "infinity",
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        check = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", _CONTAINER_NAME],
            capture_output=True,
            text=True,
        )
        if check.stdout.strip() != "running":
            raise

    return _CONTAINER_NAME


def exec_in_sandbox(conversation_id: str, code: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """Execute code in the shared sandbox container with virtual path mapping.

    The agent writes /output, /workspace, /upload — we replace them with the real
    conversation-scoped paths before exec, then mask them back in the output.
    """
    _prepare_host_dirs(conversation_id)
    real_code = replace_virtual_paths(code, conversation_id)
    workspace_dir = f"{_SANDBOX_MOUNT}/{conversation_id}/workspace"

    return subprocess.run(
        [
            "docker", "exec",
            "-w", workspace_dir,
            "-i", _CONTAINER_NAME,
            "python", "-",
        ],
        input=real_code,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
