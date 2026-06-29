import mimetypes
import subprocess
from pathlib import Path
from uuid import uuid4

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from app.core.config import settings
from app.agents.messages import RunPythonMsg
from app.agents.tools.sandbox_manager import (
    ensure_container_running,
    exec_in_sandbox,
    mask_real_paths,
)
from app.storage.minio_client import get_minio


def _snapshot_output(thread_id: str) -> set[Path]:
    output_dir = Path(settings.SANDBOX_DATA_DIR) / thread_id / "output"
    if not output_dir.exists():
        return set()
    return {p for p in output_dir.iterdir() if p.is_file()}


def _upload_new_files(thread_id: str, before: set[Path]) -> list[str]:
    output_dir = Path(settings.SANDBOX_DATA_DIR) / thread_id / "output"
    if not output_dir.exists():
        return []

    after = {p for p in output_dir.iterdir() if p.is_file()}
    new_files = sorted(after - before, key=lambda p: p.name)

    links = []
    for path in new_files:
        object_name = f"{uuid4().hex[:8]}_{path.name}"
        mime_type, _ = mimetypes.guess_type(path.name)
        try:
            get_minio().fput_object(
                settings.MINIO_BUCKET,
                object_name,
                str(path),
                content_type=mime_type or "application/octet-stream",
            )
            url = f"{settings.MINIO_PUBLIC_URL}/{settings.MINIO_BUCKET}/{object_name}"
            links.append(f"[⬇ Download {path.name}]({url})")
        except Exception:
            pass

    return links


@tool
def run_python(code: str, config: RunnableConfig) -> str:
    """Execute Python code in the conversation's sandbox and return stdout.

    Three persistent directories are available:
    - /workspace  : working directory, files here survive across all calls in this conversation
    - /output     : save files here to show them to the user (new files are auto-uploaded)
    - /upload     : user-uploaded files available for reading

    Use print() to produce output.

    Args:
        code: Python code to execute.
    """
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    before = _snapshot_output(thread_id)

    try:
        ensure_container_running()
        result = exec_in_sandbox(thread_id, code)
    except FileNotFoundError:
        return RunPythonMsg.DOCKER_NOT_AVAILABLE
    except subprocess.TimeoutExpired:
        return RunPythonMsg.TIMEOUT.format(timeout=300)

    if result.returncode != 0:
        stderr = mask_real_paths(result.stderr.strip(), thread_id)
        stdout = mask_real_paths(result.stdout.strip(), thread_id)
        if result.returncode == 137 and not stderr:
            hint = "Process was killed (OOM) — likely exceeded memory limit. Use smaller data or batches."
            return f"Error (OOM kill):\n{hint}" + (f"\n\nPartial output:\n{stdout}" if stdout else "")
        msg = stderr or f"Process exited with code {result.returncode}"
        return RunPythonMsg.EXEC_ERROR.format(stderr=msg) + (f"\n\nPartial output:\n{stdout}" if stdout else "")

    stdout = mask_real_paths(result.stdout.strip(), thread_id)
    links = _upload_new_files(thread_id, before)

    parts = []
    if stdout:
        parts.append(stdout)
    if links:
        parts.append("**Files saved:**\n" + "\n".join(links))
    return "\n\n".join(parts) if parts else RunPythonMsg.NO_OUTPUT
