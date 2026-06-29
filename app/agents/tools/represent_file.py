import hashlib
import mimetypes
import shutil
from pathlib import Path

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from app.core.config import settings
from app.storage.minio_client import get_minio
from app.agents.tools.sandbox_manager import (
    VIRTUAL_WORKSPACE,
    VIRTUAL_OUTPUT,
    VIRTUAL_UPLOAD,
)


def _resolve_virtual_path(virtual_path: str, thread_id: str) -> Path:
    """Translate a virtual path (/workspace, /output, /upload) to a real host path."""
    base = Path(settings.SANDBOX_DATA_DIR) / thread_id
    vp = virtual_path.strip()
    if vp.startswith(VIRTUAL_WORKSPACE):
        return base / "workspace" / vp[len(VIRTUAL_WORKSPACE):].lstrip("/")
    if vp.startswith(VIRTUAL_OUTPUT):
        return base / "output" / vp[len(VIRTUAL_OUTPUT):].lstrip("/")
    if vp.startswith(VIRTUAL_UPLOAD):
        return base / "upload" / vp[len(VIRTUAL_UPLOAD):].lstrip("/")
    # Fallback: treat as relative to workspace
    return base / "workspace" / vp.lstrip("/")


@tool
def represent_file(path: str, config: RunnableConfig) -> str:
    """Expose a file from the sandbox so the user can download or preview it.

    Use this when you have already created a file in /workspace or /output
    and want to make it available to the user.

    Supported virtual paths:
    - /workspace/<filename>  — file in the working directory
    - /output/<filename>     — file already in the output directory
    - /upload/<filename>     — user-uploaded file

    Examples:
        represent_file("/workspace/model.pth")
        represent_file("/output/chart.png")

    Args:
        path: Virtual path to the file inside the sandbox.
    """
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    src = _resolve_virtual_path(path, thread_id)

    if not src.exists():
        return f"Error: file not found at {path}"
    if not src.is_file():
        return f"Error: {path} is a directory, not a file"

    # Copy to /output if not already there
    output_dir = Path(settings.SANDBOX_DATA_DIR) / thread_id / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / src.name
    if src != dest:
        shutil.copy2(src, dest)

    sha = hashlib.sha256(dest.read_bytes()).hexdigest()[:16]
    object_name = f"{sha}_{src.name}"
    mime_type, _ = mimetypes.guess_type(src.name)
    try:
        get_minio().fput_object(
            settings.MINIO_BUCKET,
            object_name,
            str(dest),
            content_type=mime_type or "application/octet-stream",
        )
    except Exception as e:
        return f"Error uploading to storage: {e}"

    url = f"{settings.MINIO_PUBLIC_URL}/{settings.MINIO_BUCKET}/{object_name}"
    return f"[⬇ Download {src.name}]({url})"
