import shutil
from pathlib import Path

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from app.core.config import settings
from app.agents.tools.sandbox_manager import get_thread_id, resolve_virtual_path


@tool
def represent_file(path: str, label: str, config: RunnableConfig) -> str:
    """Expose a file from the sandbox so the user can download or preview it.

    Use this when you have already created a file in /workspace or /output
    and want to make it available to the user.

    Supported virtual paths: /workspace, /output, /upload

    Examples:
        represent_file("/workspace/model.pth")
        represent_file("/output/chart.png")

    Args:
        path: Virtual path to the file inside the sandbox.
        label: Brief human-readable description shown to the user (e.g. "Exporting chart.png").
    """
    thread_id = get_thread_id(config)
    src = resolve_virtual_path(path, thread_id)

    if not src.exists():
        return f"Error: file not found at {path}"
    if not src.is_file():
        return f"Error: {path} is a directory, not a file"

    output_dir = Path(settings.SANDBOX_DATA_DIR) / thread_id / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / src.name
    if src != dest:
        shutil.copy2(src, dest)

    url = f"{settings.BACKEND_URL}/api/v1/files/{thread_id}/{dest.name}"
    return f"[⬇ Download {src.name}]({url})"
