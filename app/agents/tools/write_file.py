from pathlib import Path

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from app.core.config import settings
from app.agents.tools.sandbox_manager import get_thread_id, resolve_virtual_path


@tool
def write_file(path: str, content: str, config: RunnableConfig) -> str:
    """Write content to a file in the sandbox, creating it if it doesn't exist.

    Supports virtual paths: /workspace, /output

    Examples:
        write_file("/workspace/hello.py", "print('hello')")
        write_file("/output/report.txt", "Results:\\n...")

    Args:
        path: Virtual path to the file.
        content: Text content to write.
    """
    thread_id = get_thread_id(config)
    real = resolve_virtual_path(path, thread_id)

    upload_base = Path(settings.SANDBOX_DATA_DIR) / thread_id / "upload"
    try:
        real.relative_to(upload_base)
        return "Error: /upload is read-only."
    except ValueError:
        pass

    try:
        real.parent.mkdir(parents=True, exist_ok=True)
        real.write_text(content)
    except Exception as e:
        return f"Error writing file: {e}"

    return f"✓ Written {len(content)} chars to {path}"
