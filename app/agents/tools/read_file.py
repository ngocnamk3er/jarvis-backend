from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from app.agents.tools.sandbox_manager import get_thread_id, resolve_virtual_path


@tool
def read_file(path: str, config: RunnableConfig) -> str:
    """Read the contents of a file in the sandbox with line numbers.

    Supports virtual paths: /workspace, /output, /upload

    Examples:
        read_file("/workspace/script.py")
        read_file("/upload/data.csv")

    Args:
        path: Virtual path to the file.
    """
    thread_id = get_thread_id(config)
    real = resolve_virtual_path(path, thread_id)

    if not real.exists():
        return f"Error: file not found: {path}"
    if not real.is_file():
        return f"Error: {path} is a directory"

    try:
        content = real.read_text(errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"

    lines = content.splitlines()
    numbered = "\n".join(f"{i+1:4d}\t{line}" for i, line in enumerate(lines))
    return f"{path} ({len(lines)} lines)\n{numbered}"
