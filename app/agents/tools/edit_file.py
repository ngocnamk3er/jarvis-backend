from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from app.agents.tools.sandbox_manager import get_thread_id, resolve_virtual_path


@tool
def edit_file(path: str, old_string: str, new_string: str, config: RunnableConfig) -> str:
    """Replace an exact string in a file. Fails if old_string is not found or is ambiguous.

    Supports virtual paths: /workspace, /output

    Examples:
        edit_file("/workspace/app.py", "def hello():", "def hello(name):")

    Args:
        path: Virtual path to the file.
        old_string: Exact string to find (must appear exactly once).
        new_string: String to replace it with.
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

    count = content.count(old_string)
    if count == 0:
        return f"Error: string not found in {path}"
    if count > 1:
        return f"Error: string appears {count} times in {path} — provide more context to make it unique"

    try:
        real.write_text(content.replace(old_string, new_string, 1))
    except Exception as e:
        return f"Error writing file: {e}"

    return f"✓ Edit applied to {path}"
