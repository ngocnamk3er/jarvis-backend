from pathlib import Path

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from app.core.config import settings
from app.agents.tools.sandbox_manager import get_thread_id


@tool
def make_file(filename: str, content: str, config: RunnableConfig) -> str:
    """Write text content directly to /output and return a download link.

    Use this instead of write_file + represent_file when you have text content
    ready to save (CSV, JSON, TXT, MD, HTML, Python, SQL, YAML, etc.).
    For binary files (DOCX, XLSX, PNG, PDF) that require a Python script to
    generate, use bash with os.environ["OUTPUT"] to get the real output path.

    Examples:
        make_file("report.md", "# Report\\n...")
        make_file("data.csv", "name,age\\nAlice,30\\n")
        make_file("query.sql", "SELECT * FROM users;")

    Args:
        filename: Name of the file (no path prefix needed).
        content: Text content to write.
    """
    thread_id = get_thread_id(config)
    safe_name = Path(filename).name

    output_dir = Path(settings.SANDBOX_DATA_DIR) / thread_id / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / safe_name

    try:
        dest.write_text(content, encoding="utf-8")
    except Exception as e:
        return f"Error writing file: {e}"

    url = f"{settings.BACKEND_URL}/api/v1/files/{thread_id}/{safe_name}"
    return f"[⬇ Download {safe_name}]({url})"
