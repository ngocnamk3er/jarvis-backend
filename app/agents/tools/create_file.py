import mimetypes
import subprocess
from pathlib import Path
from uuid import uuid4

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from app.core.config import settings
from app.agents.messages import CreateFileMsg
from app.agents.tools.sandbox_manager import (
    ensure_container_running,
    exec_in_sandbox,
    mask_real_paths,
)
from app.storage.minio_client import get_minio


@tool
def create_file(filename: str, code: str, config: RunnableConfig) -> str:
    """Create a file (docx, pdf, png, svg, xlsx, etc.) by running Python code in the sandbox.

    Three persistent directories are available:
    - /workspace  : working directory, files here survive across all calls in this conversation
    - /output     : save the output file here
    - /upload     : user-uploaded files available for reading

    Save the file to /output/<filename>:
        fig.savefig("/output/chart.png")
        df.to_excel("/output/report.xlsx", index=False)
        with open("/output/report.pdf", "wb") as f: f.write(data)

    Args:
        filename: Output filename with extension (e.g. "report.pdf", "chart.png")
        code: Python code that saves the file to /output/<filename>
    """
    safe_name = "".join(c for c in filename if c.isalnum() or c in "._-")[:100] or "file"
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    output_host_path = Path(settings.SANDBOX_DATA_DIR) / thread_id / "output" / safe_name

    try:
        ensure_container_running()
        result = exec_in_sandbox(thread_id, code)
    except FileNotFoundError:
        return CreateFileMsg.DOCKER_NOT_AVAILABLE
    except subprocess.TimeoutExpired:
        return CreateFileMsg.TIMEOUT.format(timeout=300)

    if result.returncode != 0:
        stderr = mask_real_paths(result.stderr.strip(), thread_id)
        stdout = mask_real_paths(result.stdout.strip(), thread_id)
        if result.returncode == 137 and not stderr:
            hint = "Process was killed (OOM) — likely exceeded memory limit. Use smaller data or batches."
            return f"Error (OOM kill):\n{hint}" + (f"\n\nPartial output:\n{stdout}" if stdout else "")
        msg = stderr or f"Process exited with code {result.returncode}"
        return CreateFileMsg.EXEC_ERROR.format(stderr=msg) + (f"\n\nPartial output:\n{stdout}" if stdout else "")

    stderr_hint = f"\n\nStderr: {mask_real_paths(result.stderr.strip(), thread_id)}" if result.stderr.strip() else ""

    if not output_host_path.exists():
        existing = [f.name for f in output_host_path.parent.iterdir()] if output_host_path.parent.exists() else []
        msg = CreateFileMsg.FILE_NOT_CREATED.format(filename=safe_name)
        if existing:
            msg += f" Files currently in /output: {', '.join(existing)}"
        return msg + stderr_hint

    object_name = f"{uuid4().hex[:8]}_{safe_name}"
    mime_type, _ = mimetypes.guess_type(safe_name)
    try:
        get_minio().fput_object(
            settings.MINIO_BUCKET,
            object_name,
            str(output_host_path),
            content_type=mime_type or "application/octet-stream",
        )
    except Exception as e:
        return CreateFileMsg.UPLOAD_ERROR.format(error=str(e))

    url = f"{settings.MINIO_PUBLIC_URL}/{settings.MINIO_BUCKET}/{object_name}"
    return CreateFileMsg.SUCCESS.format(filename=safe_name, url=url)
