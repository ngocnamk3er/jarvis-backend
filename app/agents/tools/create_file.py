import mimetypes
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

from langchain_core.tools import tool

from app.core.config import settings
from app.agents.messages import CreateFileMsg
from app.storage.minio_client import get_minio

_DOCKER_IMAGE = "jarvis-sandbox"
_PIP_CACHE_VOLUME = "jarvis-pip-cache"


@tool
def create_file(filename: str, code: str) -> str:
    """Create a file (docx, pdf, png, svg, xlsx, etc.) by running Python code.

    Save the output file to /output/<filename> inside the code.
    Each call runs in a fresh container — install any needed packages inside the code itself.

    Args:
        filename: Output filename with extension (e.g. "report.pdf", "chart.png")
        code: Python code that saves the file to /output/<filename>
    """
    safe_name = "".join(c for c in filename if c.isalnum() or c in "._-")[:100] or "file"
    object_name = f"{uuid4().hex[:8]}_{safe_name}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = Path(tmp_dir) / safe_name

        try:
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "--memory", "512m",
                    "--cpus", "1.0",
                    "--security-opt", "no-new-privileges:true",
                    "--tmpfs", "/tmp:size=512m,exec",
                    "-v", f"{_PIP_CACHE_VOLUME}:/root/.cache/pip",
                    "-v", f"{tmp_dir}:/output",
                    "-i", _DOCKER_IMAGE,
                    "python", "-",
                ],
                input=code,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except FileNotFoundError:
            return CreateFileMsg.DOCKER_NOT_AVAILABLE
        except subprocess.TimeoutExpired:
            return CreateFileMsg.TIMEOUT.format(timeout=300)

        if result.returncode != 0:
            return CreateFileMsg.EXEC_ERROR.format(stderr=result.stderr.strip())

        if not local_path.exists():
            return CreateFileMsg.FILE_NOT_CREATED

        mime_type, _ = mimetypes.guess_type(safe_name)
        try:
            get_minio().fput_object(
                settings.MINIO_BUCKET,
                object_name,
                str(local_path),
                content_type=mime_type or "application/octet-stream",
            )
        except Exception as e:
            return CreateFileMsg.UPLOAD_ERROR.format(error=str(e))

    url = f"{settings.MINIO_PUBLIC_URL}/{settings.MINIO_BUCKET}/{object_name}"
    return CreateFileMsg.SUCCESS.format(filename=safe_name, url=url)
