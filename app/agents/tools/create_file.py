import mimetypes
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

from langchain_core.tools import tool

from app.core.config import settings
from app.agents.tools.messages import CreateFileMsg
from app.storage.minio_client import get_minio

_DOCKER_IMAGE = "jarvis-sandbox"
_TIMEOUT_SECONDS = 30


@tool
def create_file(filename: str, code: str) -> str:
    """Create a file (docx, pdf, png, svg, xlsx, etc.) by running Python code.

    The code MUST write the output to the path in the OUTPUT_PATH environment variable.
    Available libraries: python-docx, reportlab, Pillow, svgwrite, openpyxl, fpdf2, matplotlib.

    Examples:
      DOCX:
        import os; from docx import Document
        doc = Document(); doc.add_heading('Title', 0); doc.save(os.environ['OUTPUT_PATH'])

      PDF:
        import os; from reportlab.pdfgen import canvas
        c = canvas.Canvas(os.environ['OUTPUT_PATH']); c.drawString(100,750,'Hi'); c.save()

      PNG:
        import os; from PIL import Image, ImageDraw
        img = Image.new('RGB', (800,600), 'white')
        img.save(os.environ['OUTPUT_PATH'])

      SVG:
        import os, svgwrite
        dwg = svgwrite.Drawing(os.environ['OUTPUT_PATH'], size=('200px','200px'))
        dwg.add(dwg.circle((100,100), 50, fill='blue')); dwg.save()
    """
    safe_name = "".join(c for c in filename if c.isalnum() or c in "._-")[:100] or "file"
    object_name = f"{uuid4().hex[:8]}_{safe_name}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = Path(tmp_dir) / object_name
        container_output = f"/output/{object_name}"

        try:
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "--network", "none",
                    "--memory", "256m",
                    "--cpus", "0.5",
                    "--security-opt", "no-new-privileges:true",
                    "--tmpfs", "/tmp:size=64m",
                    "-v", f"{tmp_dir}:/output",
                    "-e", f"OUTPUT_PATH={container_output}",
                    "-i",
                    _DOCKER_IMAGE,
                    "python", "-",
                ],
                input=code,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_SECONDS,
            )
        except FileNotFoundError:
            return CreateFileMsg.DOCKER_NOT_AVAILABLE
        except subprocess.TimeoutExpired:
            return CreateFileMsg.TIMEOUT.format(timeout=_TIMEOUT_SECONDS)

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
