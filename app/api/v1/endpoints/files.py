from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.config import settings

router = APIRouter()


@router.get("/{thread_id}/{filename}")
async def download_file(thread_id: str, filename: str):
    safe_name = Path(filename).name
    safe_thread = Path(thread_id).name
    path = Path(settings.SANDBOX_DATA_DIR) / safe_thread / "output" / safe_name

    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=str(path), filename=safe_name)
