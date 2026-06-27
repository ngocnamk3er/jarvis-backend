from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.config import settings

router = APIRouter()


@router.get("/{filename}")
async def download_file(filename: str):
    safe_name = Path(filename).name
    path = Path(settings.FILES_DIR) / safe_name

    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # Strip the uuid prefix for the downloaded filename
    display_name = safe_name.split("_", 1)[-1] if "_" in safe_name else safe_name

    return FileResponse(
        path=str(path),
        filename=display_name,
        media_type="application/octet-stream",
    )
