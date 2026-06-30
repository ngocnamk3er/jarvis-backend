from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

from app.core.config import settings

router = APIRouter()

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


@router.post("/upload/{thread_id}")
async def upload_file(thread_id: str, file: UploadFile = File(...)):
    safe_thread = Path(thread_id).name
    safe_name = Path(file.filename or "file").name

    upload_dir = Path(settings.SANDBOX_DATA_DIR) / safe_thread / "upload"
    upload_dir.mkdir(parents=True, exist_ok=True)

    dest = upload_dir / safe_name
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 50 MB)")
    dest.write_bytes(data)

    return {"filename": safe_name, "virtual_path": f"/upload/{safe_name}"}


@router.delete("/upload/{thread_id}/{filename}")
async def delete_uploaded_file(thread_id: str, filename: str):
    safe_thread = Path(thread_id).name
    safe_name = Path(filename).name
    path = Path(settings.SANDBOX_DATA_DIR) / safe_thread / "upload" / safe_name

    if path.exists() and path.is_file():
        path.unlink()

    return {"ok": True}


@router.get("/{thread_id}/{filename}")
async def download_file(thread_id: str, filename: str):
    safe_name = Path(filename).name
    safe_thread = Path(thread_id).name
    path = Path(settings.SANDBOX_DATA_DIR) / safe_thread / "output" / safe_name

    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=str(path), filename=safe_name)
