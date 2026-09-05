"""HTTP client for jarvis-file-service — owns the per-user folder/file tree
(metadata), MinIO (blobs), and Qdrant (embeddings). Same posture as
conversation_client.py: internal-only API behind X-Internal-Api-Key, every
call takes an explicit user_id since this backend already verified the
caller's JWT before reaching here.

No retry/circuit breaker here either, for the same reason conversation_client
has none — see that module's docstring.
"""
from datetime import datetime

import httpx
from pydantic import BaseModel

from app.core.config import settings

_client: httpx.AsyncClient | None = None


def init_client() -> None:
    global _client
    _client = httpx.AsyncClient(
        base_url=f"{settings.FILE_SERVICE_URL}{settings.API_PREFIX}",
        headers={"X-Internal-Api-Key": settings.INTERNAL_API_KEY},
        # Higher than conversation_client's 10s — uploads carry a request body,
        # not just a small JSON payload.
        timeout=30.0,
    )


async def close_client() -> None:
    if _client is not None:
        await _client.aclose()


def _get_client() -> httpx.AsyncClient:
    if _client is None:
        raise RuntimeError("file_client not initialised")
    return _client


class FileNodeDTO(BaseModel):
    id: str
    user_id: str
    parent_id: str | None = None
    name: str
    type: str
    mime_type: str | None = None
    size_bytes: int | None = None
    indexing_status: str
    created_at: datetime
    updated_at: datetime


class FileNodeDetailDTO(FileNodeDTO):
    extracted_text: str | None = None


class FileSearchResultDTO(FileNodeDTO):
    path: str


async def list_tree(user_id: str, path: str = "/") -> list[dict]:
    resp = await _get_client().get("/files/tree", params={"user_id": user_id, "path": path})
    resp.raise_for_status()
    return [FileNodeDTO(**n).model_dump() for n in resp.json()]


async def create_folder(user_id: str, parent_path: str, name: str) -> dict:
    resp = await _get_client().post(
        "/files/folders", json={"user_id": user_id, "parent_path": parent_path, "name": name}
    )
    resp.raise_for_status()
    return FileNodeDTO(**resp.json()).model_dump()


async def upload_file(
    user_id: str, parent_path: str, filename: str, content: bytes, content_type: str | None
) -> dict:
    resp = await _get_client().post(
        "/files/files",
        data={"user_id": user_id, "parent_path": parent_path},
        files={"file": (filename, content, content_type or "application/octet-stream")},
    )
    resp.raise_for_status()
    return FileNodeDTO(**resp.json()).model_dump()


async def get_node(user_id: str, node_id: str) -> dict | None:
    resp = await _get_client().get(f"/files/nodes/{node_id}", params={"user_id": user_id})
    resp.raise_for_status()
    data = resp.json()
    return FileNodeDetailDTO(**data).model_dump() if data is not None else None


async def get_content(user_id: str, node_id: str) -> tuple[bytes, str, str]:
    """Returns (raw_bytes, mime_type, filename) for preview/download in the
    frontend — see app/api/v1/endpoints/files.py's proxy of this."""
    resp = await _get_client().get(f"/files/nodes/{node_id}/content", params={"user_id": user_id})
    resp.raise_for_status()
    mime_type = resp.headers.get("content-type", "application/octet-stream")
    filename = "download"
    disposition = resp.headers.get("content-disposition", "")
    if 'filename="' in disposition:
        filename = disposition.split('filename="', 1)[1].split('"', 1)[0]
    return resp.content, mime_type, filename


async def resolve_path(user_id: str, path: str) -> dict | None:
    resp = await _get_client().get("/files/resolve", params={"user_id": user_id, "path": path})
    resp.raise_for_status()
    data = resp.json()
    return FileNodeDetailDTO(**data).model_dump() if data is not None else None


async def rename_or_move(
    user_id: str, node_id: str, name: str | None = None, parent_path: str | None = None
) -> dict:
    resp = await _get_client().patch(
        f"/files/nodes/{node_id}", params={"user_id": user_id}, json={"name": name, "parent_path": parent_path}
    )
    resp.raise_for_status()
    return FileNodeDTO(**resp.json()).model_dump()


async def delete_node(user_id: str, node_id: str) -> None:
    resp = await _get_client().delete(f"/files/nodes/{node_id}", params={"user_id": user_id})
    resp.raise_for_status()


async def grep_files(user_id: str, query: str, path: str = "/") -> list[dict]:
    resp = await _get_client().get(
        "/files/search/grep", params={"user_id": user_id, "query": query, "path": path}
    )
    resp.raise_for_status()
    return [FileSearchResultDTO(**n).model_dump() for n in resp.json()]


async def search_vector(user_id: str, query: str, top_k: int = 5) -> list[dict]:
    resp = await _get_client().post(
        "/files/search/vector", json={"user_id": user_id, "query": query, "top_k": top_k}
    )
    resp.raise_for_status()
    return resp.json()
