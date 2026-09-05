from datetime import datetime
from pydantic import BaseModel


class FileNodeOut(BaseModel):
    id: str
    parent_id: str | None = None
    name: str
    type: str
    mime_type: str | None = None
    size_bytes: int | None = None
    indexing_status: str
    created_at: datetime
    updated_at: datetime


class FolderCreateBody(BaseModel):
    name: str
    parent_path: str = "/"


class NodeUpdateBody(BaseModel):
    name: str | None = None
    parent_path: str | None = None
