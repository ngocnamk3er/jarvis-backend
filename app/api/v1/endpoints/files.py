"""The Files page: the user's own files and their departments' (app/core/spaces.py).

Every route takes `space` ("personal", the default, or a department's "dept:<slug>"). Reading needs membership,
changing needs the department's editors group; both are checked here, before the file service is called with the
space's owner id. A node is looked up by id within that owner only, so an id from another space is simply not found.
"""

from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile

from app.api.deps import CurrentUser, get_current_user
from app.core import spaces
from app.schemas.files import FileNodeOut, FolderCreateBody, NodeDetailOut, NodeUpdateBody
from app.services import file_service

router = APIRouter()


def _space(user: CurrentUser, space: str | None, write: bool = False) -> spaces.Space:
    try:
        return spaces.resolve(user.sub, user.groups, space, write=write)
    except spaces.SpaceError as e:
        raise HTTPException(status_code=e.status, detail=str(e)) from None


def content_disposition(filename: str) -> str:
    """inline, with the name in both forms: an ASCII stand-in for old clients, and the real one (RFC 6266/5987).
    A header must be latin-1, so a bare `filename="Kế toán.pdf"` cannot even be sent."""
    ascii_name = filename.encode("ascii", "replace").decode().replace("?", "_").replace('"', "_").replace("\\", "_")
    return f"inline; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename, safe='')}"


@router.get("/spaces")
async def list_spaces(user: CurrentUser = Depends(get_current_user)):
    """The spaces this user can open: their own first, then each department they belong to."""
    return [s.public() for s in spaces.spaces_for(user.sub, user.groups)]


@router.get("/tree", response_model=list[FileNodeOut])
async def list_tree(path: str = "/", space: str | None = None, user: CurrentUser = Depends(get_current_user)):
    return await file_service.list_tree(_space(user, space).owner, path)


async def _find(user: CurrentUser, space: str | None, node_id: str) -> tuple[spaces.Space, dict | None]:
    """The node within `space`; with no space given, within the first readable space that has it — a citation saved
    before spaces existed carries only the file's id."""
    if space:
        target = _space(user, space)
        return target, await file_service.get_node(target.owner, node_id)
    for target in spaces.spaces_for(user.sub, user.groups):
        node = await file_service.get_node(target.owner, node_id)
        if node is not None:
            return target, node
    return spaces.spaces_for(user.sub, ())[0], None


@router.get("/nodes/{node_id}", response_model=NodeDetailOut | None)
async def get_node(node_id: str, space: str | None = None, user: CurrentUser = Depends(get_current_user)):
    return (await _find(user, space, node_id))[1]


@router.get("/nodes/{node_id}/content")
async def get_node_content(node_id: str, space: str | None = None, user: CurrentUser = Depends(get_current_user)):
    """Raw bytes for the frontend's preview (image <img>/PDF <iframe> via a
    blob URL) and download (<a download> via the same blob) — see
    file-browser-modal.tsx. inline, not attachment: the frontend decides
    preview vs. download itself, this just needs to not be blocked."""
    target, node = await _find(user, space, node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Not found")
    content, mime_type, filename = await file_service.get_content(target.owner, node_id)
    return Response(
        content=content,
        media_type=mime_type,
        headers={"Content-Disposition": content_disposition(filename)},
    )


@router.post("/folders", response_model=FileNodeOut)
async def create_folder(body: FolderCreateBody, space: str | None = None, user: CurrentUser = Depends(get_current_user)):
    return await file_service.create_folder(_space(user, space, write=True).owner, body.parent_path, body.name)


@router.post("/upload", response_model=FileNodeOut)
async def upload_file(
    parent_path: str = Form("/"),
    file: UploadFile = File(...),
    space: str | None = None,
    user: CurrentUser = Depends(get_current_user),
):
    owner = _space(user, space, write=True).owner
    content = await file.read()
    return await file_service.upload_file(owner, parent_path, file.filename, content, file.content_type)


@router.patch("/nodes/{node_id}", response_model=FileNodeOut)
async def update_node(
    node_id: str, body: NodeUpdateBody, space: str | None = None, user: CurrentUser = Depends(get_current_user)
):
    return await file_service.rename_or_move(_space(user, space, write=True).owner, node_id, body.name, body.parent_path)


@router.delete("/nodes/{node_id}")
async def delete_node(node_id: str, space: str | None = None, user: CurrentUser = Depends(get_current_user)):
    await file_service.delete_node(_space(user, space, write=True).owner, node_id)
    return {"ok": True}
