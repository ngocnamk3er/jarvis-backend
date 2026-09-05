from fastapi import APIRouter, Depends, File, Form, Response, UploadFile

from app.api.deps import CurrentUser, get_current_user
from app.schemas.files import FileNodeOut, FolderCreateBody, NodeDetailOut, NodeUpdateBody
from app.services import file_service

router = APIRouter()


@router.get("/tree", response_model=list[FileNodeOut])
async def list_tree(path: str = "/", user: CurrentUser = Depends(get_current_user)):
    return await file_service.list_tree(user.sub, path)


@router.get("/nodes/{node_id}", response_model=NodeDetailOut | None)
async def get_node(node_id: str, user: CurrentUser = Depends(get_current_user)):
    return await file_service.get_node(user.sub, node_id)


@router.get("/nodes/{node_id}/content")
async def get_node_content(node_id: str, user: CurrentUser = Depends(get_current_user)):
    """Raw bytes for the frontend's preview (image <img>/PDF <iframe> via a
    blob URL) and download (<a download> via the same blob) — see
    file-browser-modal.tsx. inline, not attachment: the frontend decides
    preview vs. download itself, this just needs to not be blocked."""
    content, mime_type, filename = await file_service.get_content(user.sub, node_id)
    return Response(
        content=content,
        media_type=mime_type,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@router.post("/folders", response_model=FileNodeOut)
async def create_folder(body: FolderCreateBody, user: CurrentUser = Depends(get_current_user)):
    return await file_service.create_folder(user.sub, body.parent_path, body.name)


@router.post("/upload", response_model=FileNodeOut)
async def upload_file(
    parent_path: str = Form("/"),
    file: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
):
    content = await file.read()
    return await file_service.upload_file(user.sub, parent_path, file.filename, content, file.content_type)


@router.patch("/nodes/{node_id}", response_model=FileNodeOut)
async def update_node(node_id: str, body: NodeUpdateBody, user: CurrentUser = Depends(get_current_user)):
    return await file_service.rename_or_move(user.sub, node_id, body.name, body.parent_path)


@router.delete("/nodes/{node_id}")
async def delete_node(node_id: str, user: CurrentUser = Depends(get_current_user)):
    await file_service.delete_node(user.sub, node_id)
    return {"ok": True}
