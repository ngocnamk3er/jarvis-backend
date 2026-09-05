from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import CurrentUser, get_current_user
from app.schemas.files import FileNodeOut, FolderCreateBody, NodeUpdateBody
from app.services import file_service

router = APIRouter()


@router.get("/tree", response_model=list[FileNodeOut])
async def list_tree(path: str = "/", user: CurrentUser = Depends(get_current_user)):
    return await file_service.list_tree(user.sub, path)


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
