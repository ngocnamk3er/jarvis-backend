"""/chat/sandbox-files: the files in a conversation's sandbox, for the chat's Sandbox panel (app/services/sandbox_files.py).

Downloading one is the existing GET /chat/sandbox-file, uploading POST /chat/sandbox-upload (with an optional folder).
Every call checks that the conversation is the user's.
"""

from fastapi import APIRouter, Depends, HTTPException
from k8s_agent_sandbox.exceptions import SandboxRequestError
from pydantic import BaseModel, Field

from app.api.deps import CurrentUser, get_current_user
from app.api.v1.endpoints.chat import _check_owns_thread
from app.services import sandbox_files
from app.services.sandbox_files import SandboxFileError, SandboxPathError

router = APIRouter()


class RenameBody(BaseModel):
    thread_id: str
    path: str = Field(max_length=4096)
    new_name: str = Field(max_length=255)


class FolderBody(BaseModel):
    thread_id: str
    parent: str = Field(default="", max_length=4096)
    name: str = Field(max_length=255)


class SaveBody(BaseModel):
    thread_id: str
    path: str = Field(max_length=4096)


async def _do(coro):
    try:
        return await coro
    except SandboxPathError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    except SandboxFileError as e:
        raise HTTPException(status_code=e.status, detail=e.message) from None
    except SandboxRequestError as e:
        raise HTTPException(status_code=503, detail="Sandbox unavailable") from e


@router.get("")
async def list_files(thread_id: str, path: str = "", wake: bool = False,
                     user: CurrentUser = Depends(get_current_user)):
    """A folder of the sandbox. `state` "none" or "paused" comes with no entries; `wake=true` opens a paused one."""
    await _check_owns_thread(thread_id, user)
    return await _do(sandbox_files.list_dir(thread_id, path, wake))


@router.delete("", status_code=204)
async def delete_file(thread_id: str, path: str, user: CurrentUser = Depends(get_current_user)):
    await _check_owns_thread(thread_id, user)
    await _do(sandbox_files.delete(thread_id, path))


@router.post("/rename")
async def rename_file(body: RenameBody, user: CurrentUser = Depends(get_current_user)):
    await _check_owns_thread(body.thread_id, user)
    return {"path": await _do(sandbox_files.rename(body.thread_id, body.path, body.new_name))}


@router.post("/folders", status_code=201)
async def make_folder(body: FolderBody, user: CurrentUser = Depends(get_current_user)):
    await _check_owns_thread(body.thread_id, user)
    return {"path": await _do(sandbox_files.make_folder(body.thread_id, body.parent, body.name))}


@router.post("/save")
async def save_to_files(body: SaveBody, user: CurrentUser = Depends(get_current_user)):
    """Copy a sandbox file into the user's Files, where it outlives the conversation."""
    await _check_owns_thread(body.thread_id, user)
    return await _do(sandbox_files.save_to_files(body.thread_id, body.path, user.sub))
