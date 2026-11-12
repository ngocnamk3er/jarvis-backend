from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import CurrentUser, get_current_user
from app.services import mcp_service

router = APIRouter()


class AddServer(BaseModel):
    name: str = Field(max_length=64)
    url: str = Field(max_length=2048)
    transport: str = "streamable_http"
    headers: dict[str, str] = Field(default_factory=dict)


class Enabled(BaseModel):
    enabled: bool


@router.get("/servers")
async def list_servers(conversation_id: str | None = None, user: CurrentUser = Depends(get_current_user)):
    return await mcp_service.list_servers(user.sub, conversation_id)


@router.post("/servers", status_code=201)
async def add_server(body: AddServer, user: CurrentUser = Depends(get_current_user)):
    return await mcp_service.add_server(user.sub, body.name, body.url, body.transport, body.headers)


@router.delete("/servers/{server_id}", status_code=204)
async def remove_server(server_id: str, user: CurrentUser = Depends(get_current_user)):
    await mcp_service.remove_server(user.sub, server_id)


@router.put("/conversations/{conversation_id}/servers/{server_id}", status_code=204)
async def set_enabled(conversation_id: str, server_id: str, body: Enabled, user: CurrentUser = Depends(get_current_user)):
    await mcp_service.set_enabled(user.sub, conversation_id, server_id, body.enabled)
