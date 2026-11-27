"""Flows (app/flows/): build, save, run.

POST /{flow_id}/run streams the run as Server-Sent Events, one `data: {json}` per event of runner.stream_flow, then
`{"type": "done"}`, with heartbeats while a node is silent (app/core/sse.py).
"""

import json
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import CurrentUser, get_current_user
from app.core.config import settings
from app.core.sse import with_heartbeat
from app.flows.runner import stream_flow
from app.services import flow_service

router = APIRouter()


class FlowBody(BaseModel):
    name: str = Field(max_length=200)
    description: str = Field(default="", max_length=2000)
    definition: dict[str, Any]
    enabled: bool = True


class ValidateBody(BaseModel):
    definition: dict[str, Any]


class RunBody(BaseModel):
    inputs: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None


@router.get("")
async def list_flows(user: CurrentUser = Depends(get_current_user)):
    return await flow_service.list_flows(user.sub)


@router.get("/catalog")
async def catalog(user: CurrentUser = Depends(get_current_user)):
    return flow_service.catalog()


@router.post("/validate")
async def validate(body: ValidateBody, user: CurrentUser = Depends(get_current_user)):
    return flow_service.validate(body.definition)


@router.post("", status_code=201)
async def create_flow(body: FlowBody, user: CurrentUser = Depends(get_current_user)):
    return await flow_service.create_flow(user.sub, body.name, body.description, body.definition)


@router.get("/{flow_id}")
async def get_flow(flow_id: str, user: CurrentUser = Depends(get_current_user)):
    return await flow_service.get_flow(user.sub, flow_id)


@router.put("/{flow_id}")
async def update_flow(flow_id: str, body: FlowBody, user: CurrentUser = Depends(get_current_user)):
    return await flow_service.update_flow(user.sub, flow_id, body.name, body.description, body.definition,
                                          body.enabled)


@router.delete("/{flow_id}", status_code=204)
async def delete_flow(flow_id: str, user: CurrentUser = Depends(get_current_user)):
    await flow_service.delete_flow(user.sub, flow_id)


@router.post("/{flow_id}/run")
async def run_flow(flow_id: str, body: RunBody, user: CurrentUser = Depends(get_current_user)):
    parsed, config = await flow_service.prepare_run(user.sub, flow_id, body.inputs, body.model)

    async def lines():
        async for event in stream_flow(parsed, body.inputs, config):
            yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(with_heartbeat(lines(), settings.SSE_HEARTBEAT_SECONDS), media_type="text/event-stream")
