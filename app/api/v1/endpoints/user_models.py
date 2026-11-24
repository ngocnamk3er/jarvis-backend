from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import CurrentUser, get_current_user
from app.services import user_model_service

router = APIRouter()


class AddModel(BaseModel):
    name: str = Field(max_length=40)
    kind: str = "openrouter"
    model: str = Field(max_length=200)
    base_url: str | None = Field(default=None, max_length=2048)
    api_key: str | None = Field(default=None, max_length=512)
    context_window: int = 128_000
    send_reasoning: bool = False


@router.get("")
async def list_models(user: CurrentUser = Depends(get_current_user)):
    return await user_model_service.list_models(user.sub)


@router.post("", status_code=201)
async def add_model(body: AddModel, user: CurrentUser = Depends(get_current_user)):
    return await user_model_service.add_model(user.sub, body.name, body.kind, body.model, body.base_url,
                                              body.api_key, body.context_window, body.send_reasoning)


@router.post("/{model_id}/check")
async def check_model(model_id: str, user: CurrentUser = Depends(get_current_user)):
    return await user_model_service.check_model(user.sub, model_id)


@router.delete("/{model_id}", status_code=204)
async def remove_model(model_id: str, user: CurrentUser = Depends(get_current_user)):
    await user_model_service.remove_model(user.sub, model_id)
