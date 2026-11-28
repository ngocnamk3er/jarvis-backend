"""Flows a user builds: saving them (always valid), and getting one ready to run."""

import re
import unicodedata
import uuid

from fastapi import HTTPException

from app.agents import user_models
from app.core import observability
from app.core.config import settings
from app.db import flow_repository
from app.flows import catalog as flow_catalog
from app.flows.runner import FlowInputError, coerce_inputs
from app.flows.schema import CONFIGS, ParsedFlow, validate_flow
from app.schemas.chat import AVAILABLE_MODELS, DEFAULT_MODEL

MAX_NAME = 60
MAX_DESCRIPTION = 500
BUILTIN_MODELS = {m["id"] for m in AVAILABLE_MODELS}


def _bad(detail) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="No such flow.")


def _check_enabled() -> None:
    if not settings.FLOWS_ENABLED:
        raise HTTPException(status_code=404, detail="Flows are turned off.")


def _check_id(flow_id: str) -> None:
    try:
        uuid.UUID(flow_id)
    except ValueError:
        raise _not_found() from None


def slugify(name: str) -> str:
    """A name as letters, digits and _: 'Tóm tắt tin tức' -> 'tom_tat_tin_tuc'."""
    s = unicodedata.normalize("NFKD", name.replace("đ", "d").replace("Đ", "D"))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")[:40].strip("_")
    return s or "flow"


def problems_of(definition: dict) -> tuple[ParsedFlow | None, list[dict]]:
    parsed, problems = validate_flow(definition, flow_catalog.tool_specs(), models=BUILTIN_MODELS)
    return parsed, [p.as_dict() for p in problems]


def _valid(definition: dict) -> ParsedFlow:
    parsed, problems = problems_of(definition)
    if problems:
        raise _bad({"message": "The flow has problems; nothing was saved.", "problems": problems})
    return parsed


def _check_meta(name: str, description: str) -> str:
    name = name.strip()
    if not 1 <= len(name) <= MAX_NAME:
        raise _bad(f"Give the flow a name of 1 to {MAX_NAME} characters.")
    if len(description) > MAX_DESCRIPTION:
        raise _bad(f"The description can be {MAX_DESCRIPTION} characters at most.")
    return name


def catalog() -> dict:
    """What the editor offers: the node types (and their settings), the tools, the models."""
    return {
        "node_types": {t: m.model_json_schema() for t, m in CONFIGS.items()},
        "tools": flow_catalog.describe(),
        "models": [{"id": m["id"], "name": m["name"]} for m in AVAILABLE_MODELS],
    }


def validate(definition: dict) -> dict:
    _, problems = problems_of(definition)
    return {"valid": not problems, "problems": problems}


async def list_flows(user_id: str) -> list[dict]:
    _check_enabled()
    return await flow_repository.list_flows(user_id)


async def get_flow(user_id: str, flow_id: str) -> dict:
    _check_enabled()
    _check_id(flow_id)
    row = await flow_repository.get_flow(user_id, flow_id)
    if row is None:
        raise _not_found()
    return row


async def create_flow(user_id: str, name: str, description: str, definition: dict) -> dict:
    _check_enabled()
    name = _check_meta(name, description)
    _valid(definition)
    if await flow_repository.count_flows(user_id) >= settings.FLOWS_MAX_PER_USER:
        raise _bad(f"You can have {settings.FLOWS_MAX_PER_USER} flows at most; delete one first.")
    try:
        return await flow_repository.create_flow(user_id, name, slugify(name), description, definition)
    except flow_repository.SlugTaken:
        raise HTTPException(status_code=409, detail="You already have a flow with this name.") from None


async def update_flow(user_id: str, flow_id: str, name: str, description: str, definition: dict,
                      enabled: bool) -> dict:
    _check_enabled()
    _check_id(flow_id)
    name = _check_meta(name, description)
    _valid(definition)
    try:
        row = await flow_repository.update_flow(user_id, flow_id, name, slugify(name), description, definition,
                                                enabled)
    except flow_repository.SlugTaken:
        raise HTTPException(status_code=409, detail="You already have a flow with this name.") from None
    if row is None:
        raise _not_found()
    return row


async def set_enabled(user_id: str, flow_id: str, enabled: bool) -> dict:
    _check_enabled()
    _check_id(flow_id)
    row = await flow_repository.set_enabled(user_id, flow_id, enabled)
    if row is None:
        raise _not_found()
    return row


async def delete_flow(user_id: str, flow_id: str) -> None:
    _check_enabled()
    _check_id(flow_id)
    if not await flow_repository.delete_flow(user_id, flow_id):
        raise _not_found()


async def prepare_run(user_id: str, flow_id: str, inputs: dict, model: str | None,
                      file_spaces: list[dict] | None = None) -> tuple[ParsedFlow, dict]:
    """The saved flow ready to run, and the run's config; 404 / 400 before anything runs."""
    row = await get_flow(user_id, flow_id)
    parsed = _valid(row["definition"])          # saved flows are valid; the tools on offer may have changed since
    try:
        coerce_inputs(parsed.input_fields, inputs)
    except FlowInputError as e:
        raise _bad(str(e)) from None
    model = model or DEFAULT_MODEL
    if not user_models.is_user_model(model) and model not in BUILTIN_MODELS:
        raise _bad(f"Unknown model '{model}'.")
    try:
        model_info = await user_models.info(user_id, model)
    except user_models.UserModelError as e:
        raise _bad(str(e)) from None
    thread_id = f"flow-{flow_id}"
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            # The departments whose documents a workspace tool step may read (app/core/spaces.py: for_agent).
            "file_spaces": file_spaces or [],
            "model": model,
            "thinking_effort": "high",
            # No sandbox behind a flow run: web_search/web_fetch cut a long result inline instead of saving it.
            "sandbox": False,
            **model_info,
        },
        "recursion_limit": 100,
        "callbacks": observability.callbacks(),
        "metadata": observability.trace_metadata(thread_id, user_id),
    }
    return parsed, config
