"""A resume (after a bash approval, or an answer to a clarifying question) must carry on with
the model the conversation was using. It used to fall back to the default model, which cannot
see images, partway through a turn."""

from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from app.api.deps import get_current_user
from app.api.v1.endpoints import chat as chat_endpoint
from app.schemas.chat import DEFAULT_MODEL

VISION = "qwen/qwen3.6-35b-a3b"


def conv(last_model=VISION, last_subagent=None, owner="u1"):
    return SimpleNamespace(user_id=owner, last_model=last_model, last_subagent_model=last_subagent)


@pytest.fixture
def api(monkeypatch):
    seen = {"resume": None, "clarify": None, "touch": None}
    state = {"conv": conv()}

    async def get_conversation(thread_id):
        return state["conv"]

    async def touch(thread_id, model=None, subagent_model=None):
        seen["touch"] = (model, subagent_model)

    async def resume(thread_id, decision, graph, user_id, model, subagent_model, web_search):
        seen["resume"] = (model, subagent_model)
        yield "data: {}\n\n"

    async def resume_clarify(thread_id, answer, graph, user_id, model, subagent_model, web_search):
        seen["clarify"] = (model, subagent_model)
        yield "data: {}\n\n"

    monkeypatch.setattr(chat_endpoint.conversation_client, "get_conversation", get_conversation)
    monkeypatch.setattr(chat_endpoint.conversation_client, "touch_conversation", touch)
    monkeypatch.setattr(chat_endpoint.chat_service, "resume", resume)
    monkeypatch.setattr(chat_endpoint.chat_service, "resume_clarify", resume_clarify)
    app = FastAPI()
    app.state.graph = object()
    app.include_router(chat_endpoint.router, prefix="/chat")
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(sub="u1")
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
    client.seen, client.state = seen, state
    return client


async def resume(api, **extra):
    return await api.post("/chat/resume", json={"thread_id": "t", "decision": "approve", **extra})


async def clarify(api, **extra):
    return await api.post("/chat/resume_clarify", json={"thread_id": "t", "answer": "yes", **extra})


async def test_a_resume_without_a_model_continues_on_the_conversations_model(api):
    r = await resume(api)
    assert r.status_code == 200
    assert api.seen["resume"][0] == VISION and api.seen["touch"][0] == VISION      # and last_model is not overwritten


async def test_a_clarify_answer_without_a_model_does_too(api):
    await clarify(api)
    assert api.seen["clarify"][0] == VISION and api.seen["touch"][0] == VISION


async def test_a_model_the_request_names_is_used(api):
    await resume(api, model="anthropic/claude-opus-4.8")
    assert api.seen["resume"][0] == "anthropic/claude-opus-4.8"
    await clarify(api, model="anthropic/claude-opus-4.8")
    assert api.seen["clarify"][0] == "anthropic/claude-opus-4.8"


async def test_a_conversation_with_no_recorded_model_gets_the_default(api):
    api.state["conv"] = conv(last_model=None)
    await resume(api)
    assert api.seen["resume"][0] == DEFAULT_MODEL


async def test_the_subagent_model_is_kept_unless_the_request_names_one(api):
    api.state["conv"] = conv(last_subagent="deepseek/deepseek-v4-pro")
    await resume(api)
    assert api.seen["resume"][1] == "deepseek/deepseek-v4-pro"
    await resume(api, subagent_model="qwen/qwen3.5-9b")
    assert api.seen["resume"][1] == "qwen/qwen3.5-9b"


async def test_someone_elses_conversation_is_still_refused(api):
    api.state["conv"] = conv(owner="someone-else")
    assert (await resume(api)).status_code == 404 and api.seen["resume"] is None
