"""Models users add themselves: stored with the key encrypted, looked up at call time, checked step by step."""

import json
from datetime import datetime, timezone

import httpx
import pytest
from cryptography.fernet import Fernet

from app.agents import llm, user_models
from app.core.config import settings
from app.services import model_check, user_model_service

UID = "11111111-1111-1111-1111-111111111111"


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch):
    monkeypatch.setattr(settings, "MCP_SECRET_KEY", Fernet.generate_key().decode())
    user_models._cache.clear()
    llm._delegates.clear()


class FakeRepo:
    """user_model_repository in memory."""

    def __init__(self):
        self.rows: dict[str, dict] = {}

    async def list_models(self, user_id):
        return [r for r in self.rows.values() if r["user_id"] == user_id]

    async def get_model(self, user_id, model_id):
        r = self.rows.get(model_id)
        return r if r and r["user_id"] == user_id else None

    async def create_model(self, user_id, name, kind, model, base_url, api_key_enc, context_window, send_reasoning):
        if any(r["user_id"] == user_id and r["name"] == name for r in self.rows.values()):
            return None
        mid = f"00000000-0000-0000-0000-{len(self.rows) + 1:012d}"
        row = {"id": mid, "user_id": user_id, "name": name, "kind": kind, "model": model, "base_url": base_url,
               "api_key_enc": api_key_enc, "vision": False, "context_window": context_window,
               "send_reasoning": send_reasoning, "last_status": None, "last_checked_at": None, "last_result": None,
               "created_at": datetime.now(timezone.utc)}
        self.rows[mid] = row
        return row

    async def delete_model(self, user_id, model_id):
        return self.rows.pop(model_id, None) is not None

    async def save_check(self, user_id, model_id, status, result, vision):
        r = self.rows[model_id]
        r.update(last_status=status, last_result=result, last_checked_at=datetime.now(timezone.utc))
        if vision is not None:
            r["vision"] = vision


@pytest.fixture
def repo(monkeypatch):
    fake = FakeRepo()
    for name in ("list_models", "get_model", "create_model", "delete_model", "save_check"):
        monkeypatch.setattr(user_models.user_model_repository, name, getattr(fake, name))
        monkeypatch.setattr(user_model_service.user_model_repository, name, getattr(fake, name))

    async def allow(url):          # the public-address check needs DNS; it has its own tests
        if "10.0.0.5" in url:
            from app.agents.mcp_guard import McpPolicyError
            raise McpPolicyError("That address is not publicly reachable; only public servers can be added.")

    monkeypatch.setattr(user_model_service, "check_url", allow)
    monkeypatch.setattr(model_check, "check_url", allow)
    return fake


async def _add(**kw):
    args = dict(user_id="u1", name="Mine", kind="endpoint", model="llama3", base_url="https://api.example.com/v1",
                api_key="sk-secret-123", context_window=32_000, send_reasoning=False)
    args.update(kw)
    return await user_model_service.add_model(**args)


# ---- adding ---------------------------------------------------------------------------------------------

async def test_an_own_endpoint_is_stored_with_its_key_encrypted_and_the_key_never_comes_back(repo):
    added = await _add()
    assert added["id"].startswith("user:") and added["has_key"] is True and added["status"] == "unchecked"
    assert "sk-secret-123" not in json.dumps(added) and "api_key" not in added
    stored = next(iter(repo.rows.values()))["api_key_enc"]
    assert stored and "sk-secret-123" not in stored                       # encrypted at rest
    listed = await user_model_service.list_models("u1")
    assert "sk-secret-123" not in json.dumps(listed) and listed["models"][0]["name"] == "Mine"
    assert listed["catalog"][0]["id"] == added["id"] and listed["catalog"][0]["custom"] is True


async def test_an_openrouter_model_needs_no_url_or_key_and_uses_the_servers(repo, monkeypatch):
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "server-key")
    added = await _add(kind="openrouter", name="Mistral", model="mistralai/mistral-large-4-0", base_url="http://ignored", api_key="ignored")
    row = next(iter(repo.rows.values()))
    assert row["base_url"] is None and row["api_key_enc"] is None and row["send_reasoning"] is True
    custom = await user_models.resolve("u1", added["id"])
    assert custom.api_key == "server-key" and custom.base_url == settings.OPENROUTER_BASE_URL and custom.model == "mistralai/mistral-large-4-0"


@pytest.mark.parametrize("kw, text", [
    ({"name": "  "}, "name of 1 to 40"),
    ({"kind": "bogus"}, "Kind must be"),
    ({"model": "has space"}, "no spaces"),
    ({"context_window": 10}, "context window"),
    ({"base_url": ""}, "needs a base URL"),
    ({"base_url": "http://10.0.0.5/v1"}, "not publicly reachable"),
])
async def test_what_cannot_be_added_is_refused_with_a_reason(repo, kw, text):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as e:
        await _add(**kw)
    assert e.value.status_code == 400 and text in e.value.detail and not repo.rows


async def test_a_name_can_be_used_once_and_the_number_of_models_is_capped(repo, monkeypatch):
    from fastapi import HTTPException

    await _add()
    with pytest.raises(HTTPException) as e:
        await _add()
    assert e.value.status_code == 409
    monkeypatch.setattr(settings, "USER_MODELS_MAX", 1)
    with pytest.raises(HTTPException) as e:
        await _add(name="Second")
    assert "At most 1" in e.value.detail


async def test_a_key_cannot_be_stored_without_a_server_secret(repo, monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(settings, "MCP_SECRET_KEY", "")
    with pytest.raises(HTTPException) as e:
        await _add()
    assert "MCP_SECRET_KEY" in e.value.detail
    assert (await _add(api_key=None))["has_key"] is False                  # a keyless endpoint (a local Ollama) needs none


async def test_models_are_private_to_their_owner_and_removal_forgets_them(repo):
    added = await _add()
    uid = added["id"][len("user:"):]
    assert await user_models.resolve("someone-else", added["id"]) is None
    assert (await user_model_service.list_models("someone-else"))["models"] == []
    await user_model_service.remove_model("u1", uid)
    assert await user_models.resolve("u1", added["id"]) is None
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as e:
        await user_model_service.remove_model("u1", "not-a-uuid")
    assert e.value.status_code == 404


# ---- how a turn reaches a user's model --------------------------------------------------------------------

async def test_info_carries_what_the_turn_needs_but_never_the_key(repo):
    added = await _add(context_window=64_000)
    info = await user_models.info("u1", added["id"])
    assert info == {"model_vision": False, "model_context_window": 64_000, "no_fallback": True}
    assert await user_models.info("u1", "qwen/qwen3.6-35b-a3b") == {}          # a built-in model: nothing extra
    assert await user_models.info("u1", None) == {}
    with pytest.raises(user_models.UserModelError):
        await user_models.info("u1", "user:00000000-0000-0000-0000-0000000000ff")


def test_vision_follows_the_models_own_record_for_a_users_model_and_the_list_for_the_rest():
    builtin = lambda m: m == "qwen/qwen3.6-35b-a3b"
    assert user_models.supports_vision({"model": "user:abc", "model_vision": True}, builtin) is True
    assert user_models.supports_vision({"model": "user:abc"}, builtin) is False
    assert user_models.supports_vision({"model": "qwen/qwen3.6-35b-a3b"}, builtin) is True


def test_a_turn_config_gets_the_model_facts_and_no_secret():
    from app.services.chat_service import _make_config

    cfg = _make_config("t", "u1", model="user:abc", model_info={"model_vision": True, "model_context_window": 9000, "no_fallback": True})
    c = cfg["configurable"]
    assert c["model"] == "user:abc" and c["model_context_window"] == 9000 and c["no_fallback"] is True
    assert "api_key" not in json.dumps(cfg, default=str)


async def test_a_call_on_a_users_model_goes_to_its_own_endpoint_client(repo, monkeypatch):
    added = await _add(model="llama3")
    model = llm.ThinkingChatOpenAI(model="x", api_key="k", base_url="http://openrouter", honor_model_override=True)
    monkeypatch.setattr(model, "_configurable", lambda: {"model": added["id"], "user_id": "u1"})
    custom = await model._custom_for_turn()
    assert custom.model == "llama3" and custom.api_key == "sk-secret-123" and custom.base_url == "https://api.example.com/v1"
    delegate = llm._delegate_for(custom)
    assert delegate.model_name == "llama3" and delegate.openai_api_base == "https://api.example.com/v1"
    assert delegate.delegate is True and delegate.honor_model_override is False and delegate.send_reasoning is False
    assert llm._delegate_for(custom) is delegate                                # reused between turns

    monkeypatch.setattr(model, "_configurable", lambda: {"model": "qwen/qwen3.6-35b-a3b", "user_id": "u1"})
    assert await model._custom_for_turn() is None                              # a built-in model: the ordinary path
    monkeypatch.setattr(model, "_configurable", lambda: {"model": "user:00000000-0000-0000-0000-0000000000ff", "user_id": "u1"})
    with pytest.raises(user_models.UserModelError):
        await model._custom_for_turn()


def test_a_fallback_model_refuses_to_answer_for_a_turn_on_an_own_endpoint(monkeypatch):
    fallback = llm.ThinkingChatOpenAI(model="f", api_key="k", base_url="http://openrouter", honor_model_override=False)
    monkeypatch.setattr(fallback, "_configurable", lambda: {"no_fallback": True})
    with pytest.raises(RuntimeError, match="fallback models are not used"):
        fallback._refuse_fallback()
    monkeypatch.setattr(fallback, "_configurable", lambda: {})
    fallback._refuse_fallback()                                                  # an ordinary turn: fallbacks work as before


def test_the_reasoning_field_is_only_sent_where_it_is_wanted():
    from langchain_core.messages import HumanMessage

    on = llm.ThinkingChatOpenAI(model="m", api_key="k", base_url="http://x", send_reasoning=True)
    off = llm.ThinkingChatOpenAI(model="m", api_key="k", base_url="http://x", send_reasoning=False)
    assert on._get_request_payload([HumanMessage(content="hi")])["extra_body"]["reasoning"]["effort"] == "high"
    assert "extra_body" not in off._get_request_payload([HumanMessage(content="hi")])


# ---- the connection check ---------------------------------------------------------------------------------

def _custom(**kw):
    base = dict(id="user:x", name="n", kind="endpoint", model="llama3", base_url="https://api.example.com/v1",
                api_key="sk-1", vision=False, context_window=8000, send_reasoning=False)
    base.update(kw)
    return user_models.CustomModel(**base)


def _server(models=("llama3",), chat="pong", tools=True, vision_word="red", key="sk-1", chat_status=200):
    """An OpenAI-compatible endpoint that is handled in memory."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("authorization", "") != f"Bearer {key}":
            return httpx.Response(401, json={"error": {"message": "Incorrect API key provided"}})
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": m} for m in models]})
        body = json.loads(request.content)
        if chat_status != 200:
            return httpx.Response(chat_status, json={"error": {"message": "boom"}})
        if body.get("tools"):
            msg = {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function",
                   "function": {"name": "add", "arguments": "{\"a\":17,\"b\":25}"}}]} if tools else {"role": "assistant", "content": "42"}
        elif any(isinstance(m["content"], list) for m in body["messages"]):
            msg = {"role": "assistant", "content": vision_word}
        else:
            msg = {"role": "assistant", "content": chat}
        return httpx.Response(200, json={"choices": [{"message": msg}]})
    return httpx.MockTransport(handler)


def _names(result):
    return {s["name"]: s["ok"] for s in result["steps"]}


async def test_a_good_endpoint_that_calls_tools_and_sees_images_checks_out_ok(repo):
    status, result, vision = await model_check.check(_custom(), transport=_server())
    assert status == "ok" and vision is True
    assert _names(result) == {"Address": True, "Connection": True, "Test call": True, "Tool calls": True, "Images": True}
    assert result["capabilities"] == {"tools": True, "vision": True}


async def test_a_chat_only_model_is_degraded_not_broken(repo):
    status, result, vision = await model_check.check(_custom(), transport=_server(tools=False, vision_word="I cannot tell"))
    assert status == "degraded" and vision is False
    assert _names(result)["Test call"] is True and _names(result)["Tool calls"] is False


async def test_a_rejected_key_names_the_step_and_stops_there(repo):
    status, result, vision = await model_check.check(_custom(api_key="wrong"), transport=_server())
    assert status == "error" and vision is None and result["capabilities"] is None
    assert _names(result) == {"Address": True, "API key": False}
    assert "401" in result["steps"][-1]["detail"] and "wrong" not in json.dumps(result)         # the reason, never the key


async def test_a_model_the_endpoint_does_not_have_is_reported(repo):
    status, result, _ = await model_check.check(_custom(model="gpt-9"), transport=_server(models=("llama3",)))
    assert status == "error" and _names(result)["Model"] is False and "gpt-9" in result["steps"][-1]["detail"]


async def test_an_address_the_server_may_not_call_is_refused_before_any_request(repo):
    calls = []
    transport = httpx.MockTransport(lambda r: calls.append(r) or httpx.Response(200, json={}))
    status, result, _ = await model_check.check(_custom(base_url="http://10.0.0.5/v1"), transport=transport)
    assert status == "error" and _names(result) == {"Address": False} and calls == []


async def test_an_endpoint_that_errors_on_the_test_call_is_an_error_with_its_message(repo):
    status, result, _ = await model_check.check(_custom(), transport=_server(chat_status=500))
    assert status == "error" and _names(result)["Test call"] is False and "boom" in result["steps"][-1]["detail"]


async def test_a_server_that_does_not_list_models_is_still_tested_by_a_call(repo):
    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(404, json={})
        return _server()(request) if False else httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "pong"}}]})
    status, result, _ = await model_check.check(_custom(), transport=httpx.MockTransport(handler))
    assert _names(result)["Models list"] is None and _names(result)["Test call"] is True and status == "degraded"


async def test_checking_a_stored_model_saves_the_status_and_what_it_can_see(repo, monkeypatch):
    added = await _add()
    uid = added["id"][len("user:"):]

    async def fake_check(custom, transport=None):
        assert custom.api_key == "sk-secret-123"                                 # the stored key was decrypted for the call
        return "ok", {"steps": [], "capabilities": {"tools": True, "vision": True}, "total_ms": 5}, True

    monkeypatch.setattr(model_check, "check", fake_check)
    out = await user_model_service.check_model("u1", uid)
    assert out["status"] == "ok" and out["vision"] is True and out["checked_at"] and out["has_key"] is True
    assert (await user_models.info("u1", added["id"]))["model_vision"] is True     # the agent now believes it sees images
