"""The /flows API: saving only valid flows, each user seeing only their own, and a test run streamed as SSE. The
database is an in-memory stand-in for flow_repository; the model and tools are the fakes of flow_fakes."""

import json
import uuid
from datetime import UTC, datetime

import httpx
import pytest
from app.api.deps import get_current_user
from app.api.v1.endpoints import flows as flows_endpoint
from app.db import flow_repository
from app.flows import catalog, compiler
from app.services import flow_service
from fastapi import FastAPI

from tests import flow_fakes
from tests.flow_fakes import TOOLS, EchoModel, news_flow


class MemoryRepo:
    def __init__(self):
        self.rows: dict[str, dict] = {}

    def _mine(self, user_id, flow_id):
        row = self.rows.get(flow_id)
        return row if row and row["user_id"] == user_id else None

    async def list_flows(self, user_id):
        return [{k: v for k, v in r.items() if k not in ("definition", "user_id")}
                for r in self.rows.values() if r["user_id"] == user_id]

    async def count_flows(self, user_id):
        return sum(r["user_id"] == user_id for r in self.rows.values())

    async def get_flow(self, user_id, flow_id):
        return self._mine(user_id, flow_id)

    def _check_slug(self, user_id, slug, flow_id=None):
        if any(r["user_id"] == user_id and r["slug"] == slug and r["id"] != flow_id for r in self.rows.values()):
            raise flow_repository.SlugTaken(slug)

    async def create_flow(self, user_id, name, slug, description, definition):
        self._check_slug(user_id, slug)
        now = datetime.now(UTC)
        row = {"id": str(uuid.uuid4()), "user_id": user_id, "name": name, "slug": slug, "description": description,
               "definition": definition, "version": 1, "enabled": True, "created_at": now, "updated_at": now}
        self.rows[row["id"]] = row
        return row

    async def update_flow(self, user_id, flow_id, name, slug, description, definition, enabled):
        row = self._mine(user_id, flow_id)
        if row is None:
            return None
        self._check_slug(user_id, slug, flow_id)
        row.update(name=name, slug=slug, description=description, definition=definition, enabled=enabled,
                   version=row["version"] + 1)
        return row

    async def set_enabled(self, user_id, flow_id, enabled):
        row = self._mine(user_id, flow_id)
        if row is not None:
            row["enabled"] = enabled
        return row

    async def delete_flow(self, user_id, flow_id):
        return self.rows.pop(flow_id, None) is not None if self._mine(user_id, flow_id) else False


@pytest.fixture
def repo(monkeypatch):
    r = MemoryRepo()
    for name in ("list_flows", "count_flows", "get_flow", "create_flow", "update_flow", "delete_flow", "set_enabled"):
        monkeypatch.setattr(flow_repository, name, getattr(r, name))
    monkeypatch.setattr(catalog, "flow_tools", lambda: TOOLS)
    fake_llm = EchoModel()
    real_compile = compiler.compile_flow
    monkeypatch.setattr("app.flows.runner.compile_flow",
                        lambda f, llm=None, tools=None: real_compile(f, llm=fake_llm, tools=TOOLS))
    flow_fakes.tool_calls.clear()
    return r


def client_for(user: str) -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(flows_endpoint.router, prefix="/flows")
    app.dependency_overrides[get_current_user] = lambda: type("U", (), {"sub": user})()
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def create(c, name="Tóm tắt tin tức", definition=None):
    return await c.post("/flows", json={"name": name, "description": "news", "definition": definition or news_flow()})


async def test_a_valid_flow_is_saved_with_a_slug_made_from_its_name(repo):
    async with client_for("u1") as c:
        r = await create(c)
        assert r.status_code == 201, r.text
        assert r.json()["slug"] == "tom_tat_tin_tuc" and r.json()["version"] == 1
        listed = (await c.get("/flows")).json()
        assert [f["name"] for f in listed] == ["Tóm tắt tin tức"]


async def test_an_invalid_flow_is_not_saved_and_the_problems_name_their_nodes(repo):
    d = news_flow()
    d["nodes"][2]["config"]["prompt"] = "{{ghost.text}}"
    async with client_for("u1") as c:
        r = await create(c, definition=d)
        assert r.status_code == 400
        problems = r.json()["detail"]["problems"]
        assert any(p["node"] == "sum" and "ghost" in p["message"] for p in problems)
        assert repo.rows == {}


async def test_validate_checks_without_saving(repo):
    d = news_flow()
    d["edges"].pop()
    async with client_for("u1") as c:
        r = (await c.post("/flows/validate", json={"definition": d})).json()
    assert r["valid"] is False and repo.rows == {}


async def test_two_flows_cannot_share_a_name(repo):
    async with client_for("u1") as c:
        await create(c, "Tin tức")
        assert (await create(c, "tin  TỨC")).status_code == 409   # same slug


async def test_another_users_flow_is_not_there_for_them(repo):
    async with client_for("owner") as c:
        flow_id = (await create(c)).json()["id"]
    async with client_for("someone_else") as c:
        assert (await c.get(f"/flows/{flow_id}")).status_code == 404
        assert (await c.put(f"/flows/{flow_id}", json={"name": "x", "definition": news_flow()})).status_code == 404
        assert (await c.delete(f"/flows/{flow_id}")).status_code == 404
        assert (await c.post(f"/flows/{flow_id}/run", json={"inputs": {"topic": "t"}})).status_code == 404
        assert (await c.get("/flows")).json() == []
    assert flow_id in repo.rows


async def test_a_flow_id_that_is_not_a_uuid_is_just_not_found(repo):
    async with client_for("u1") as c:
        assert (await c.get("/flows/not-a-uuid")).status_code == 404


async def test_saving_again_bumps_the_version(repo):
    async with client_for("u1") as c:
        flow_id = (await create(c)).json()["id"]
        r = await c.put(f"/flows/{flow_id}", json={"name": "Tin mới", "definition": news_flow(), "enabled": False})
        assert r.status_code == 200 and r.json()["version"] == 2 and r.json()["slug"] == "tin_moi"
        assert r.json()["enabled"] is False


async def test_delete(repo):
    async with client_for("u1") as c:
        flow_id = (await create(c)).json()["id"]
        assert (await c.delete(f"/flows/{flow_id}")).status_code == 204
        assert (await c.get(f"/flows/{flow_id}")).status_code == 404


async def test_a_user_has_a_limit_on_flows(repo, monkeypatch):
    monkeypatch.setattr(flow_service.settings, "FLOWS_MAX_PER_USER", 1)
    async with client_for("u1") as c:
        await create(c, "a")
        r = await create(c, "b")
    assert r.status_code == 400 and "at most" in r.json()["detail"]


async def test_the_catalog_offers_node_types_tools_and_models(repo):
    async with client_for("u1") as c:
        cat = (await c.get("/flows/catalog")).json()
    assert set(cat["node_types"]) == {"input", "llm", "tool", "output"}
    search = next(t for t in cat["tools"] if t["name"] == "web_search")
    assert [a["name"] for a in search["args"]] == ["query"]           # `label` is filled in by the flow itself
    assert cat["models"]


async def test_a_test_run_streams_node_events_tokens_and_the_result(repo):
    async with client_for("u1") as c:
        flow_id = (await create(c)).json()["id"]
        async with c.stream("POST", f"/flows/{flow_id}/run", json={"inputs": {"topic": "Huế"}}) as r:
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
            events = [json.loads(line[6:]) async for line in r.aiter_lines() if line.startswith("data: ")]
    assert [e["type"] for e in events][-2:] == ["result", "done"]
    assert events[-2]["text"].startswith("Summary: [deepseek/deepseek-v4-flash] Summarise: results for Huế news")
    assert {(e["node"], e["phase"]) for e in events if e["type"] == "node"} >= {("s", "end"), ("out", "end")}
    assert flow_fakes.tool_calls[0]["user_id"] == "u1" and flow_fakes.tool_calls[0]["sandbox"] is False


async def test_bad_inputs_or_model_are_refused_before_the_run_starts(repo):
    async with client_for("u1") as c:
        flow_id = (await create(c)).json()["id"]
        r = await c.post(f"/flows/{flow_id}/run", json={"inputs": {}})
        assert r.status_code == 400 and "required" in r.json()["detail"]
        r = await c.post(f"/flows/{flow_id}/run", json={"inputs": {"topic": "t"}, "model": "made/up"})
        assert r.status_code == 400 and "Unknown model" in r.json()["detail"]
    assert flow_fakes.tool_calls == []


async def test_flows_can_be_turned_off(repo, monkeypatch):
    monkeypatch.setattr(flow_service.settings, "FLOWS_ENABLED", False)
    async with client_for("u1") as c:
        assert (await c.get("/flows")).status_code == 404
        assert (await create(c)).status_code == 404


def test_slugs():
    assert flow_service.slugify("Đọc báo & tóm tắt!") == "doc_bao_tom_tat"
    assert flow_service.slugify("???") == "flow"


async def test_a_flow_is_switched_off_and_on_without_a_new_version(repo):
    async with client_for("u1") as c:
        flow_id = (await create(c)).json()["id"]
        r = await c.patch(f"/flows/{flow_id}", json={"enabled": False})
        assert r.status_code == 200 and r.json()["enabled"] is False and r.json()["version"] == 1
    async with client_for("someone_else") as c:
        assert (await c.patch(f"/flows/{flow_id}", json={"enabled": True})).status_code == 404
    assert repo.rows[flow_id]["enabled"] is False
