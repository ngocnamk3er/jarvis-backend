"""Department document spaces: who can read and change which files, through the Files API and the agent's tools.

A department is a Keycloak group under /departments; its members read its files, its editors subgroup also changes
them. Every check is the backend's: the file service is only ever called with an owner id the user may use."""

from types import SimpleNamespace
from urllib.parse import unquote

import httpx
import pytest
from app.agents import retrieval
from app.agents.tools import files as tools
from app.api import deps
from app.api.deps import get_current_user
from app.api.v1.endpoints import files as files_endpoint
from app.core import spaces
from fastapi import FastAPI

ACCOUNTANT = ("/departments/Kế toán/editors", "/departments/Nhân sự")   # edits Kế toán, reads Nhân sự
CLERK = ("/departments/Kế toán",)                                     # reads Kế toán only


# --- reading the token's groups -------------------------------------------------------------------------------


def test_departments_come_from_the_groups_under_the_departments_group():
    groups = (*ACCOUNTANT, "/admins", "/departmentsX/Y", "/departments", "/departments/Kế toán/team-a")
    assert spaces.departments(groups) == [("Kế toán", True), ("Nhân sự", False)]
    assert spaces.departments(("/departments/Pháp chế/team-a",)) == [("Pháp chế", False)]   # any other subgroup: a member
    assert spaces.departments(()) == []


def test_each_department_is_a_space_owned_by_a_plain_ascii_id():
    got = spaces.spaces_for("u1", ACCOUNTANT)
    assert [(s.id, s.name, s.owner, s.can_edit) for s in got] == [
        ("personal", "My files", "u1", True),
        ("dept:ke-toan", "Kế toán", "dept:ke-toan", True),
        ("dept:nhan-su", "Nhân sự", "dept:nhan-su", False),
    ]
    assert spaces.slug("Đào tạo & Phát triển") == "dao-tao-phat-trien"
    assert spaces.slug("研发").startswith("x") and spaces.slug("研发") != spaces.slug("销售")


def test_who_may_use_a_space():
    assert spaces.resolve("u1", (), None).owner == "u1"
    assert spaces.resolve("u1", (), "personal", write=True).owner == "u1"
    assert spaces.resolve("u1", CLERK, "dept:ke-toan").owner == "dept:ke-toan"
    assert spaces.resolve("u1", ACCOUNTANT, "dept:ke-toan", write=True).owner == "dept:ke-toan"
    with pytest.raises(spaces.SpaceError) as e:
        spaces.resolve("u1", CLERK, "dept:ke-toan", write=True)      # a member, not an editor
    assert e.value.status == 403
    for groups, sid in ((CLERK, "dept:nhan-su"), ((), "dept:ke-toan"), (ACCOUNTANT, "dept:nope"), (ACCOUNTANT, "u2")):
        with pytest.raises(spaces.SpaceError) as e:
            spaces.resolve("u1", groups, sid)
        assert e.value.status == 404                                 # not in it, or no such space: told apart by no one


async def test_the_current_user_carries_the_tokens_groups(monkeypatch):
    creds = SimpleNamespace(credentials="t")
    monkeypatch.setattr(deps, "verify_token", lambda t: {"sub": "u1", "groups": [*CLERK, 5]})
    assert (await deps.get_current_user(creds)).groups == CLERK
    monkeypatch.setattr(deps, "verify_token", lambda t: {"sub": "u1"})    # a user in no group has no claim at all
    assert (await deps.get_current_user(creds)).groups == ()


# --- the Files API --------------------------------------------------------------------------------------------


class FakeFiles:
    """file_service, keeping each owner's nodes apart the way the file service does."""

    def __init__(self):
        self.nodes = {"n-own": ("u1", "mine.txt"), "n-kt": ("dept:ke-toan", "Kế toán quý 3.pdf"),
                      "n-ns": ("dept:nhan-su", "noi-quy.pdf")}
        self.calls = []

    async def list_tree(self, owner, path):
        self.calls.append(("list", owner, path))
        return []

    async def get_node(self, owner, node_id):
        hit = self.nodes.get(node_id)
        return None if hit is None or hit[0] != owner else {"id": node_id, "name": hit[1]}

    async def get_content(self, owner, node_id):
        assert self.nodes[node_id][0] == owner
        return b"%PDF", "application/pdf", self.nodes[node_id][1]

    async def create_folder(self, owner, parent, name):
        self.calls.append(("folder", owner, name))
        return _node(owner, name)

    async def upload_file(self, owner, parent, filename, content, mime):
        self.calls.append(("upload", owner, filename))
        return _node(owner, filename)

    async def rename_or_move(self, owner, node_id, name, parent):
        self.calls.append(("rename", owner, node_id))
        return _node(owner, name)

    async def delete_node(self, owner, node_id):
        self.calls.append(("delete", owner, node_id))


def _node(owner, name):
    return {"id": "new", "user_id": owner, "parent_id": None, "name": name, "type": "file",
            "indexing_status": "pending", "created_at": "2026-11-28T00:00:00Z", "updated_at": "2026-11-28T00:00:00Z"}


@pytest.fixture
def api(monkeypatch):
    fake = FakeFiles()
    for name in ("list_tree", "get_node", "get_content", "create_folder", "upload_file", "rename_or_move", "delete_node"):
        monkeypatch.setattr(files_endpoint.file_service, name, getattr(fake, name))
    who = {"groups": ACCOUNTANT}
    app = FastAPI()
    app.include_router(files_endpoint.router, prefix="/files")
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(sub="u1", groups=who["groups"])
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
    client.fake, client.who = fake, who
    return client


async def test_the_spaces_a_user_can_open(api):
    r = await api.get("/files/spaces")
    assert r.json() == [
        {"id": "personal", "name": "My files", "kind": "personal", "can_edit": True},
        {"id": "dept:ke-toan", "name": "Kế toán", "kind": "department", "can_edit": True},
        {"id": "dept:nhan-su", "name": "Nhân sự", "kind": "department", "can_edit": False},
    ]
    api.who["groups"] = ()
    assert [s["id"] for s in (await api.get("/files/spaces")).json()] == ["personal"]


async def test_a_member_reads_a_department_but_only_an_editor_changes_it(api):
    assert (await api.get("/files/tree", params={"space": "dept:nhan-su", "path": "/"})).status_code == 200
    assert api.fake.calls[-1] == ("list", "dept:nhan-su", "/")
    before = list(api.fake.calls)
    for r in (
        await api.post("/files/upload", params={"space": "dept:nhan-su"}, files={"file": ("a.txt", b"x")}),
        await api.post("/files/folders", params={"space": "dept:nhan-su"}, json={"parent_path": "/", "name": "d"}),
        await api.patch("/files/nodes/n-ns", params={"space": "dept:nhan-su"}, json={"name": "b.pdf"}),
        await api.delete("/files/nodes/n-ns", params={"space": "dept:nhan-su"}),
    ):
        assert r.status_code == 403 and "not change" in r.json()["detail"]
    assert api.fake.calls == before                                  # the file service was never asked

    r = await api.post("/files/upload", params={"space": "dept:ke-toan"}, files={"file": ("a.txt", b"x")})
    assert r.status_code == 200 and api.fake.calls[-1] == ("upload", "dept:ke-toan", "a.txt")
    await api.delete("/files/nodes/n-kt", params={"space": "dept:ke-toan"})
    assert api.fake.calls[-1] == ("delete", "dept:ke-toan", "n-kt")


async def test_someone_outside_a_department_gets_nothing_of_it(api):
    api.who["groups"] = ()
    for r in (
        await api.get("/files/tree", params={"space": "dept:ke-toan"}),
        await api.post("/files/upload", params={"space": "dept:ke-toan"}, files={"file": ("a.txt", b"x")}),
        await api.get("/files/nodes/n-kt/content", params={"space": "dept:ke-toan"}),
        await api.get("/files/nodes/n-kt/content"),                 # not by its id alone either
    ):
        assert r.status_code == 404
    assert (await api.get("/files/nodes/n-kt")).json() is None
    assert api.fake.calls == []


async def test_without_a_space_your_own_files_are_meant(api):
    await api.post("/files/upload", files={"file": ("a.txt", b"x")})
    assert api.fake.calls[-1] == ("upload", "u1", "a.txt")


async def test_a_citation_that_carries_only_the_file_id_still_opens(api):
    r = await api.get("/files/nodes/n-kt/content")                  # saved before spaces existed
    assert r.status_code == 200 and r.content == b"%PDF"
    assert (await api.get("/files/nodes/n-kt", params={"space": "dept:nhan-su"})).json() is None   # wrong space


async def test_a_vietnamese_file_name_survives_the_download_header(api):
    r = await api.get("/files/nodes/n-kt/content", params={"space": "dept:ke-toan"})
    header = r.headers["content-disposition"]
    assert header.isascii()
    assert unquote(header.split("filename*=UTF-8''", 1)[1]) == "Kế toán quý 3.pdf"


async def test_the_file_client_reads_the_real_name_back():
    from app.clients import file_client

    header = files_endpoint.content_disposition("Kế toán quý 3.pdf")
    resp = httpx.Response(200, content=b"x", headers={"content-type": "application/pdf", "content-disposition": header},
                          request=httpx.Request("GET", "http://files/x"))

    class Client:
        async def get(self, *a, **k):
            return resp

    old = file_client._client
    file_client._client = Client()
    try:
        assert (await file_client.get_content("u1", "n"))[2] == "Kế toán quý 3.pdf"
    finally:
        file_client._client = old


# --- the agent's tools ----------------------------------------------------------------------------------------


def _config(groups=ACCOUNTANT):
    return {"configurable": {"user_id": "u1", "thread_id": "t", "file_spaces": spaces.for_agent(groups)}}


async def test_the_workspace_root_shows_the_departments(monkeypatch):
    seen = []

    async def list_tree(owner, path):
        seen.append((owner, path))
        return [{"type": "file", "name": "mine.txt", "size_bytes": 10}] if owner == "u1" else []

    monkeypatch.setattr(tools.file_client, "list_tree", list_tree)
    out = await tools.list_files.ainvoke({"path": "/"}, config=_config())
    assert "🏢 @Kế toán/" in out and "🏢 @Nhân sự/" in out and "mine.txt" in out
    out = await tools.list_files.ainvoke({"path": "/"}, config=_config(()))
    assert "@" not in out
    await tools.list_files.ainvoke({"path": "/@Kế toán"}, config=_config())
    await tools.list_files.ainvoke({"path": "@ke-toan/2024"}, config=_config())       # by slug, no leading slash
    assert seen[-2:] == [("dept:ke-toan", "/"), ("dept:ke-toan", "/2024")]


async def test_a_department_the_user_is_not_in_is_refused_by_name(monkeypatch):
    async def resolve_path(owner, path):
        raise AssertionError("the file service must not be asked")

    monkeypatch.setattr(tools.file_client, "resolve_path", resolve_path)
    monkeypatch.setattr(tools.file_client, "list_tree", resolve_path)
    out = await tools.read_file.ainvoke({"path": "/@Nhân sự/luong.xlsx"}, config=_config(CLERK))
    assert out.startswith("Error: no department 'Nhân sự'") and "/@Kế toán" in out
    out = await tools.list_files.ainvoke({"path": "/@Kế toán"}, config=_config(()))
    assert "in no department" in out


async def test_reading_and_copying_a_department_file(monkeypatch):
    seen = []

    async def resolve_path(owner, path):
        seen.append((owner, path))
        return {"id": "n-kt", "type": "file", "name": "quy-che.txt", "indexing_status": "done",
                "extracted_text": "Điều 1.\nĐiều 2.", "size_bytes": 8}

    async def get_content(owner, node_id):
        seen.append((owner, node_id))
        return b"x", "text/plain", "quy-che.txt"

    async def write_file(thread_id, name, content):
        seen.append(("sandbox", name))

    monkeypatch.setattr(tools.file_client, "resolve_path", resolve_path)
    monkeypatch.setattr(tools.file_client, "get_content", get_content)
    monkeypatch.setattr(tools.sandbox_manager, "write_file", write_file)
    out = await tools.read_file.ainvoke({"path": "/@Kế toán/quy-che.txt", "offset": 1}, config=_config(CLERK))
    assert "Điều 1." in out
    await tools.fetch_file.ainvoke({"path": "/@Kế toán/quy-che.txt"}, config=_config(CLERK))
    assert seen == [("dept:ke-toan", "/quy-che.txt"), ("dept:ke-toan", "/quy-che.txt"), ("dept:ke-toan", "n-kt"),
                    ("sandbox", "quy-che.txt")]


def _hit(file_id, path, score, text="..."):
    return {"file_id": file_id, "path": path, "chunk_text": text, "score": score}


async def test_search_covers_the_users_files_and_their_departments(monkeypatch):
    asked = []

    async def search_vector(owner, query, top_k):
        asked.append(owner)
        return {"u1": [_hit("a", "/notes.md", 0.02)],
                "dept:ke-toan": [_hit("b", "/quy-che.pdf", 0.03), _hit("c", "/cu.pdf", 0.01)],
                "dept:nhan-su": [_hit("d", "/noi-quy.pdf", 0.015)]}[owner]

    monkeypatch.setattr(tools.file_client, "search_vector", search_vector)
    monkeypatch.setattr(tools.retrieval, "enabled", lambda: False)
    msg = await tools.search_files.ainvoke(
        {"type": "tool_call", "id": "1", "name": "workspace_search_files", "args": {"query": "quy chế", "top_k": 3}},
        config=_config())
    assert sorted(asked) == ["dept:ke-toan", "dept:nhan-su", "u1"]
    paths = [c["path"] for c in msg.artifact["citations"]]
    assert paths == ["/@Kế toán/quy-che.pdf", "/notes.md", "/@Nhân sự/noi-quy.pdf"]           # by score, top 3
    assert [c["space"] for c in msg.artifact["citations"]] == ["dept:ke-toan", "personal", "dept:nhan-su"]
    assert "/@Kế toán/quy-che.pdf" in msg.content

    asked.clear()
    await tools.search_files.ainvoke(
        {"type": "tool_call", "id": "1", "name": "workspace_search_files", "args": {"query": "q"}}, config=_config(()))
    assert asked == ["u1"]                                            # in no department: only their own


async def test_exact_search_takes_each_spaces_hits_in_turn(monkeypatch):
    async def search_fulltext(owner, query, top_k):
        return {"u1": [_hit("a", "/a", 1.0), _hit("b", "/b", 1.0)], "dept:ke-toan": [_hit("c", "/c", 1.0)],
                "dept:nhan-su": []}[owner]

    monkeypatch.setattr(tools.file_client, "search_fulltext", search_fulltext)
    msg = await tools.search_files.ainvoke(
        {"type": "tool_call", "id": "1", "name": "workspace_search_files",
         "args": {"query": "HD-01", "exact": True, "top_k": 3}}, config=_config())
    assert [c["path"] for c in msg.artifact["citations"]] == ["/a", "/@Kế toán/c", "/b"]


async def test_agentic_retrieval_searches_every_owner_and_says_whose(monkeypatch):
    async def search_vector(owner, query, top_k):
        return [_hit(f"{owner}-{query}", "/x", 0.5, text=f"{owner} {query}")]

    monkeypatch.setattr(retrieval.file_client, "search_vector", search_vector)
    got = await retrieval._retrieve(["u1", "dept:ke-toan"], ["q1", "q2"], 5)
    assert sorted((c["owner"], c["file_id"]) for c in got) == [
        ("dept:ke-toan", "dept:ke-toan-q1"), ("dept:ke-toan", "dept:ke-toan-q2"), ("u1", "u1-q1"), ("u1", "u1-q2")]


# --- the run's config carries the spaces ------------------------------------------------------------------------


def test_chat_and_research_runs_carry_the_departments():
    from app.agents.async_research import _INHERITED
    from app.services.chat_service import _make_config

    assert _make_config("t", "u1", file_spaces=spaces.for_agent(CLERK))["configurable"]["file_spaces"] == [
        {"id": "dept:ke-toan", "name": "Kế toán", "owner": "dept:ke-toan", "can_edit": False}]
    assert _make_config("t", "u1")["configurable"]["file_spaces"] == []
    assert "file_spaces" in _INHERITED


async def test_the_chat_endpoint_hands_the_tokens_departments_to_the_run(monkeypatch):
    from app.api.v1.endpoints import chat as chat_endpoint

    seen = {}

    async def stream(*args, file_spaces=None, **kw):
        seen["spaces"] = file_spaces
        yield "data: {}\n\n"

    async def touch(*a, **k):
        pass

    async def owns(thread_id, user):
        return SimpleNamespace(user_id="u1")

    class Lease:
        def release(self):
            pass

    async def lock(thread_id):
        return Lease()

    async def guarded(lease, gen):
        async for chunk in gen:
            yield chunk

    monkeypatch.setattr(chat_endpoint, "_check_owns_thread", owns)
    monkeypatch.setattr(chat_endpoint, "_lock_conversation", lock)
    monkeypatch.setattr(chat_endpoint.conversation_client, "touch_conversation", touch)
    monkeypatch.setattr(chat_endpoint.chat_service, "stream", stream)
    monkeypatch.setattr(chat_endpoint.chat_service, "guarded", guarded)
    app = FastAPI()
    app.state.graph = object()
    app.include_router(chat_endpoint.router, prefix="/chat")
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(sub="u1", groups=CLERK)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/chat/stream", json={"thread_id": "t", "content": "hi"})
    assert r.status_code == 200
    assert seen["spaces"] == spaces.for_agent(CLERK)
