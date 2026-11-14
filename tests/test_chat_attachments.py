"""Files attached to a chat message land in the conversation's sandbox, and only there."""

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from app.api.deps import get_current_user
from app.api.v1.endpoints import chat as chat_endpoint
from app.agents.tools import sandbox_manager


@pytest.fixture
def client(monkeypatch):
    sandbox: dict[str, bytes] = {}
    calls = {"owner": []}

    async def owns(thread_id, user):
        calls["owner"].append(thread_id)
        if thread_id != "mine":
            raise HTTPException(status_code=404, detail="Conversation not found")

    async def exec_bash(thread_id, command):
        assert command == "ls -1A"
        return {"stdout": "\n".join(sandbox), "exit_code": 0}

    async def write_file(thread_id, name, content):
        sandbox[name] = content

    monkeypatch.setattr(chat_endpoint, "_check_owns_thread", owns)
    monkeypatch.setattr(sandbox_manager, "exec_bash", exec_bash)
    monkeypatch.setattr(sandbox_manager, "write_file", write_file)
    app = FastAPI()
    app.include_router(chat_endpoint.router, prefix="/chat")
    app.dependency_overrides[get_current_user] = lambda: type("U", (), {"sub": "u1"})()
    c = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
    c.sandbox, c.calls = sandbox, calls
    return c


async def upload(client, name="report.pdf", data=b"%PDF-1.4 hello", thread="mine", ctype="application/pdf"):
    return await client.post("/chat/sandbox-upload", data={"thread_id": thread}, files={"file": (name, data, ctype)})


async def test_an_attached_file_goes_into_the_sandbox_and_the_reply_says_where(client):
    r = await upload(client)
    assert r.status_code == 200 and r.json() == {"name": "report.pdf", "size": 14, "mime": "application/pdf"}
    assert client.sandbox == {"report.pdf": b"%PDF-1.4 hello"}


async def test_a_second_file_with_the_same_name_does_not_overwrite_the_first(client):
    await upload(client, data=b"first")
    r = await upload(client, data=b"second")
    assert r.json()["name"] == "report (2).pdf"
    assert (await upload(client, data=b"third")).json()["name"] == "report (3).pdf"
    assert client.sandbox["report.pdf"] == b"first" and client.sandbox["report (2).pdf"] == b"second"


async def test_a_name_without_an_extension_and_an_unsafe_name_are_handled(client):
    await upload(client, name="notes", data=b"a")
    assert (await upload(client, name="notes", data=b"b")).json()["name"] == "notes (2)"
    r = await upload(client, name="../../etc/passwd; rm -rf x.txt", data=b"c")
    assert r.json()["name"].endswith(".txt") and "/" not in r.json()["name"] and ";" not in r.json()["name"]


async def test_someone_elses_conversation_is_refused_and_nothing_is_written(client):
    r = await upload(client, thread="theirs")
    assert r.status_code == 404 and client.sandbox == {}


async def test_a_file_over_the_limit_is_refused(client, monkeypatch):
    monkeypatch.setattr(chat_endpoint, "MAX_ATTACHMENT_BYTES", 10)
    r = await upload(client, data=b"x" * 11)
    assert r.status_code == 413 and client.sandbox == {}


async def test_the_type_is_guessed_from_the_name_when_the_browser_sends_none(client):
    r = await upload(client, name="photo.png", data=b"x", ctype="application/octet-stream")
    assert r.json()["mime"] == "image/png"
