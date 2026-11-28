"""The sandbox file browser: listing, deleting, renaming, making folders and saving to Files. The sandbox is a temp
directory and its commands really run there with bash, so the shell commands themselves are what is tested."""

import asyncio
import os
from types import SimpleNamespace

import httpx
import pytest
from app.api.deps import get_current_user
from app.api.v1.endpoints import chat as chat_endpoint
from app.api.v1.endpoints import sandbox_files as files_endpoint
from app.services import sandbox_files as sf
from fastapi import FastAPI, HTTPException


@pytest.fixture
def box(tmp_path, monkeypatch):
    """A sandbox working directory with the runtime's own files in it, as the stock runtime has."""
    (tmp_path / "main.py").write_text("server")
    (tmp_path / "requirements.txt").write_text("x")
    (tmp_path / "__pycache__").mkdir()
    state = SimpleNamespace(value="running", execs=[], uploads=[], conflicts=0)

    async def exec_bash(thread_id, command):
        state.execs.append(command)
        proc = await asyncio.create_subprocess_exec("bash", "-c", command, cwd=tmp_path,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate()
        return {"stdout": out.decode(), "stderr": err.decode(), "exit_code": proc.returncode, "timed_out": False}

    async def sandbox_state(thread_id):
        return state.value

    async def read_file(thread_id, name):
        return (tmp_path / name).read_bytes(), "application/octet-stream", name.rsplit("/", 1)[-1]

    async def upload_file(user_id, parent, name, content, mime):
        if state.conflicts:
            state.conflicts -= 1
            raise httpx.HTTPStatusError("taken", request=httpx.Request("POST", "http://f"),
                                        response=httpx.Response(409))
        state.uploads.append((user_id, parent, name, content, mime))
        return {"name": name, "path": f"/{name}"}

    monkeypatch.setattr(sf.sandbox_manager, "exec_bash", exec_bash)
    monkeypatch.setattr(sf.sandbox_manager, "sandbox_state", sandbox_state)
    monkeypatch.setattr(sf.sandbox_manager, "read_file", read_file)
    monkeypatch.setattr(sf.file_client, "upload_file", upload_file)
    state.root = tmp_path
    return state


async def test_a_folder_is_listed_folders_first_without_the_runtimes_own_files(box):
    (box.root / "report.csv").write_text("a,b\n1,2\n")
    (box.root / "charts").mkdir()
    (box.root / "Báo cáo tháng 11.docx").write_bytes(b"x" * 10)
    (box.root / "odd\nname\twith tabs").write_text("")
    out = await sf.list_dir("t", "")
    assert out["state"] == "running" and out["truncated"] is False
    assert [(e["name"], e["type"]) for e in out["entries"]] == [
        ("charts", "dir"), ("Báo cáo tháng 11.docx", "file"), ("odd\nname\twith tabs", "file"), ("report.csv", "file")]
    report = out["entries"][-1]
    assert report["size"] == 8 and report["path"] == "report.csv" and report["modified"].endswith("+00:00")
    assert out["entries"][0]["size"] is None


async def test_the_runtimes_file_names_are_only_hidden_at_the_top(box):
    (box.root / "proj").mkdir()
    (box.root / "proj" / "main.py").write_text("mine")
    out = await sf.list_dir("t", "proj")
    assert [e["path"] for e in out["entries"]] == ["proj/main.py"]


async def test_looking_never_starts_or_wakes_a_sandbox(box):
    for state in ("none", "paused"):
        box.value = state
        out = await sf.list_dir("t", "")
        assert out == {"state": state, "path": "", "entries": [], "truncated": False}
    assert box.execs == []
    box.value = "paused"
    (box.root / "kept.txt").write_text("x")
    assert [e["name"] for e in (await sf.list_dir("t", "", wake=True))["entries"]] == ["kept.txt"]


async def test_a_long_folder_is_cut_and_says_so(box, monkeypatch):
    monkeypatch.setattr(sf, "MAX_ENTRIES", 3)
    for i in range(6):
        (box.root / f"f{i}.txt").write_text("x")
    out = await sf.list_dir("t", "")
    assert len(out["entries"]) == 3 and out["truncated"] is True


@pytest.mark.parametrize("path, ok", [
    ("/workspace/a/b.txt", "a/b.txt"), ("./a/b/", "a/b"), ("a//b", "a/b"),
    ("/etc/passwd", None), ("../x", None), ("a/../../b", None), ("main.py", None), ("__pycache__/x.pyc", None),
])
def test_paths_stay_inside_the_working_directory_and_off_the_runtimes_files(path, ok):
    if ok is None:
        with pytest.raises(sf.SandboxPathError):
            sf.clean(path)
    else:
        assert sf.clean(path) == ok


async def test_a_missing_folder_is_not_found(box):
    with pytest.raises(sf.SandboxFileError) as e:
        await sf.list_dir("t", "nope")
    assert e.value.status == 404


async def test_delete_removes_a_file_or_a_whole_folder(box):
    (box.root / "d").mkdir()
    (box.root / "d" / "x.txt").write_text("x")
    (box.root / "f.txt").write_text("x")
    await sf.delete("t", "d")
    await sf.delete("t", "f.txt")
    assert sorted(os.listdir(box.root)) == ["__pycache__", "main.py", "requirements.txt"]
    with pytest.raises(sf.SandboxFileError) as e:
        await sf.delete("t", "f.txt")
    assert e.value.status == 404
    for bad in ("", "main.py", "../x"):
        with pytest.raises(sf.SandboxPathError):
            await sf.delete("t", bad)
    assert (box.root / "main.py").exists()


async def test_rename_keeps_the_folder_and_refuses_to_overwrite(box):
    (box.root / "v1.2").mkdir()
    (box.root / "v1.2" / "a.txt").write_text("a")
    (box.root / "v1.2" / "b.txt").write_text("b")
    assert await sf.rename("t", "v1.2/a.txt", "c.txt") == "v1.2/c.txt"          # a dot in the folder's name survives
    assert (box.root / "v1.2" / "c.txt").read_text() == "a"
    with pytest.raises(sf.SandboxFileError) as e:
        await sf.rename("t", "v1.2/c.txt", "b.txt")
    assert e.value.status == 409 and (box.root / "v1.2" / "b.txt").read_text() == "b"
    for bad_name in ("x/y", "..", ""):
        with pytest.raises(sf.SandboxPathError):
            await sf.rename("t", "v1.2/c.txt", bad_name)
    (box.root / "top.txt").write_text("t")
    with pytest.raises(sf.SandboxPathError):
        await sf.rename("t", "top.txt", "main.py")                                # onto the runtime's own file


async def test_make_folder(box):
    assert await sf.make_folder("t", "", "data") == "data"
    assert await sf.make_folder("t", "data", "raw") == "data/raw"
    assert (box.root / "data" / "raw").is_dir()
    with pytest.raises(sf.SandboxFileError) as e:
        await sf.make_folder("t", "", "data")
    assert e.value.status == 409
    with pytest.raises(sf.SandboxFileError) as e:
        await sf.make_folder("t", "missing", "x")
    assert e.value.status == 404


async def test_save_to_files_copies_the_bytes_and_finds_a_free_name(box):
    (box.root / "out").mkdir()
    (box.root / "out" / "chart.png").write_bytes(b"\x89PNG data")
    box.conflicts = 1                                                             # "chart.png" is taken in Files
    saved = await sf.save_to_files("t", "out/chart.png", "u1")
    assert saved == {"name": "chart (2).png", "path": "/chart (2).png"}
    assert box.uploads == [("u1", "/", "chart (2).png", b"\x89PNG data", "image/png")]


async def test_only_a_file_within_the_size_limit_can_be_saved(box, monkeypatch):
    (box.root / "d").mkdir()
    with pytest.raises(sf.SandboxFileError) as e:
        await sf.save_to_files("t", "d", "u1")
    assert e.value.status == 404
    monkeypatch.setattr(sf, "MAX_SAVE_BYTES", 4)
    (box.root / "big.bin").write_bytes(b"12345")
    with pytest.raises(sf.SandboxFileError) as e:
        await sf.save_to_files("t", "big.bin", "u1")
    assert e.value.status == 413 and box.uploads == []


async def test_changes_need_a_sandbox_to_exist(box):
    box.value = "none"
    for call in (sf.delete("t", "x"), sf.rename("t", "x", "y"), sf.save_to_files("t", "x", "u1")):
        with pytest.raises(sf.SandboxFileError) as e:
            await call
        assert e.value.status == 404
    assert box.execs == []


# ---- the API ----------------------------------------------------------------------------------------------------------

def client_for(user: str, monkeypatch) -> httpx.AsyncClient:
    async def check(thread_id, u):
        if u.sub != "owner":
            raise HTTPException(status_code=404, detail="Conversation not found")

    monkeypatch.setattr(files_endpoint, "_check_owns_thread", check)
    monkeypatch.setattr(chat_endpoint, "_check_owns_thread", check)
    app = FastAPI()
    app.include_router(files_endpoint.router, prefix="/chat/sandbox-files")
    app.include_router(chat_endpoint.router, prefix="/chat")
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(sub=user)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def test_the_api_lists_changes_and_maps_errors(box, monkeypatch):
    (box.root / "a.txt").write_text("a")
    async with client_for("owner", monkeypatch) as c:
        r = await c.get("/chat/sandbox-files", params={"thread_id": "t"})
        assert r.status_code == 200 and [e["name"] for e in r.json()["entries"]] == ["a.txt"]
        assert (await c.get("/chat/sandbox-files", params={"thread_id": "t", "path": "../x"})).status_code == 400
        assert (await c.post("/chat/sandbox-files/folders", json={"thread_id": "t", "name": "d"})).status_code == 201
        r = await c.post("/chat/sandbox-files/rename", json={"thread_id": "t", "path": "a.txt", "new_name": "d"})
        assert r.status_code == 409
        r = await c.post("/chat/sandbox-files/rename", json={"thread_id": "t", "path": "a.txt", "new_name": "b.txt"})
        assert r.json() == {"path": "b.txt"}
        assert (await c.delete("/chat/sandbox-files", params={"thread_id": "t", "path": "b.txt"})).status_code == 204
        assert (await c.delete("/chat/sandbox-files", params={"thread_id": "t", "path": "main.py"})).status_code == 400


async def test_someone_elses_conversation_is_not_reachable(box, monkeypatch):
    (box.root / "secret.txt").write_text("s")
    async with client_for("stranger", monkeypatch) as c:
        assert (await c.get("/chat/sandbox-files", params={"thread_id": "t"})).status_code == 404
        assert (await c.delete("/chat/sandbox-files", params={"thread_id": "t", "path": "secret.txt"})).status_code == 404
        r = await c.post("/chat/sandbox-files/save", json={"thread_id": "t", "path": "secret.txt"})
        assert r.status_code == 404
    assert (box.root / "secret.txt").exists() and box.execs == []


async def test_an_upload_can_go_into_a_folder(box, monkeypatch):
    (box.root / "inbox").mkdir()
    (box.root / "inbox" / "note.txt").write_text("old")

    async def write_file(thread_id, name, content):
        (box.root / name).write_bytes(content)

    monkeypatch.setattr(chat_endpoint.sandbox_manager, "write_file", write_file)
    async with client_for("owner", monkeypatch) as c:
        r = await c.post("/chat/sandbox-upload", data={"thread_id": "t", "folder": "inbox"},
                         files={"file": ("note.txt", b"new", "text/plain")})
        assert r.status_code == 200 and r.json()["name"] == "inbox/note (2).txt"
        r = await c.post("/chat/sandbox-upload", data={"thread_id": "t", "folder": "../up"},
                         files={"file": ("x.txt", b"x", "text/plain")})
        assert r.status_code == 400
    assert (box.root / "inbox" / "note (2).txt").read_bytes() == b"new"
    assert (box.root / "inbox" / "note.txt").read_text() == "old"


async def test_sandbox_state_reads_the_cluster_without_touching_it(monkeypatch):
    from datetime import UTC, datetime, timedelta

    from app.agents.tools import sandbox_manager as sm

    world = {"names": [], "claims": [], "sandboxes": []}

    class Client:
        async def list_all_sandboxes(self, namespace, label_selector):
            return world["names"]

    async def list_objects(api, namespace):
        return world["claims"] if api is sm._CLAIM_API else world["sandboxes"]

    monkeypatch.setattr(sm, "_get_client", lambda: Client())
    monkeypatch.setattr(sm, "_list_objects", list_objects)

    def at(minutes):
        return (datetime.now(UTC) + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")   # as Kubernetes writes it

    assert await sm.sandbox_state("t") == "none"
    world["names"] = ["claim-1"]
    world["claims"] = [{"metadata": {"name": "claim-1"}, "status": {"sandbox": {"name": "sb-1"}}}]
    world["sandboxes"] = [{"metadata": {"name": "sb-1"}, "spec": {"shutdownTime": at(30)}}]
    assert await sm.sandbox_state("t") == "running"
    world["sandboxes"][0]["spec"]["shutdownTime"] = at(-1)                       # paused: its pod is gone
    assert await sm.sandbox_state("t") == "paused"
