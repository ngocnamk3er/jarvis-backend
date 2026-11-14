"""The tools over the user's file workspace tell a model that reached for them by mistake where the file really is."""

import httpx

from app.agents.tools import files as tools


def _config():
    return {"configurable": {"user_id": "u1", "thread_id": "t"}}


async def test_read_file_on_a_name_that_is_not_in_the_workspace_points_at_bash(monkeypatch):
    async def resolve_path(user_id, path):
        return None

    monkeypatch.setattr(tools, "_user_id", lambda config: "u1")
    monkeypatch.setattr(tools.file_client, "resolve_path", resolve_path)
    out = await tools.read_file.ainvoke({"path": "doanh_thu.csv"}, config=_config())
    assert out.startswith("Error: not found: doanh_thu.csv.")
    assert "attached to the user's message" in out and "`bash`" in out          # where such a file is, and what reads it


async def test_fetch_file_on_a_name_that_is_not_in_the_workspace_points_at_bash(monkeypatch):
    async def resolve_path(user_id, path):
        return None

    monkeypatch.setattr(tools, "_user_id", lambda config: "u1")
    monkeypatch.setattr(tools.file_client, "resolve_path", resolve_path)
    out = await tools.fetch_file.ainvoke({"path": "ghi_chu.txt"}, config=_config())
    assert out.startswith("Error: not found: ghi_chu.txt.") and "`bash`" in out


async def test_list_files_on_a_missing_path_points_at_bash_but_other_errors_do_not(monkeypatch):
    def failing(status):
        async def list_tree(user_id, path):
            request = httpx.Request("GET", "http://files/tree")
            raise httpx.HTTPStatusError("x", request=request, response=httpx.Response(status, text="boom", request=request))
        return list_tree

    monkeypatch.setattr(tools, "_user_id", lambda config: "u1")
    monkeypatch.setattr(tools.file_client, "list_tree", failing(404))
    assert "`bash`" in await tools.list_files.ainvoke({"path": "/x.csv"}, config=_config())
    monkeypatch.setattr(tools.file_client, "list_tree", failing(500))
    assert await tools.list_files.ainvoke({"path": "/"}, config=_config()) == "Error: boom"


def test_the_descriptions_say_who_each_tool_is_for():
    assert "attached to the user's message" in tools.read_file.description and "`bash`" in tools.read_file.description
    assert "Do not read every file" in tools.search_files.description


def test_the_tools_over_the_file_workspace_are_named_after_it():
    # `read_file` read as "any file" to the model, which tried it on files attached to the message (4 of 4 runs);
    # with the workspace in the name it went straight to bash instead
    from app.agents.tools import tools as registered

    names = {t.name for t in registered}
    assert {"workspace_list_files", "workspace_read_file", "workspace_search_files", "workspace_copy_to_sandbox"} <= names
    assert not {"list_files", "read_file", "search_files", "fetch_file"} & names
