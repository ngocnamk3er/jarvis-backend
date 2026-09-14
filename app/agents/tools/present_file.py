import json

from k8s_agent_sandbox.exceptions import SandboxRequestError
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.agents.tools.sandbox_manager import get_thread_id, normalize_workspace_path, read_file


@tool
async def present_file(path: str, label: str, config: RunnableConfig) -> str:
    """Hand a file you generated in the sandbox to the user as a download in
    the chat.

    Use this after creating a document with bash (a .docx/.pptx/.xlsx/.pdf,
    a chart .png, a .csv export, etc.). It reads the file back from the
    sandbox and renders a download chip in your reply — the user clicks to
    save it. It does NOT go into the user's file workspace; it's scoped to
    this chat.

    Args:
        path: The path you saved the file at with bash — a plain relative
            name like "report.docx" or "out/chart.png".
        label: Brief human-readable description shown to the user (e.g.
            "Sharing the quarterly report").
    """
    thread_id = get_thread_id(config)
    # normalize_workspace_path also accepts a leading "/workspace/" or "./"
    # if the model uses one, folding it down to the same relative form —
    # harmless either way, just don't assume /workspace is a real, listable
    # directory (see bash.py's docstring). Only a path that would escape the
    # working directory (a real absolute path, a "..") is rejected; the
    # sandbox re-checks too.
    name = normalize_workspace_path(path)
    from pathlib import PurePosixPath

    if name.startswith("/") or ".." in PurePosixPath(name).parts:
        return "Error: path must be inside your working directory (no '..', no absolute path)."

    try:
        content, mime, filename = await read_file(thread_id, name)
    except SandboxRequestError as e:
        detail = e.response.text if e.response is not None else str(e)
        return f"Error: {detail}"

    # thread_id travels in the payload so the frontend download link
    # (GET /api/v1/chat/sandbox-file) works on reload too, without any extra
    # plumbing through chat_service / serialize_messages.
    return json.dumps(
        {
            "__file__": {
                "name": filename,
                "mime": mime,
                "size": len(content),
                "path": name,
                "thread_id": thread_id,
            }
        }
    )
