import json

import httpx
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from app.agents.tools.sandbox_manager import get_thread_id, read_file


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
        path: The same path you saved the file at with bash — normally a plain
            relative name like "report.docx" (resolved from the conversation's
            working directory).
        label: Brief human-readable description shown to the user (e.g.
            "Sharing the quarterly report").
    """
    thread_id = get_thread_id(config)
    # The sandbox resolves a relative name against the conversation's dir and an
    # absolute path as-is (both must stay under /workspace) — so pass the path
    # through almost untouched; just normalise a leading "./".
    name = path.strip()
    if name.startswith("./"):
        name = name[2:]

    try:
        content, mime, filename = await read_file(thread_id, name)
    except httpx.HTTPStatusError as e:
        return f"Error: {e.response.text}"
    except httpx.HTTPError as e:
        return f"Error: sandbox unavailable ({e})."

    # thread_id travels in the payload so the frontend download link
    # (GET /api/v1/chat/sandbox-file) works on reload too, without any extra
    # plumbing through chat_service / serialize_messages.
    return json.dumps(
        {"__file__": {"name": filename, "mime": mime, "size": len(content), "path": name, "thread_id": thread_id}}
    )
