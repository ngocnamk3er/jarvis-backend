"""Agent tools over the user's persistent file workspace (jarvis-file-service)
— distinct from `bash`'s ephemeral, per-conversation sandbox /workspace.
Files uploaded through the UI (any conversation) live here and persist
across conversations.

No stateful `cd`: every tool takes an explicit `path` argument instead. The
system prompt already tells the agent to fire tool calls in parallel, and a
stateful "current directory" would race across concurrent calls the same
way it would with any shell -- explicit paths sidestep that entirely. See
~/.claude/plans/cosmic-drifting-pinwheel.md for the full rationale.

Read-only — no HumanInTheLoopMiddleware approval needed, same posture as
web_search/web_fetch (unlike bash, which can mutate the sandbox)."""

import httpx
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from app.clients import file_client


def _user_id(config: RunnableConfig) -> str:
    return config.get("configurable", {}).get("user_id", "")


def _format_size(n: int | None) -> str:
    if n is None:
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


@tool
async def list_files(path: str, config: RunnableConfig) -> str:
    """List the folders and files at a path in the user's persistent file
    workspace (uploaded via the UI — separate from bash's ephemeral sandbox).

    Args:
        path: Root-relative path to list. Pass "/" for the workspace root,
            or e.g. "/docs/2024" for a subfolder.
    """
    try:
        nodes = await file_client.list_tree(_user_id(config), path)
    except httpx.HTTPStatusError as e:
        return f"Error: {e.response.text}"
    if not nodes:
        return f"(empty) {path}"
    lines = []
    for n in nodes:
        if n["type"] == "folder":
            lines.append(f"📁 {n['name']}/")
        else:
            size = _format_size(n["size_bytes"])
            lines.append(f"📄 {n['name']} ({size})")
    return "\n".join(lines)


@tool
async def read_file(path: str, config: RunnableConfig) -> str:
    """Read a file's extracted text content from the user's file workspace.

    Args:
        path: Root-relative file path, e.g. "/docs/2024/report.pdf".
    """
    try:
        node = await file_client.resolve_path(_user_id(config), path)
    except httpx.HTTPStatusError as e:
        return f"Error: {e.response.text}"
    if node is None:
        return f"Error: not found: {path}"
    if node["type"] != "file":
        return f"Error: {path} is a folder, not a file"
    if node["indexing_status"] in ("pending", "extracting"):
        return "This file is still being processed — try again in a moment."
    if node["extracted_text"] is None:
        return f"'{node['name']}' has no extractable text (unsupported or binary file type)."
    return node["extracted_text"]


@tool
async def grep_files(pattern: str, path: str, config: RunnableConfig) -> str:
    """Keyword-search file names and contents under a path in the user's
    file workspace (case-insensitive substring match). This is the
    plain-text half of hybrid search — see search_files for the semantic half.

    Args:
        pattern: Text to search for.
        path: Root-relative path to search under. Pass "/" to search the
            whole workspace.
    """
    try:
        results = await file_client.grep_files(_user_id(config), pattern, path)
    except httpx.HTTPStatusError as e:
        return f"Error: {e.response.text}"
    if not results:
        return f"No matches for '{pattern}' under {path}"
    return "\n".join(f"{r['path']} ({_format_size(r['size_bytes'])})" for r in results)


@tool
async def search_files(query: str, top_k: int, config: RunnableConfig) -> str:
    """Semantic search over the user's file workspace — finds relevant
    content by meaning, not just literal keyword match (the plain-text half
    is grep_files). Use this for "find files about X" style questions where
    the exact wording in the file may differ from the query.

    Args:
        query: Natural-language description of what to find.
        top_k: Max number of chunks to return — 5 is a reasonable default.
    """
    try:
        results = await file_client.search_vector(_user_id(config), query, top_k)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400:
            return "Semantic search isn't configured for this deployment — use grep_files instead."
        return f"Error: {e.response.text}"
    if not results:
        return f"No semantic matches for '{query}'"
    lines = []
    for r in results:
        snippet = r["chunk_text"][:300]
        lines.append(f"{r['path']} (score={r['score']:.2f}):\n{snippet}")
    return "\n\n".join(lines)
