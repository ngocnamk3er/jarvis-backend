"""Agent tools over the user's persistent file workspace (jarvis-file-service)
— distinct from `bash`'s ephemeral, per-conversation sandbox /workspace.
Files uploaded through the UI (any conversation) live here and persist
across conversations.

No stateful `cd`: every tool takes an explicit `path` argument instead. The
system prompt already tells the agent to fire tool calls in parallel, and a
stateful "current directory" would race across concurrent calls the same
way it would with any shell -- explicit paths sidestep that entirely. See
~/.claude/plans/cosmic-drifting-pinwheel.md for the full rationale.

No HumanInTheLoopMiddleware approval on any of these, same posture as
web_search/web_fetch (unlike bash, which can run arbitrary commands). All
but `fetch_file` are read-only, and `fetch_file` only writes a file the
user already owns into that user's own throwaway sandbox — the same thing
web_fetch does with a page it just downloaded."""

import re
import uuid

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.agents import retrieval
from app.agents.tools import sandbox_manager
from app.agents.tools.sandbox_manager import get_thread_id
from app.agents.tools.sandbox_save import save_and_stub
from app.clients import file_client

# The bytes sit in backend memory between the two hops (workspace → here →
# sandbox), so this bounds memory, not sandbox disk. Well above the office
# documents this exists for; a media file that trips it wants a different
# approach anyway.
_MAX_FETCH_BYTES = 50 << 20
# Long enough to keep a real document name recognisable, short enough to
# stay well inside any filesystem's per-component limit once the workspace
# name has been widened by escaping.
_MAX_NAME_CHARS = 80


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

    A long document is saved to a file in the bash sandbox and you get back
    a short preview instead of the whole thing — use `bash` to grep or read
    out the part you actually need.

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
        # A spreadsheet lands here every time — extraction covers pdf, docx
        # and the plain-text extensions, nothing else. Saying only "no text"
        # reads as "this file is unusable", which is wrong and is where the
        # agent used to give up; name the tool that does work on it.
        return (
            f"'{node['name']}' has no extractable text (unsupported or binary file type). "
            f"If its content is structured — a spreadsheet, a PDF of scans, an image — use "
            f"fetch_file to copy it into the bash sandbox and open it with a library there."
        )

    thread_id = get_thread_id(config)
    slug = re.sub(r"[^A-Za-z0-9]+", "-", node["name"]).strip("-")[:40] or "file"
    filename = f"read_{slug}_{uuid.uuid4().hex[:6]}.txt"
    return await save_and_stub(
        thread_id, filename, node["extracted_text"], kind=f"read_file({path!r})"
    )


def _sandbox_name(name: str) -> str:
    """A filename safe to write into the sandbox, extension intact.

    The extension is load-bearing rather than cosmetic: pandas, openpyxl
    and pypdf all dispatch on it, so a workbook that lands as `download`
    is one the agent cannot open without first being told what it is.
    """
    # `\w` under Unicode keeps letters of any script, so a Vietnamese name
    # stays readable instead of collapsing to `b_o_c_o.xlsx`; what goes is
    # the path separators, whitespace and shell metacharacters. Stripping
    # the leading `._-` then disposes of both `..` traversal and dotfiles.
    cleaned = re.sub(r"[^\w.\-]+", "_", name, flags=re.UNICODE).strip("._-")
    if not cleaned:
        return "file"
    if len(cleaned) <= _MAX_NAME_CHARS:
        return cleaned
    # Truncate the stem, never the extension — losing `.xlsx` off a long
    # name would defeat the one thing this function is careful about.
    stem, dot, ext = cleaned.rpartition(".")
    if dot and len(ext) <= 12:
        return f"{stem[: _MAX_NAME_CHARS - len(ext) - 1]}.{ext}"
    return cleaned[:_MAX_NAME_CHARS]


@tool
async def fetch_file(path: str, config: RunnableConfig) -> str:
    """Copy a file from the user's workspace into the bash sandbox, byte for byte.

    Use this when the file's structure is the point and flattened text
    will not do: a spreadsheet's sheets and cells, a PDF's tables and
    layout, an image, anything you want to open with a library. The file
    lands in your bash working directory under its own name, and from
    there it is yours: open it with `bash` and whatever library the
    sandbox has.

    When you only need the words out of a document, prefer `read_file`:
    it hands you the text directly instead of costing you a second round
    trip through bash.

    Args:
        path: Root-relative file path, e.g. "/reports/q3-2024.xlsx".
    """
    user_id = _user_id(config)
    try:
        node = await file_client.resolve_path(user_id, path)
    except httpx.HTTPStatusError as e:
        return f"Error: {e.response.text}"
    if node is None:
        return f"Error: not found: {path}"
    if node["type"] != "file":
        return f"Error: {path} is a folder, not a file"

    size = node.get("size_bytes") or 0
    if size > _MAX_FETCH_BYTES:
        return (
            f"Error: {path} is {_format_size(size)}, over the "
            f"{_format_size(_MAX_FETCH_BYTES)} limit for copying into the sandbox."
        )

    try:
        content, mime_type, _ = await file_client.get_content(user_id, node["id"])
    except httpx.HTTPStatusError as e:
        return f"Error: {e.response.text}"

    name = _sandbox_name(node["name"])
    try:
        await sandbox_manager.write_file(get_thread_id(config), name, content)
    except Exception as e:  # noqa: BLE001 — the sandbox can fail many ways; all of
        # them mean the same thing to the model, and none should end the turn.
        return f"Error: could not write {name} into the sandbox — {type(e).__name__}: {e}"

    return (
        f"Copied {path} into the bash sandbox as ./{name} "
        f"({_format_size(len(content))}, {mime_type}). Open it with bash."
    )


def _snippet(text: str, query: str, width: int = 300) -> str:
    """A window of `text` centered on where `query` actually appears,
    rather than always the start.

    A chunk can span several pages when a short document collapses into
    one chunk_words-sized chunk (see chunking.py) — the first `width`
    chars then only cover the opening page, and an exact-match hit near
    the end is invisible in the preview even though it is what matched.
    A real trace on 2026-10-03 showed the model call search_files(exact=
    True), get back a truncated preview that stopped short of the ID it
    had just found, and spend a second round trip on read_file just to
    see what search_files had already matched.

    Falls back to the start when `query` is not found verbatim — e.g. an
    ascii_folding match across a diacritic the plain text does not share,
    or a non-phrase (any order) match whose words are scattered wider
    than one window can show.
    """
    idx = text.lower().find(query.lower())
    if idx == -1:
        for word in query.split():
            idx = text.lower().find(word.lower())
            if idx != -1:
                break
    if idx == -1:
        return text[:width]
    start = max(0, min(idx - width // 3, len(text) - width))
    start = max(0, start)
    end = min(len(text), start + width)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


def _citation(r: dict) -> dict:
    """One hit, shaped for the frontend's source chips and viewer.

    Everything a viewer needs to open the file and draw the highlight
    without another round trip to the search index: which file, which
    pages, the boxes (fractions of the page), and the char span into the
    file's extracted text for formats that have no pages at all.
    """
    return {
        "file_id": r["file_id"],
        "path": r["path"],
        "page_start": r.get("page_start", 0),
        "page_end": r.get("page_end", 0),
        "boxes": r.get("boxes", []),
        "char_start": r.get("char_start", 0),
        "char_end": r.get("char_end", 0),
        "score": r.get("rerank_score", r.get("score")),
        "snippet": r["chunk_text"][:240],
    }


# content_and_artifact: the string goes to the model, the artifact does not.
# The artifact is where the citations ride — the frontend needs the boxes to
# draw highlights, the model would only pay tokens for them. LangChain keeps
# it on the ToolMessage, so it reaches the live SSE stream (tool_end) and
# survives in the checkpoint for when the conversation is reloaded.
@tool(response_format="content_and_artifact")
async def search_files(
    query: str, config: RunnableConfig, top_k: int = 5, exact: bool = False
) -> tuple[str, dict | None]:
    """Search the user's file workspace — the way to find what the user
    has uploaded. Two modes:

    - exact=False (default): semantic search. Matches by meaning, so it
      answers "the file about X" without needing the file's name or its
      exact wording.
    - exact=True: exact-phrase search over the same indexed chunks — the
      words adjacent, in the order given. Use this for a code, an ID, a
      name, or a quoted string, where it is the literal wording that
      matters — a query like "the invoice about late fees" would not
      find invoice number INV-2024-0871 that way, but the number itself
      will. Still needs real words to match on; it is not a filename or
      path lookup (use list_files for that).

    Pairs with `list_files`, which shows what is there by path when you
    want to browse rather than search. `read_file` and `fetch_file` both
    need a path — one of these results, one from a listing, or one the
    user gave you.

    Args:
        query: What to find — a natural-language description (exact=False)
            or the exact text to match (exact=True).
        top_k: Max number of chunks to return — 5 is a reasonable default.
        exact: True for exact-phrase matching instead of semantic search.
    """
    user_id = _user_id(config)
    trace: list[str] = []
    try:
        if exact:
            results = await file_client.search_fulltext(user_id, query, top_k)
        elif retrieval.enabled():
            results, trace = await retrieval.agentic_search(user_id, query, top_k)
        else:
            results = await file_client.search_vector(user_id, query, top_k)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400:
            return (
                "Semantic search isn't configured for this deployment — browse with "
                "list_files instead."
            ), None
        return f"Error: {e.response.text}", None
    if not results:
        kind = "exact matches" if exact else "semantic matches"
        return f"No {kind} for '{query}'", None

    lines = []
    for r in results:
        # exact search has no score of its own (see search_fulltext) —
        # saying "similarity=1.00" would claim a measurement that was
        # never taken. rerank_score is absent when agentic retrieval is
        # off or fell back to plain vector search, so report whichever
        # ranking actually applied rather than implying a judgement that
        # never happened.
        if exact:
            label = "exact match"
        elif "rerank_score" in r:
            label = f"relevance={r['rerank_score']:.0f}/10, similarity={r['score']:.2f}"
        else:
            label = f"similarity={r['score']:.2f}"
        # Page 0 means the format had no pagination to report (a .txt, a
        # .docx), so say nothing rather than print a page that does not
        # exist. The boxes that come back alongside are for a viewer to
        # draw with and would only be noise here.
        start, end = r.get("page_start", 0), r.get("page_end", 0)
        if start:
            page = f"p.{start}" if start == end else f"pp.{start}-{end}"
            label = f"{page}, {label}"
        preview = _snippet(r["chunk_text"], query, 300) if exact else r["chunk_text"][:300]
        lines.append(f"{r['path']} ({label}):\n{preview}")
    body = "\n\n".join(lines)
    text = f"[retrieval: {'; '.join(trace)}]\n\n{body}" if trace else body
    return text, {"citations": [_citation(r) for r in results]}
