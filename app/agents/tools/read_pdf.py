"""read_pdf: the text of a PDF in the sandbox, written to a file next to it.

The text goes into a file rather than into the conversation, so a 300-page
document costs the model a short summary, not its whole length. The model then
reads the parts it needs with `bash` (grep, sed -n). This is the same idea as
`save_and_stub` for web pages, with the extraction done here because the sandbox
has no PDF library.

Pages are classified by check_pdf's rules. A page that is only a picture (a
scan) or whose text is garbled gets a marker line and no text; it is listed in
the result so the model knows that part of the document was not read.
"""

import asyncio
from pathlib import PurePosixPath

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.agents.tools.check_pdf import MAX_FILE_BYTES, _ranges, analyze_with_text
from app.agents.tools.sandbox_errors import read_error
from app.agents.tools.sandbox_manager import (
    get_thread_id,
    normalize_workspace_path,
    read_file,
    write_file,
)

MAX_PAGES = 1000
PREVIEW_CHARS = 1500
_READABLE = {"text", "scan-with-text"}


def output_name(name: str) -> str:
    p = PurePosixPath(name)
    return str(p.with_suffix(".txt")) if p.suffix.lower() == ".pdf" else name + ".txt"


def build_text(infos, texts) -> str:
    parts = []
    for info, text in zip(infos, texts):
        if info.kind in _READABLE:
            parts.append(f"=== page {info.number} ===\n{text.strip()}")
        else:
            why = "scanned page, no text" if info.kind == "scan" else "text unreadable"
            parts.append(f"=== page {info.number} === [{why}]")
    return "\n\n".join(parts) + "\n"


@tool
async def read_pdf(path: str, label: str, config: RunnableConfig) -> str:
    """Extract the text of a PDF in your bash sandbox into a text file.

    The text is written to a .txt file beside the PDF (report.pdf becomes
    report.txt), one block per page starting with a `=== page N ===` line. You
    get back a short summary and a preview, not the whole text: read what you
    need with bash, e.g. `grep -n "total" report.txt` or
    `sed -n '/=== page 12 ===/,/=== page 13 ===/p' report.txt`.

    Pages that are only a scanned picture have no text to extract; they are
    marked in the file and listed in the summary. Tables and columns come out
    as plain lines of text, in reading order.

    Args:
        path: The PDF's path in your working directory, e.g. "contract.pdf".
        label: Brief human-readable description shown to the user (e.g. "Reading the PDF").
    """
    name = normalize_workspace_path(path)
    if name.startswith("/") or ".." in PurePosixPath(name).parts:
        return "Error: path must be inside your working directory (no '..', no absolute path)."
    thread_id = get_thread_id(config)
    try:
        data, _, filename = await read_file(thread_id, name)
    except Exception as e:  # noqa: BLE001 — a missing file or a sandbox hiccup, same to the model
        return read_error(name, e)
    if len(data) > MAX_FILE_BYTES:
        return f"Error: {filename} is {len(data) >> 20} MB, over the {MAX_FILE_BYTES >> 20} MB limit."
    if not data.lstrip()[:5].startswith(b"%PDF"):
        return f"Error: {filename} is not a PDF (it does not start with %PDF)."
    try:
        total, infos, texts = await asyncio.to_thread(analyze_with_text, data, MAX_PAGES)
    except Exception as e:  # noqa: BLE001 — encrypted or corrupt files raise assorted types
        return f"Error: {filename} could not be read as a PDF ({type(e).__name__}); it may be encrypted or damaged."
    if not infos:
        return f"{filename} has no pages."

    body = build_text(infos, texts)
    out = output_name(name)
    try:
        await write_file(thread_id, out, body.encode("utf-8"))
    except Exception as e:  # noqa: BLE001
        return f"Error: could not write {out} into the sandbox ({type(e).__name__}: {e})."

    unread = [i.number for i in infos if i.kind not in _READABLE]
    lines = [f"{filename}: {total} pages. Text of {len(infos) - len(unread)} pages written to {out} "
             f"({len(body):,} chars), one `=== page N ===` block per page."]
    if unread:
        lines.append(f"Not read (scanned or unreadable, no text in the file): pages {_ranges(unread)}.")
    if total > len(infos):
        lines.append(f"Only the first {len(infos)} pages were read; {total - len(infos)} more were not.")
    lines.append(f"Read parts of it with bash, for example: grep -n \"keyword\" {out}")
    lines.append(f"--- start of {out} ---\n{body[:PREVIEW_CHARS].rstrip()}"
                 + ("\n... (rest is in the file)" if len(body) > PREVIEW_CHARS else ""))
    return "\n".join(lines)
