"""check_pdf: which pages of a PDF in the sandbox carry real text, which are only
a picture of a page (a scan), and which have a text layer that is garbage.

A PDF has no "scanned" flag, so this looks at each page: how many words its
text layer yields, how much of the page one picture covers, and how much of the
text is unreadable. The word threshold is the same one the OCR worker uses to
decide a page needs OCR (jarvis-ocr-worker/app/services/layout.py), so the two
agree about what counts as a scan.

The PDF is read on the backend, not in the sandbox, which has no PDF library.
"""

import asyncio
import io
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.agents.tools.sandbox_errors import read_error
from app.agents.tools.sandbox_manager import get_thread_id, normalize_workspace_path, read_file

MIN_WORDS_FOR_TEXT = 8       # fewer than this and the page is treated as having no text
MAX_FILE_BYTES = 50 << 20
MAX_PAGES = 300              # pages looked at; the rest are counted but not classified
PICTURE_COVER = 0.85         # a picture covering this much of the page is a scan background
SHORT_TEXT_WORDS = 40        # ... and this little text on top means the picture is the content
GARBLED_RATIO = 0.2          # share of unreadable characters that makes a text layer unreliable

# What pdfminer emits for a glyph it cannot map to a character, plus the
# Unicode replacement character and stray control characters.
_BAD = re.compile(r"\(cid:\d+\)|�|[\x00-\x08\x0b\x0c\x0e-\x1f]")


@dataclass(frozen=True)
class PageInfo:
    number: int
    kind: str          # "text" | "scan" | "scan-with-text" | "garbled"
    words: int
    picture_cover: float
    bad_ratio: float


def bad_ratio(text: str) -> float:
    """Share of the text that is unmapped glyphs or control characters."""
    if not text:
        return 0.0
    bad = sum(len(m.group()) for m in _BAD.finditer(text))
    return min(1.0, bad / len(text))


def classify(words: int, cover: float, bad: float) -> str:
    if words < MIN_WORDS_FOR_TEXT:
        return "scan"
    if bad >= GARBLED_RATIO:
        return "garbled"
    if cover >= PICTURE_COVER and words < SHORT_TEXT_WORDS:
        return "scan-with-text"
    return "text"


def _picture_cover(page) -> float:
    """Fraction of the page covered by its largest picture."""
    area = float(page.width * page.height) or 1.0
    best = 0.0
    for im in page.images:
        w = max(0.0, min(im["x1"], page.width) - max(im["x0"], 0.0))
        h = max(0.0, min(im["bottom"], page.height) - max(im["top"], 0.0))
        best = max(best, w * h / area)
    return min(best, 1.0)


def analyze_with_text(data: bytes, max_pages: int = MAX_PAGES) -> tuple[int, list[PageInfo], list[str]]:
    """(total page count, per-page info, per-page text) for the first `max_pages`
    pages. Raises on a file that is not a readable PDF."""
    import pdfplumber

    infos: list[PageInfo] = []
    texts: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages[:max_pages], 1):
            text = page.extract_text() or ""
            words = len(page.extract_words() or [])
            cover = _picture_cover(page)
            bad = bad_ratio(text)
            infos.append(PageInfo(i, classify(words, cover, bad), words, cover, bad))
            texts.append(text)
    return total, infos, texts


def analyze(data: bytes) -> tuple[int, list[PageInfo]]:
    """(total page count, per-page info for the first MAX_PAGES pages)."""
    total, infos, _ = analyze_with_text(data)
    return total, infos


def _ranges(numbers: list[int]) -> str:
    out, start, prev = [], None, None
    for n in numbers:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append((start, prev))
            start = prev = n
    if start is not None:
        out.append((start, prev))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in out)


_MEANING = {
    "text": "have a real text layer",
    "scan": "have no text layer: only a picture of the page (a scan)",
    "scan-with-text": "are mostly one picture with a little text on top (probably a scan)",
    "garbled": "have a text layer but the text is unreadable (broken font encoding)",
}


def describe(name: str, size: int, total: int, infos: list[PageInfo]) -> str:
    by_kind: dict[str, list[int]] = {}
    for p in infos:
        by_kind.setdefault(p.kind, []).append(p.number)
    lines = [f"{name}: {total} page{'s' if total != 1 else ''}, {size >> 10} KB."]
    if set(by_kind) == {"text"}:
        lines.append("Verdict: text PDF. Every page has a real text layer.")
    elif "text" not in by_kind:
        lines.append("Verdict: scanned PDF. No page has a usable text layer.")
    else:
        lines.append("Verdict: mixed. Some pages have text and some do not.")
    for kind in ("text", "scan", "scan-with-text", "garbled"):
        if kind in by_kind:
            lines.append(f"- Pages {_ranges(by_kind[kind])} {_MEANING[kind]}.")
    if total > len(infos):
        lines.append(f"Only the first {len(infos)} pages were checked; {total - len(infos)} more were not.")
    return "\n".join(lines)


@tool
async def check_pdf(path: str, label: str, config: RunnableConfig) -> str:
    """Check what kind of PDF a file in your bash sandbox is, page by page.

    A PDF can hold real text, or only a picture of each page (a scan), or a mix
    of both, and nothing in the file name says which. This tells you for each
    page whether it has a text layer, is a scan, is a picture with a little text
    on it, or has a text layer that is garbled. Use it before deciding how to
    read a PDF, instead of guessing.

    It only inspects the file; it does not return the document's text — use
    `read_pdf` for the text.

    Args:
        path: The PDF's path in your working directory, e.g. "contract.pdf".
        label: Brief human-readable description shown to the user (e.g. "Checking whether the PDF is scanned").
    """
    name = normalize_workspace_path(path)
    if name.startswith("/") or ".." in PurePosixPath(name).parts:
        return "Error: path must be inside your working directory (no '..', no absolute path)."
    try:
        data, _, filename = await read_file(get_thread_id(config), name)
    except Exception as e:  # noqa: BLE001 — a missing file or a sandbox hiccup, same to the model
        return read_error(name, e)
    if len(data) > MAX_FILE_BYTES:
        return f"Error: {filename} is {len(data) >> 20} MB, over the {MAX_FILE_BYTES >> 20} MB limit."
    if not data.lstrip()[:5].startswith(b"%PDF"):
        return f"Error: {filename} is not a PDF (it does not start with %PDF)."
    try:
        # pdfplumber is synchronous and CPU-bound; keep it off the event loop.
        total, infos = await asyncio.to_thread(analyze, data)
    except Exception as e:  # noqa: BLE001 — encrypted or corrupt files raise assorted types
        return f"Error: {filename} could not be read as a PDF ({type(e).__name__}); it may be encrypted or damaged."
    if not infos:
        return f"{filename} has no pages."
    return describe(filename, len(data), total, infos)
