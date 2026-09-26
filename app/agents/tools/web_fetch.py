import codecs
import io
import re
import uuid
import zipfile

import httpx
from bs4 import BeautifulSoup
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from markdownify import markdownify

from app.agents.tools.sandbox_manager import get_thread_id
from app.agents.tools.sandbox_save import save_and_stub

_NOISE_TAGS = [
    "script",
    "style",
    "noscript",
    "template",
    "header",
    "footer",
    "nav",
    "aside",
    "iframe",
]
# Sandbox-side safety cap, not a context budget — the model never sees this
# much directly, save_and_stub() files it and hands back a short preview.
_MAX_SAVED_CHARS = 300_000

# Enforced while reading the body rather than after it, because "after" is
# too late: a URL pointing at a disk image would already be resident in the
# backend's memory. In bytes, since the cap has to hold before decoding.
_MAX_DOWNLOAD_BYTES = 8 << 20

_HTML_MIMES = {"text/html", "application/xhtml+xml"}
# Not text/* by media type, but plain text once decoded.
_TEXT_MIMES = {
    "application/json",
    "application/xml",
    "application/javascript",
    "application/x-yaml",
    "application/yaml",
    "application/x-ndjson",
}
_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_DOCX_MIMES = {_DOCX_MIME}

# Leading bytes that identify a format outright. Only formats whose first
# bytes are unambiguous belong here — HTML and plain text have no signature
# and are left to the header and to _is_probably_text().
_SIGNATURES = (
    (b"%PDF-", "application/pdf"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"\x1f\x8b", "application/gzip"),
    (b"BZh", "application/x-bzip2"),
    (b"\x7fELF", "application/x-executable"),
)
# How much of the body _is_probably_text() judges. Enough to catch a binary
# that happens to open with printable bytes, small enough to stay free.
_TEXT_SNIFF_BYTES = 8192


def _html_to_text(body: bytes, charset: str | None) -> str:
    """Markdown from an HTML body, decoded by whatever the page says it is.

    Takes bytes rather than str so BeautifulSoup can read a `<meta charset>`
    declaration, which is where plenty of pages — old ones especially — put
    their encoding instead of in the Content-Type header. Decoding here
    instead would mean guessing utf-8 and replacing every byte of a
    windows-1258 or shift_jis page with U+FFFD before bs4 ever saw the tag
    that would have decoded it correctly. `from_encoding` is the header's
    charset when the server sent one, since that outranks the document's
    own claim, and None when it did not, which is what lets bs4 sniff.
    """
    soup = BeautifulSoup(body, "html.parser", from_encoding=charset)
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            if attr.startswith("on") or attr in ("style", "class", "id", "data-mw"):
                del tag[attr]
    # Prefer main article body if present
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id="mw-content-text")
        or soup.find(id="bodyContent")
    )
    target = main if main else soup.body or soup
    return markdownify(str(target), strip=["script", "style"])


def _extract_pdf(data: bytes) -> str | None:
    """The text of a PDF, or None when it has none to give.

    A PDF stores its words compressed, so decoding the raw bytes as text
    yields the container and not the content — `%PDF-1.6 ... FlateDecode`,
    in which the words being searched for do not literally appear. Returns
    None for a scanned PDF too: those are page images with no text layer,
    which is a real answer rather than a failure.
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip() or None
    except Exception:
        return None


def _extract_docx(data: bytes) -> str | None:
    """The paragraph text of a .docx, or None if it will not open.

    A .docx is a zip archive of XML, so it mis-decodes the same way a PDF
    does; it is here because the alternative branch is the one that used to
    hand the model the zip header.
    """
    try:
        import docx

        document = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in document.paragraphs).strip() or None
    except Exception:
        return None


def _zip_kind(body: bytes) -> str:
    """Which of the zip-based formats this archive is.

    .docx and .xlsx are zip files, so the PK signature cannot tell them
    from an ordinary archive — only the member list can.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            names = set(z.namelist())
    except Exception:
        return "application/zip"   # truncated or not really a zip
    return _DOCX_MIME if "word/document.xml" in names else "application/zip"


def _sniff(body: bytes) -> str | None:
    """What the bytes actually are, or None when they do not say.

    A Content-Type header is a claim the server makes, and a wrong one is
    common: PDFs and Office documents arrive labelled
    application/octet-stream all the time. The leading bytes are not a
    claim, so where they identify a format they outrank the header.
    """
    for magic, mime in _SIGNATURES:
        if body.startswith(magic):
            return mime
    if body.startswith(b"PK\x03\x04"):
        return _zip_kind(body)
    return None


def _is_probably_text(body: bytes) -> bool:
    """Whether an unidentified body can be read as text.

    The last resort, for the servers that send markdown, CSV or source
    code as application/octet-stream with no signature to go on. A NUL
    byte or an invalid UTF-8 sequence rules it out; an incremental
    decoder is used so a multi-byte character straddling the end of the
    sample is not mistaken for corruption.
    """
    sample = body[:_TEXT_SNIFF_BYTES]
    if b"\x00" in sample:
        return False
    try:
        codecs.getincrementaldecoder("utf-8")().decode(sample)
    except UnicodeDecodeError:
        return False
    return True


def _to_text(body: bytes, mime: str, charset: str | None) -> str | None:
    """Decode a response body by what it actually is.

    None means "this is not readable as text" — the caller turns that into
    an error the model can act on. Returning a best-effort decode instead
    was the old behaviour and is worse than useless: str(pdf_bytes) is
    300k characters of mojibake that reads as success, costs a turn and
    thousands of tokens, and destroys the bytes on the way through, so the
    model cannot even fall back to extracting the file itself.
    """
    sniffed = _sniff(body)
    kind = sniffed or mime

    if kind == "application/pdf":
        return _extract_pdf(body)
    if kind in _DOCX_MIMES:
        return _extract_docx(body)
    if kind in _HTML_MIMES:
        return _html_to_text(body, charset)
    if kind.startswith("text/") or kind in _TEXT_MIMES:
        return body.decode(charset or "utf-8", errors="replace")
    # Unidentified by both signature and header. Refusing everything here
    # would throw away plain text served under a careless content type,
    # so let the bytes themselves decide — but only when no signature
    # already said this is a binary we have no reader for.
    if sniffed is None and _is_probably_text(body):
        return body.decode(charset or "utf-8", errors="replace")
    return None


def _filename_for(url: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", url).strip("-")[:40] or "page"
    return f"fetch_{slug}_{uuid.uuid4().hex[:6]}.md"


@tool
async def web_fetch(url: str, label: str, config: RunnableConfig) -> str:
    """Fetch the content of a web page and return it as markdown.

    HTML pages, PDFs, .docx and plain-text formats are read as text. Any
    other type (images, archives, binaries) returns an error rather than
    unreadable content — if that happens, look for an HTML version of the
    same document instead of retrying.

    When multiple URLs need to be read, call this tool in parallel — one
    call per URL — rather than sequentially. Parallel calls complete in
    the same time as a single call.

    A long page is saved to a file in your sandbox and you get back a short
    preview + the file name instead of the whole thing — use `bash` to grep
    or read out the part you actually need.

    Args:
        url: URL to fetch.
        label: Brief human-readable description shown to the user.
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            async with client.stream(
                "GET", url, headers={"User-Agent": "Mozilla/5.0 (compatible; JarvisBot/1.0)"}
            ) as response:
                response.raise_for_status()
                chunks: list[bytes] = []
                size = 0
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)
                    size += len(chunk)
                    if size >= _MAX_DOWNLOAD_BYTES:
                        break
                body = b"".join(chunks)
                mime = response.headers.get("content-type", "").split(";")[0].strip().lower()
                # charset_encoding, not encoding: the latter substitutes a
                # utf-8 default that is indistinguishable from the server
                # actually saying utf-8, and _to_text needs to know which.
                charset = response.charset_encoding
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} for {url}"
    except httpx.RequestError as e:
        return f"Error: Could not fetch {url} — {e}"

    truncated = size >= _MAX_DOWNLOAD_BYTES
    text = _to_text(body, mime, charset)

    if text is None:
        # A truncated binary is the likelier explanation than an empty one,
        # and the two want different follow-ups, so say which happened.
        if truncated:
            return (
                f"Error: {url} is larger than {_MAX_DOWNLOAD_BYTES // (1 << 20)}MB and was cut "
                f"short, so its content could not be read. Look for a smaller or paginated "
                f"version of the same document."
            )
        # Name what the bytes are, not what the header claimed — the two
        # disagree often enough that the header alone misleads whoever
        # reads the log, and the model too.
        detected = _sniff(body) or mime or "an unknown content type"
        return (
            f"Error: {url} is {detected}, which web_fetch cannot read as text. "
            f"If this is a document, look for an HTML version of it."
        )

    if len(text) > _MAX_SAVED_CHARS:
        text = text[:_MAX_SAVED_CHARS] + "\n\n[...content truncated...]"
    text = text.strip()

    thread_id = get_thread_id(config)
    return await save_and_stub(thread_id, _filename_for(url), text, kind=f"web_fetch({url})")
