import re
import uuid

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


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
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


def _filename_for(url: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", url).strip("-")[:40] or "page"
    return f"fetch_{slug}_{uuid.uuid4().hex[:6]}.md"


@tool
async def web_fetch(url: str, label: str, config: RunnableConfig) -> str:
    """Fetch the content of a web page and return it as markdown.

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
            response = await client.get(
                url, headers={"User-Agent": "Mozilla/5.0 (compatible; JarvisBot/1.0)"}
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} for {url}"
    except httpx.RequestError as e:
        return f"Error: Could not fetch {url} — {e}"

    content_type = response.headers.get("content-type", "")
    text = _html_to_text(response.text) if "text/html" in content_type else response.text

    if len(text) > _MAX_SAVED_CHARS:
        text = text[:_MAX_SAVED_CHARS] + "\n\n[...content truncated...]"
    text = text.strip()

    thread_id = get_thread_id(config)
    return await save_and_stub(thread_id, _filename_for(url), text, kind=f"web_fetch({url})")
