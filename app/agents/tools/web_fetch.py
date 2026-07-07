import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify
from langchain_core.tools import tool

_NOISE_TAGS = ["script", "style", "noscript", "template", "header", "footer", "nav", "aside", "iframe"]


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            if attr.startswith("on") or attr in ("style", "class", "id", "data-mw"):
                del tag[attr]
    # Prefer main article body if present
    main = soup.find("main") or soup.find("article") or soup.find(id="mw-content-text") or soup.find(id="bodyContent")
    target = main if main else soup.body or soup
    return markdownify(str(target), strip=["script", "style"])


@tool
async def web_fetch(url: str, label: str) -> str:
    """Fetch the content of a web page and return it as markdown.

    When multiple URLs need to be read, call this tool in parallel — one
    call per URL — rather than sequentially. Parallel calls complete in
    the same time as a single call.

    Args:
        url: URL to fetch.
        label: Brief human-readable description shown to the user.
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            response = await client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; JarvisBot/1.0)"})
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} for {url}"
    except httpx.RequestError as e:
        return f"Error: Could not fetch {url} — {e}"

    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type:
        text = _html_to_text(response.text)
        limit = 12000
    else:
        text = response.text
        limit = 20000

    if len(text) > limit:
        text = text[:limit] + "\n\n[...content truncated...]"
    return text.strip()
