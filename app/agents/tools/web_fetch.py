import httpx
from markdownify import markdownify
from langchain_core.tools import tool


@tool
async def web_fetch(url: str, label: str) -> str:
    """Fetch the content of a web page and return it as markdown.

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
    text = markdownify(response.text, strip=["script", "style"]) if "text/html" in content_type else response.text
    if len(text) > 20000:
        text = text[:20000] + "\n\n[...content truncated...]"
    return text.strip()
