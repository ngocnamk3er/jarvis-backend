import httpx
from markdownify import markdownify
from langchain_core.tools import tool


@tool
def web_fetch(url: str) -> str:
    """Fetch the content of a web page and return it as markdown.
    Use this to read articles, documentation, or any public web page when you have a specific URL.
    Do not use for URLs that require authentication or login.
    """
    try:
        with httpx.Client(follow_redirects=True, timeout=15) as client:
            response = client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; JarvisBot/1.0)"},
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} for {url}"
    except httpx.RequestError as e:
        return f"Error: Could not fetch {url} — {e}"

    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type:
        text = markdownify(response.text, strip=["script", "style"])
    else:
        text = response.text

    # Truncate to avoid overwhelming context
    if len(text) > 20000:
        text = text[:20000] + "\n\n[...content truncated...]"

    return text.strip()
