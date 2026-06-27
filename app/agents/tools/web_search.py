from langchain_core.tools import tool
from tavily import TavilyClient

from app.core.config import settings
from app.agents.tools.messages import WebSearchMsg


@tool
def web_search(query: str) -> str:
    """Search the internet for current information, news, or any topic.
    Use this when you need up-to-date information that may not be in your training data.
    """
    client = TavilyClient(api_key=settings.TAVILY_API_KEY)
    response = client.search(query=query, max_results=5)

    results = response.get("results", [])
    if not results:
        return WebSearchMsg.NO_RESULTS

    lines = []
    for r in results:
        lines.append(f"**{r['title']}**")
        lines.append(r["url"])
        if r.get("content"):
            lines.append(r["content"][:400])
        lines.append("")

    return "\n".join(lines).strip()
