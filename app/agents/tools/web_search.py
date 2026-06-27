from langchain_core.tools import tool
from tavily import TavilyClient

from app.core.config import settings
from app.agents.tools.messages import WebSearchMsg


@tool
def web_search(query: str) -> str:
    """Search the internet for current information.
    Use a single, specific query. Call this tool at most once or twice per user request — stop and answer as soon as you have enough information.
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
