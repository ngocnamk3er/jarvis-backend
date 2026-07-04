from langchain_core.tools import tool
from tavily import TavilyClient

from app.core.config import settings
from app.agents.messages import WebSearchMsg


@tool
def web_search(query: str, label: str) -> str:
    """Search the internet for current information.
    Use a single, specific query. Call this tool at most once or twice per user request — stop and answer as soon as you have enough information.

    Args:
        query: Search query.
        label: Brief human-readable description shown to the user (e.g. "Searching for Bitcoin price").
    """
    client = TavilyClient(api_key=settings.TAVILY_API_KEY)
    response = client.search(query=query, max_results=5, include_answer=True)

    lines = []

    answer = response.get("answer", "")
    if answer:
        lines.append(f"**Direct answer:** {answer}\n")

    results = response.get("results", [])
    if not results:
        return WebSearchMsg.NO_RESULTS if not lines else "\n".join(lines).strip()

    for r in results:
        lines.append(f"**{r['title']}**")
        lines.append(r["url"])
        if r.get("content"):
            lines.append(r["content"])
        lines.append("")

    return "\n".join(lines).strip()
