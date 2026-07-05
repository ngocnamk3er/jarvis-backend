from langchain_core.tools import tool
from tavily import AsyncTavilyClient

from app.core.config import settings
from app.agents.messages import WebSearchMsg


@tool
async def web_search(query: str, label: str) -> str:
    """Search the internet for current information.
    Use a single, specific query. Call this tool at most once or twice per user request.

    Args:
        query: Search query.
        label: Brief human-readable description shown to the user.
    """
    client = AsyncTavilyClient(api_key=settings.TAVILY_API_KEY)
    response = await client.search(query=query, max_results=5, include_answer=True)

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
