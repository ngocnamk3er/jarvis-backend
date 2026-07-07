from langchain_core.tools import tool
from tavily import AsyncTavilyClient

from app.core.config import settings
from app.agents.messages import WebSearchMsg


@tool
async def web_search(query: str, label: str) -> str:
    """Search the internet for current information.

    When multiple independent topics need to be researched, call this tool
    in parallel — one call per query — rather than sequentially. Parallel
    calls complete in the same time as a single call.

    Examples of when to call in parallel:
    - "GDP of Vietnam AND GDP of Thailand" → two simultaneous calls
    - Need data from multiple sources → call each query at the same time
    - Different aspects of a topic → split into focused parallel queries

    Args:
        query: A single, specific search query.
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
