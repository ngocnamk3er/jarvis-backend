import uuid

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from tavily import AsyncTavilyClient

from app.agents.messages import WebSearchMsg
from app.agents.tools.sandbox_manager import get_thread_id
from app.agents.tools.sandbox_save import save_and_stub
from app.core.config import settings


@tool
async def web_search(query: str, label: str, config: RunnableConfig) -> str:
    """Search the internet for current information.

    When multiple independent topics need to be researched, call this tool
    in parallel — one call per query — rather than sequentially. Parallel
    calls complete in the same time as a single call.

    Examples of when to call in parallel:
    - "GDP of Vietnam AND GDP of Thailand" → two simultaneous calls
    - Need data from multiple sources → call each query at the same time
    - Different aspects of a topic → split into focused parallel queries

    A large result set is saved to a file in your sandbox and you get back a
    short preview instead of every result in full — use `bash` to read out
    the part you actually need.

    Args:
        query: A single, specific search query.
        label: Brief human-readable description shown to the user.
    """
    client = AsyncTavilyClient(api_key=settings.TAVILY_API_KEY)
    response = await client.search(query=query, max_results=5)

    lines = []
    results = response.get("results", [])
    if not results:
        return WebSearchMsg.NO_RESULTS
    for r in results:
        lines.append(f"**{r['title']}**")
        lines.append(r["url"])
        if r.get("content"):
            lines.append(r["content"])
        lines.append("")
    text = "\n".join(lines).strip()

    thread_id = get_thread_id(config)
    filename = f"search_{uuid.uuid4().hex[:6]}.md"
    return await save_and_stub(thread_id, filename, text, kind=f"web_search({query!r})")
