"""The research subagent as a standalone graph, for the background tasks
async_research.py runs — as opposed to RESEARCH_SUBAGENT (subagents.py), which
the main agent waits on through the `task` tool.

Same model chain, same search/fetch tools and same per-tool caps, minus what
can't work with nobody attached to the run: `bash` pauses for a human's
approval, and `generate_visualization_svg` needs the sandbox's Chromium. A
background run also sets `sandbox: False` in its config (see
async_research.py), which makes web_search/web_fetch cut a long result instead
of saving it to a file nothing could read back.

Compiled with the app's checkpointer, so each task's thread keeps its messages
and plan as it goes — that is what check_research reads to say what a running
task is doing, and what survives a restart.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

from app.agents.llm import build_llm_with_fallback
from app.agents.middleware import SoftHardToolCallLimitMiddleware, ToolToggleMiddleware
from app.agents.research_prompt import research_system_prompt
from app.agents.tools.web_fetch import web_fetch
from app.agents.tools.web_search import web_search


def build_research_graph(checkpointer=None):
    return create_agent(
        model=build_llm_with_fallback(is_subagent_model=True),
        tools=[web_search, web_fetch],
        system_prompt=research_system_prompt(sandbox=False),
        name="research",
        checkpointer=checkpointer,
        middleware=[
            TodoListMiddleware(),
            # `disabled_tools` comes in the run's config from async_research.py,
            # so a conversation with web search turned off gets the same answer
            # here as it would from the in-process subagent.
            ToolToggleMiddleware(),
            SoftHardToolCallLimitMiddleware(tool_name="web_search", soft_limit=10, hard_limit=12),
            SoftHardToolCallLimitMiddleware(tool_name="web_fetch", soft_limit=10, hard_limit=12),
        ],
    )
