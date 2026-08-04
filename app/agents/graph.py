from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    TodoListMiddleware,
)
from deepagents.middleware.summarization import SummarizationMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from deepagents.backends import StateBackend

from app.agents.llm import build_llm_with_fallback
from app.agents.middleware import SoftHardToolCallLimitMiddleware
from app.agents.prompt import build_system_prompt
from app.agents.tools import tools
from app.agents.subagents import RESEARCH_SUBAGENT


def build_graph(checkpointer=None):
    return create_agent(
        model=build_llm_with_fallback(),
        tools=tools,
        system_prompt=build_system_prompt(),
        checkpointer=checkpointer,
        middleware=[
            SummarizationMiddleware(
                model=build_llm_with_fallback(),
                backend=StateBackend,
                trigger=("tokens", 60000),
                keep=("messages", 20),
                trim_tokens_to_summarize=40000,
            ),
            HumanInTheLoopMiddleware(
                interrupt_on={"bash": {"allowed_decisions": ["approve", "reject"]}},
            ),
            TodoListMiddleware(),
            # Two-tier per-run caps (not thread-wide — resets every turn): up to
            # soft_limit calls get blocked-but-continue, a burst past hard_limit
            # blocks-and-stops-the-run entirely. See SoftHardToolCallLimitMiddleware's
            # docstring for why "end" on hard breach requires that tool to be the
            # only one called in that step.
            SoftHardToolCallLimitMiddleware(
                tool_name="web_search",
                soft_limit=50,
                hard_limit=60,
            ),
            SoftHardToolCallLimitMiddleware(
                tool_name="web_fetch",
                soft_limit=50,
                hard_limit=60,
            ),
            SoftHardToolCallLimitMiddleware(
                tool_name="task",
                soft_limit=5,
                hard_limit=7,
            ),
            SubAgentMiddleware(
                backend=StateBackend,
                subagents=[RESEARCH_SUBAGENT],
            ),
        ],
    )
