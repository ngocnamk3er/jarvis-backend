from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    TodoListMiddleware,
    ToolCallLimitMiddleware,
)
from deepagents.middleware.summarization import SummarizationMiddleware
from deepagents.backends import StateBackend

from app.agents.llm import build_llm
from app.agents.prompt import build_system_prompt
from app.agents.tools import tools


def build_graph(checkpointer=None):
    return create_agent(
        model=build_llm(),
        tools=tools,
        system_prompt=build_system_prompt(),
        checkpointer=checkpointer,
        middleware=[
            SummarizationMiddleware(
                model=build_llm(),
                backend=StateBackend,
                trigger=("tokens", 60000),
                keep=("messages", 20),
                trim_tokens_to_summarize=40000,
            ),
            HumanInTheLoopMiddleware(
                interrupt_on={"bash": {"allowed_decisions": ["approve", "reject"]}},
            ),
            TodoListMiddleware(),
            ToolCallLimitMiddleware(
                tool_name="web_search",
                run_limit=50,
                exit_behavior="end",
            ),
            ToolCallLimitMiddleware(
                tool_name="web_fetch",
                run_limit=50,
                exit_behavior="end",
            ),
        ],
    )
