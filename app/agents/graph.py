from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware, TodoListMiddleware

from app.agents.llm import build_llm
from app.agents.prompt import SYSTEM_PROMPT
from app.agents.tools import tools


def build_graph(checkpointer=None):
    return create_agent(
        model=build_llm(),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        middleware=[
            SummarizationMiddleware(
                model=build_llm(),
                trigger=("tokens", 60000),
                keep=("messages", 20),
            ),
            HumanInTheLoopMiddleware(
                interrupt_on={"bash": {"allowed_decisions": ["approve", "reject"]}},
            ),
            TodoListMiddleware(),
        ],
    )
