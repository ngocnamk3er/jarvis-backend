from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, TodoListMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from deepagents.backends import StateBackend

from app.agents.llm import build_llm_with_fallback
from app.agents.middleware import ContextTokensMiddleware, SoftHardToolCallLimitMiddleware
from app.agents.prompt import build_system_prompt
from app.agents.tools import tools
from app.agents.subagents import RESEARCH_SUBAGENT


def build_graph(
    checkpointer=None,
    store=None,
    include_tools: bool = True,
):
    """`include_tools=False` drops the raw `tools` list below (bash, web_search,
    viz, etc.) — for the dedicated graph used by
    conversation_service.get_messages() (see app.state.history_graph in
    main.py), which only ever calls graph.aget_state() to read a checkpoint.
    aget_state() doesn't invoke the model or any node, so which tools are
    bound to the model is irrelevant there — only the *middleware* list
    matters, because several middleware own state-schema channels
    (ContextTokensMiddleware's `context_tokens`, TodoListMiddleware's
    `todos`, etc.) that must stay identical across every graph sharing this
    checkpointer for aget_state() to deserialize a checkpoint correctly.
    That's why this flag only touches the `tools=` argument to create_agent()
    below and leaves the full `middleware` list untouched either way.

    Both graphs built here share the same checkpointer/store, so they operate
    on the exact same persisted thread state — this works because varying
    include_tools never changes node topology or state schema (it's just a
    different argument to the same create_agent() call, no middleware
    added/removed).
    """
    middleware = [
        ContextTokensMiddleware(),
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
    ]
    return create_agent(
        model=build_llm_with_fallback(),
        tools=tools if include_tools else [],
        system_prompt=build_system_prompt(),
        checkpointer=checkpointer,
        store=store,
        middleware=middleware,
    )
