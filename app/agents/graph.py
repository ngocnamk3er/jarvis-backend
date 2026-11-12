from deepagents.backends import StateBackend
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, TodoListMiddleware

from app.agents.async_research import BackgroundResearchMiddleware
from app.agents.llm import build_llm_with_fallback
from pathlib import Path

from app.agents.middleware import (
    ContextBreakdownMiddleware,
    ContextTokensMiddleware,
    SoftHardToolCallLimitMiddleware,
    ToolToggleMiddleware,
)
from app.agents.skills import SkillsMiddleware, load_skills
from app.agents.tool_search import ToolSearchMiddleware
from app.agents.user_mcp import UserMCPMiddleware, deferred_catalog
from app.agents.prompt import build_system_prompt
from app.agents.subagents import RESEARCH_SUBAGENT
from app.agents.tools import tools
from app.core.config import settings


def _bash_needs_approval(request) -> bool:
    """Whether this bash call should stop and ask a human. False auto-approves.

    False when the run is `unattended` — nobody is there to answer. Benchmarks
    approve every call anyway, and the round trip is not free: each approval
    ends the SSE stream and needs a fresh /chat/resume, so a run wanting nine
    bash calls needs nine requests, one long chain to lose an access token in
    or to run past the client's round cap. Cases on 2026-09-23 finished having
    answered nothing for that reason.

    Read per call rather than fixed at build time, because this graph also
    serves real chat, where a human does have to see the prompt.
    """
    try:
        cfg = (request.runtime.config or {}).get("configurable", {})
    except AttributeError:
        return True
    return not cfg.get("unattended", False)


_DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def build_graph(
    checkpointer=None,
    store=None,
    include_tools: bool = True,
    extra_tools: list | None = None,
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

    `extra_tools` are tools loaded at startup (MCP — see mcp_tools.py), added
    to the bound list. They are deferred by default (MCP_DEFER).

    Both graphs built here share the same checkpointer/store, so they operate
    on the exact same persisted thread state — this works because varying
    include_tools never changes node topology or state schema (it's just a
    different argument to the same create_agent() call, no middleware
    added/removed).
    """
    middleware = [
        ContextTokensMiddleware(),
        # Drops web_search/web_fetch (etc.) per-run when the user toggles them
        # off — see the `web_search` flag in ChatRequest / chat_service._make_config.
        ToolToggleMiddleware(),
        HumanInTheLoopMiddleware(
            interrupt_on={"bash": {"allowed_decisions": ["approve", "reject"],
                                   "when": _bash_needs_approval}},
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
    # Conditional on config, not on the request, so the history graph (see
    # include_tools above) gets the same `research_tasks` state channel.
    if settings.BACKGROUND_RESEARCH:
        middleware += [
            SoftHardToolCallLimitMiddleware(tool_name="start_research", soft_limit=3, hard_limit=5),
            BackgroundResearchMiddleware(),
        ]
    # Skills and deferred tools both change what the model is sent, so they
    # come after everything that adds tools or prompt text, and before the
    # breakdown middleware, which has to measure the final request.
    skills = load_skills(Path(settings.SKILLS_DIR) if settings.SKILLS_DIR else _DEFAULT_SKILLS_DIR) \
        if settings.SKILLS_ENABLED else {}
    if skills:
        middleware.append(SkillsMiddleware(skills))
    extra_tools = extra_tools or []
    mcp_names = frozenset(t.name for t in extra_tools)
    deferred = frozenset(n.strip() for n in settings.DEFERRED_TOOLS.split(",") if n.strip())
    if settings.MCP_DEFER:
        deferred |= mcp_names
    # Outside ToolSearch: the user's own MCP tools must be in the request by the
    # time it decides what to hold back.
    middleware.append(UserMCPMiddleware())
    middleware.append(ToolSearchMiddleware(deferred, extra_catalog=deferred_catalog))
    # Last on purpose: it measures the request after every other middleware
    # has had its say (see its docstring).
    middleware.append(ContextBreakdownMiddleware(mcp_tools=mcp_names))
    return create_agent(
        model=build_llm_with_fallback(),
        tools=(tools + extra_tools) if include_tools else [],
        system_prompt=build_system_prompt(),
        checkpointer=checkpointer,
        store=store,
        middleware=middleware,
    )
