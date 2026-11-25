"""Sub-agent specs for the `task` tool (deepagents' SubAgentMiddleware)."""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, TodoListMiddleware
from deepagents.middleware.subagents import CompiledSubAgent

from app.agents.llm import build_llm_with_fallback
from app.agents.research_prompt import research_system_prompt
from app.agents.middleware import SoftHardToolCallLimitMiddleware, ToolToggleMiddleware
from app.agents.tools.web_search import web_search
from app.agents.tools.web_fetch import web_fetch
from app.agents.tools.bash import bash
from app.agents.tools.generate_visualization_svg import generate_visualization_svg

# tool -> (soft, hard) calls per subagent run. The soft limit is the budget its prompt states.
RESEARCH_LIMITS = {"web_search": (10, 12), "web_fetch": (10, 12), "bash": (5, 7), "generate_visualization_svg": (3, 5)}

# Built as a CompiledSubAgent (our own create_agent() call) rather than a raw
# SubAgent spec so the model can carry a .with_fallbacks() chain (see
# build_llm_with_fallback) — deepagents' resolve_model() breaks on a
# .with_fallbacks()-wrapped model passed through a raw SubAgent's "model"
# field, since it isn't a BaseChatModel subclass (verified live: raises
# AttributeError trying to treat it as a provider-prefixed string).
RESEARCH_SUBAGENT: CompiledSubAgent = {
    "name": "research",
    "description": (
        "Delegate multi-step research here: comparing several things, gathering information "
        "from multiple sources, or deep-diving one topic. Runs its own search/fetch loop in "
        "isolation and returns one synthesized report — use it instead of calling web_search/"
        "web_fetch directly when the task needs more than 1-2 lookups."
    ),
    "runnable": create_agent(
        model=build_llm_with_fallback(is_subagent_model=True),
        tools=[web_search, web_fetch, bash, generate_visualization_svg],
        system_prompt=research_system_prompt(sandbox=True, budget={t: s for t, (s, _) in RESEARCH_LIMITS.items()}),
        name="research",
        # Caps each subagent *instance's* own tool use — independent of, and in
        # addition to, the main agent's SoftHardToolCallLimitMiddleware(tool_name="task")
        # cap on how many subagents get spawned in the first place.
        middleware=[
            TodoListMiddleware(),
            # Same web-search on/off toggle as the main agent — the root run's
            # `disabled_tools` config reaches here (LangGraph merges it in), so
            # turning web off also strips web_search/web_fetch from the
            # research subagent, not just the caller.
            ToolToggleMiddleware(),
            # Two-tier per-run caps, same as the main agent's (see graph.py) — soft
            # blocks-but-continues; past hard, the subagent writes its report with
            # what it has, in a last turn that may not call tools.
            *(SoftHardToolCallLimitMiddleware(tool_name=t, soft_limit=s, hard_limit=h)
              for t, (s, h) in RESEARCH_LIMITS.items()),
            # Appended last to match deepagents' own raw-SubAgent ordering
            # (it appends this after any custom middleware from the spec).
            HumanInTheLoopMiddleware(
                interrupt_on={"bash": {"allowed_decisions": ["approve", "reject"]}}
            ),
        ],
    ),
}
