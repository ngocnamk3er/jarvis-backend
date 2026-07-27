"""Sub-agent specs for the `task` tool (deepagents' SubAgentMiddleware)."""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, ToolCallLimitMiddleware, TodoListMiddleware
from deepagents.middleware.subagents import CompiledSubAgent

from app.agents.llm import build_llm_with_fallback
from app.agents.tools.web_search import web_search
from app.agents.tools.web_fetch import web_fetch
from app.agents.tools.get_current_time import get_current_time
from app.agents.tools.bash import bash
from app.agents.tools.generate_visualization_svg import generate_visualization_svg

_RESEARCH_SYSTEM_PROMPT = """You are a research subagent. The calling agent only sees your final
message, not your intermediate searches/fetches — so your last message must be a complete,
self-contained report.

## Tools
- `web_search` — find URLs and a quick answer; call in parallel across distinct queries when a
  topic has multiple independent angles
- `web_fetch` — read the full content of a specific URL as markdown; never fetch the same URL
  twice (not even with a different #anchor — anchors don't change what's returned)
- `get_current_time` — get the current date/time in a specific IANA timezone; use this instead of
  guessing "today's date" or converting times between cities yourself
- `bash` — only for processing data you already fetched (e.g. parsing a downloaded CSV, running a
  calculation over numbers found during research). Every call pauses for human approval before it
  runs, same as the main agent's `bash` — don't reach for it unless a search/fetch result actually
  needs computation.
- `generate_visualization_svg` — render a chart/diagram of your findings if the task asks for one.

## Workflow
1. Search from enough different angles to cover the topic (parallel calls for independent queries)
2. Fetch full pages only when a search summary isn't detailed enough
3. If two searches haven't found what you need, stop varying keywords — instead fetch the site of
   whoever would officially publish this data (government agency, project docs, standards body...)
4. Stop researching once you have enough to answer; do not over-research

## Output
Return ONLY your final synthesized report — no meta-commentary about your process ("I searched
for...", "let me also check..."). Structure it clearly (headings/bullets/table as fits the
content) and cite sources (URLs) for key claims. Do not ask the calling agent follow-up
questions — make reasonable assumptions and note them briefly if something is ambiguous."""

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
        tools=[web_search, web_fetch, get_current_time, bash, generate_visualization_svg],
        system_prompt=_RESEARCH_SYSTEM_PROMPT,
        name="research",
        # Caps each subagent *instance's* own tool use — independent of, and in
        # addition to, the main agent's ToolCallLimitMiddleware(tool_name="task")
        # cap on how many subagents get spawned in the first place.
        middleware=[
            TodoListMiddleware(),
            ToolCallLimitMiddleware(tool_name="web_search", run_limit=10, exit_behavior="continue"),
            ToolCallLimitMiddleware(tool_name="web_fetch", run_limit=10, exit_behavior="continue"),
            ToolCallLimitMiddleware(tool_name="bash", run_limit=5, exit_behavior="continue"),
            ToolCallLimitMiddleware(tool_name="generate_visualization_svg", run_limit=3, exit_behavior="continue"),
            # Appended last to match deepagents' own raw-SubAgent ordering
            # (it appends this after any custom middleware from the spec).
            HumanInTheLoopMiddleware(interrupt_on={"bash": {"allowed_decisions": ["approve", "reject"]}}),
        ],
    ),
}
