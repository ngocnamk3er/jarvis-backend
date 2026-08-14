from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    TodoListMiddleware,
)
from deepagents import FilesystemMiddleware, MemoryMiddleware
from deepagents.middleware.summarization import SummarizationMiddleware, SummarizationToolMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from deepagents.backends import StateBackend

from app.agents.llm import build_llm_with_fallback
from app.agents.memory import memory_backend
from app.agents.middleware import ContextTokensMiddleware, SoftHardToolCallLimitMiddleware
from app.agents.prompt import build_system_prompt
from app.agents.tools import tools
from app.agents.subagents import RESEARCH_SUBAGENT

# FilesystemMiddleware has no way to restrict which tools it registers on the
# deepagents version this project is pinned to (>=0.6.0 — verified against an
# actual 0.6.11 install: its __init__ has no `tools=` parameter at all, that
# was only added in a later 0.x release). It unconditionally creates all 7:
# ls, read_file, write_file, edit_file, glob, grep, execute. All 7 still only
# ever touch memory_backend (the per-user Postgres Store namespace) — never
# the sandbox filesystem `bash` uses — so there's no functional overlap, but
# ls/glob/grep/execute are pointless here (there's only ever the one
# "AGENTS.md" key, and StoreBackend doesn't implement execute at all — calling
# it just returns an error to the model, verified: building FilesystemMiddleware
# with a StoreBackend does not fail at construction time). Descriptions below
# disambiguate every one of them from sandbox file operations so the model
# doesn't confuse the two, and steer it away from the four pointless ones.
_MEMORY_TOOL_DESCRIPTIONS = {
    "write_file": (
        "Save a new personal-memory note about the current user (e.g. their "
        "preferences, recurring context) as 'AGENTS.md'. This is NOT for "
        "sandbox files — use bash for those. Persists across all of this "
        "user's conversations."
    ),
    "edit_file": (
        "Update the existing personal-memory note ('AGENTS.md') about the "
        "current user. This is NOT for sandbox files — use bash for those."
    ),
    "read_file": (
        "Read back the personal-memory note ('AGENTS.md') about the current "
        "user before editing it. This is NOT for sandbox files — use bash "
        "for those."
    ),
    "ls": "Lists personal-memory notes — there is only ever 'AGENTS.md'. Not for sandbox files.",
    "glob": "Searches personal-memory notes — there is only ever 'AGENTS.md', so this is rarely useful. Not for sandbox files.",
    "grep": "Searches inside 'AGENTS.md'. For anything else, use bash instead.",
    "execute": "Not supported for personal memory — this will always fail. Use bash for running commands.",
}


def build_graph(checkpointer=None, store=None, include_compact_tool: bool = False):
    """`include_compact_tool=True` adds the `compact_conversation` tool
    (SummarizationToolMiddleware) to the returned graph — kept OFF by default
    and only turned on for the dedicated graph used by chat_service.compact()
    (see app.state.compact_graph in main.py). Reasons this is a separate
    graph rather than always-on middleware:
    - Every middleware-provided tool gets bound to the model on *every* LLM
      call for that graph, regardless of whether it's ever invoked — an
      always-on compact_conversation would add its schema to every single
      request, forever, for a tool meant to be used rarely.
    - The tool's own built-in description ("Use this proactively when the
      conversation is getting long...") actively encourages the model to
      call it unprompted — verified live: system_prompt=None on
      SummarizationToolMiddleware only suppresses an additional nudge
      fragment, it does NOT touch the tool's own description, which the
      model still reads as part of normal function-calling. The intent here
      is strictly user-triggered (a Compact button), never model-initiated —
      so the only reliable fix is to not bind the tool to the model at all
      during normal chat, not just avoid encouraging it.
    Both graphs share the same checkpointer/store, so they operate on the
    exact same persisted thread state — this works because
    SummarizationToolMiddleware contributes only to the tool list (no
    before_model/after_model/before_agent hook of its own, verified via
    agent.get_graph()), so the two graphs have identical node topology and
    state schema regardless of this flag; only the bound tool set differs.
    """
    summarization_middleware = SummarizationMiddleware(
        model=build_llm_with_fallback(),
        backend=StateBackend,
        trigger=("tokens", 60000),
        keep=("messages", 20),
        trim_tokens_to_summarize=40000,
    )
    middleware = [
        summarization_middleware,
        ContextTokensMiddleware(),
    ]
    if include_compact_tool:
        # compact_conversation tool — reuses summarization_middleware's own
        # engine/thresholds (shares its _summarization_event state key) so
        # a manual compact and the automatic 60k-token trigger stay
        # consistent.
        middleware.append(SummarizationToolMiddleware(summarization_middleware, system_prompt=None))
    middleware += [
        MemoryMiddleware(backend=memory_backend, sources=["AGENTS.md"]),
        FilesystemMiddleware(
            backend=memory_backend,
            custom_tool_descriptions=_MEMORY_TOOL_DESCRIPTIONS,
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
    ]
    return create_agent(
        model=build_llm_with_fallback(),
        tools=tools,
        system_prompt=build_system_prompt(),
        checkpointer=checkpointer,
        store=store,
        middleware=middleware,
    )
