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
from app.agents.middleware import SoftHardToolCallLimitMiddleware
from app.agents.prompt import build_system_prompt
from app.agents.tools import tools
from app.agents.subagents import RESEARCH_SUBAGENT

# Memory tools are scoped to write_file/edit_file/read_file only (not ls/glob/
# grep/execute) — this project deliberately removed the generic file-tool
# surface in favor of `bash` for all sandbox file work (see prompt.py).
# read_file is required by FilesystemMiddleware whenever edit_file is enabled
# (it raises ValueError otherwise — edit_file needs to read-before-diff). These
# three names are reused here only because MemoryMiddleware's own prompt tells
# the model to call `edit_file`; the descriptions below disambiguate them from
# sandbox file operations so the model doesn't confuse the two.
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
}


def build_graph(checkpointer=None, store=None):
    summarization_middleware = SummarizationMiddleware(
        model=build_llm_with_fallback(),
        backend=StateBackend,
        trigger=("tokens", 60000),
        keep=("messages", 20),
        trim_tokens_to_summarize=40000,
    )
    return create_agent(
        model=build_llm_with_fallback(),
        tools=tools,
        system_prompt=build_system_prompt(),
        checkpointer=checkpointer,
        store=store,
        middleware=[
            summarization_middleware,
            # compact_conversation tool — reuses summarization_middleware's own
            # engine/thresholds (shares its _summarization_event state key) so
            # a manual compact and the automatic 60k-token trigger stay
            # consistent. system_prompt=None: only the frontend's Compact
            # button calls this, the model is never nudged to call it itself.
            SummarizationToolMiddleware(summarization_middleware, system_prompt=None),
            MemoryMiddleware(backend=memory_backend, sources=["AGENTS.md"]),
            FilesystemMiddleware(
                backend=memory_backend,
                tools=["write_file", "edit_file", "read_file"],
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
        ],
    )
