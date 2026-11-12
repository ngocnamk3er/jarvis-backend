"""Background research: a research task that outlives the chat turn.

The `task` subagent blocks the main agent until it finishes. These tools start
the same kind of research as an asyncio task in this process instead, return a
task id at once, and let the main agent look in on it later — in this turn or a
later one. The user can keep chatting meanwhile, and ask what it is doing.

Each task runs the research graph (research_graph.py) on a thread of its own in
the app's checkpointer. Two things follow from that:

- progress is just that thread's state: every step the graph takes is
  checkpointed, so a check reads what has been searched and fetched so far while
  the task is still running — no side table, no events to store;
- a restart loses the asyncio task but not what it had done. check_research then
  reports it as "interrupted" (or "success", if it had in fact finished) rather
  than the task vanishing.

Tasks run in this process, so a restart or deploy interrupts them. Running them
in a separate worker would remove that; the tool interface would not change.
"""

import asyncio
import contextvars
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated, Any, Awaitable, Callable, NotRequired, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import ToolRuntime
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from app.agents.research_graph import build_research_graph
from app.core import observability
from app.core.config import settings

logger = logging.getLogger(__name__)

# How many of a task's most recent tool calls `check_research` lists. The rest
# are summarised as a count — a long task would otherwise answer "what is it
# doing" with a page of what it did an hour ago.
_RECENT_STEPS = 8

# Run settings a task inherits from the conversation that started it, so the
# user's model choice and the web-search toggle apply to it too.
_INHERITED = ("model", "subagent_model", "thinking_effort", "disabled_tools", "user_id")

# Finished tasks kept in the registry, for their error text. Beyond this the
# oldest are dropped, and a status is judged from the checkpoint alone.
_KEEP_FINISHED = 200

# Statuses that are settled: nothing will change them, so a later check trusts
# what was recorded instead of re-deriving it.
_SETTLED = {"success", "error", "cancelled", "interrupted"}

_SYSTEM_PROMPT = """## Background research
`start_research` runs a research agent in the background and returns a task id straight away.
It keeps working while you and the user carry on — including across later messages.

- Use the `research` subagent through `task` when you need the findings to write your reply.
  Use `start_research` when the research is long, or the user asked for it in the background,
  or you can usefully answer something else first.
- After starting one, tell the user the task id and what it is looking into, then stop. Do not
  check on it straight away.
- Call `check_research` when the user asks how it is going or for the result. It reports the
  status, what the task has searched and fetched so far, and the final report once done. If it
  is still running, say what it is doing now and stop — never poll in a loop.
- A status you saw earlier is stale. Always call `check_research` (one task) or `list_research`
  (all of them, or when you have lost a task id) before saying where a task stands.
- A task cut short by a service restart shows as `interrupted`. Say so and offer to start it again.
- A `success` task with `stopped_early` ran out of its search budget before it wrote a report. It has
  no report: say so plainly, give what its steps show it covered, and offer a narrower follow-up.
- `cancel_research` stops a task the user no longer wants.
- Show the full task id every time."""


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _text_of(content: Any) -> str:
    """A message's content as plain text: a string, or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


def _describe_call(call: dict) -> str:
    args = call.get("args") or {}
    return str(args.get("label") or args.get("query") or args.get("url") or "")


def summarize_progress(values: dict | None) -> dict:
    """What a task has done so far, from its thread's state.

    Checkpointed after every step, so this is current while the task is still
    going — it's what lets a check answer "what is it doing" and not just "is it
    done". `plan` is the task's own todo list (TodoListMiddleware).
    """
    messages = (values or {}).get("messages") or []
    answered = {m.tool_call_id for m in messages if m.type == "tool"}
    steps = []
    for m in messages:
        if m.type != "ai":
            continue
        for call in m.tool_calls or []:
            if call.get("name") == "write_todos":
                continue
            steps.append(
                {"tool": call.get("name"), "what": _describe_call(call), "done": call.get("id") in answered}
            )
    plan = [{"task": t.get("content"), "status": t.get("status")} for t in (values or {}).get("todos") or []]
    return {
        "steps_total": len(steps),
        "steps_done": sum(1 for s in steps if s["done"]),
        "recent_steps": steps[-_RECENT_STEPS:],
        "plan": plan,
    }


def _finished_cleanly(values: dict | None, next_nodes: tuple) -> bool:
    """Whether a thread's last step was the model answering with no further tool
    calls and nothing is queued — i.e. the graph ran to its end."""
    messages = (values or {}).get("messages") or []
    if next_nodes or not messages:
        return False
    last = messages[-1]
    return last.type == "ai" and not last.tool_calls


# What SoftHardToolCallLimitMiddleware's last message says when it ends a run
# (app/agents/middleware.py). A run ended that way reaches its end, so it looks
# like a finished one — with the stop notice where the report should be.
_LIMIT_STOP = re.compile(r"tool call limit reached: hard limit exceeded")


def _stopped_early(values: dict | None) -> str | None:
    """The stop notice if the task was cut off by a tool-call limit, else None.
    Such a task has no report: it ran out of budget before writing one."""
    messages = (values or {}).get("messages") or []
    if not messages:
        return None
    text = _text_of(messages[-1].content)
    return text if _LIMIT_STOP.search(text) else None


def _final_report(values: dict | None) -> str:
    messages = (values or {}).get("messages") or []
    return _text_of(messages[-1].content) if messages else "(completed with no output)"


@dataclass
class _Job:
    task: asyncio.Task
    status: str = "running"
    error: str | None = None
    seq: int = field(default=0)


class ResearchRunner:
    """Owns the asyncio tasks of this process. One instance, `research_runner`."""

    def __init__(self) -> None:
        self._graph = None
        self._jobs: dict[str, _Job] = {}
        self._seq = 0

    def init(self, checkpointer, graph=None) -> None:
        """Called once from main.lifespan with the app's checkpointer. `graph`
        is for tests, which pass one with a scripted model."""
        self._graph = graph or build_research_graph(checkpointer)

    @property
    def ready(self) -> bool:
        return self._graph is not None

    def running(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status == "running")

    @staticmethod
    def thread_id(task_id: str) -> str:
        return f"research-{task_id}"

    def start(self, description: str, inherited: dict) -> str:
        """Launch a task and return its id. Raises RuntimeError when at the cap."""
        if self.running() >= settings.BACKGROUND_RESEARCH_MAX_RUNNING:
            raise RuntimeError(
                f"{self.running()} research tasks are already running (the limit is "
                f"{settings.BACKGROUND_RESEARCH_MAX_RUNNING}). Wait for one to finish or cancel one."
            )
        task_id = str(uuid.uuid4())
        parent = inherited.get("parent_thread_id") or task_id
        config = {
            "configurable": {
                **{k: inherited[k] for k in _INHERITED if inherited.get(k) is not None},
                "thread_id": self.thread_id(task_id),
                # No sandbox behind this run — see research_graph.py.
                "sandbox": False,
            },
            "recursion_limit": 100,
            # Langfuse: filed under the conversation that started it.
            "callbacks": observability.callbacks(),
            "metadata": observability.trace_metadata(parent, inherited.get("user_id") or ""),
        }
        self._seq += 1
        # An empty context, not the caller's. A task inherits the context it is
        # created in, and the one here is the main agent's tool call — whose
        # LangChain callbacks include the chat stream's event handler. Inherited,
        # every step the research agent takes (a web_search starting, a model
        # token) is reported into the chat stream of the turn that started it:
        # tool badges that never finish, and text interleaved into the main
        # answer. The run's own callbacks (Langfuse) are passed in `config`.
        job = _Job(
            task=asyncio.create_task(
                self._run(task_id, description, config), context=contextvars.Context()
            ),
            seq=self._seq,
        )
        self._jobs[task_id] = job
        self._prune()
        return task_id

    async def _run(self, task_id: str, description: str, config: dict) -> None:
        # `_jobs[task_id]` is set right after create_task() and this coroutine
        # only starts at its first await, so it is there by the time it's read.
        try:
            await self._graph.ainvoke({"messages": [{"role": "user", "content": description}]}, config=config)
        except asyncio.CancelledError:
            self._jobs[task_id].status = "cancelled"
            raise
        except Exception as e:  # noqa: BLE001 — a task's failure is its result, not the app's
            logger.warning("research task %s failed", task_id, exc_info=True)
            self._jobs[task_id].status = "error"
            self._jobs[task_id].error = f"{type(e).__name__}: {str(e)[:300]}"
        else:
            self._jobs[task_id].status = "success"

    def _prune(self) -> None:
        finished = sorted((j.seq, tid) for tid, j in self._jobs.items() if j.status != "running")
        for _, tid in finished[: max(0, len(finished) - _KEEP_FINISHED)]:
            del self._jobs[tid]

    def cancel(self, task_id: str) -> bool:
        """Stop a task running in this process. False if there is none."""
        job = self._jobs.get(task_id)
        if job is None or job.task.done():
            return False
        job.status = "cancelled"
        job.task.cancel()
        return True

    async def inspect(self, task_id: str, recorded: str) -> tuple[str, str | None, dict]:
        """(status, error, thread values) for a task.

        A task still in this process's registry is judged by it. Otherwise the
        process has restarted since it began, or it is long finished, and it is
        judged by what was recorded when last seen — or, if that was not yet
        settled, by whether its thread ran to its end.
        """
        snap = await self._graph.aget_state({"configurable": {"thread_id": self.thread_id(task_id)}})
        values = snap.values or {}
        job = self._jobs.get(task_id)
        if job is not None:
            return job.status, job.error, values
        if recorded in _SETTLED:
            return recorded, None, values
        return ("success" if _finished_cleanly(values, snap.next) else "interrupted"), None, values

    async def shutdown(self) -> None:
        """Stop every running task. Their threads stay in the checkpointer, so
        after the restart a check reports them as interrupted."""
        tasks = [j.task for j in self._jobs.values() if not j.task.done()]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


research_runner = ResearchRunner()


async def describe_tasks(tasks: dict) -> list[dict]:
    """A conversation's research tasks as the UI's side panel shows them, newest
    first: live status, what each has searched and fetched so far, its plan, and
    the finished report. Read straight from the runner and the task's thread,
    like check_research — the panel shows the same thing the agent would be told,
    without going through the model."""

    async def one(task: dict) -> dict:
        row = {
            "task_id": task["task_id"],
            "description": task["description"],
            "started_at": task["created_at"],
        }
        try:
            status, error, values = await research_runner.inspect(task["task_id"], task["status"])
        except Exception:  # noqa: BLE001 — one unreadable task must not blank the panel
            logger.warning("could not read research task %s", task["task_id"], exc_info=True)
            return {**row, "status": task["status"], **summarize_progress(None)}
        row.update(status=status, **summarize_progress(values))
        if status == "success":
            if stopped := _stopped_early(values):
                row["stopped_early"] = stopped
            else:
                row["report"] = _final_report(values)
        elif status == "error":
            row["error"] = error or "The research task failed."
        return row

    rows = await asyncio.gather(*(one(t) for t in tasks.values()))
    return sorted(rows, key=lambda r: r["started_at"], reverse=True)


class ResearchTask(TypedDict):
    task_id: str
    status: str
    description: str
    created_at: str
    last_checked_at: str
    last_updated_at: str


def _merge_tasks(existing: dict | None, update: dict) -> dict:
    return {**(existing or {}), **update}


class ResearchState(AgentState):
    """The tasks started in this conversation, persisted in the agent's own
    state so their ids survive context compaction and later turns."""

    research_tasks: Annotated[NotRequired[dict[str, ResearchTask]], _merge_tasks]


def _tracked(task_id: str, runtime: ToolRuntime) -> ResearchTask | str:
    task = (runtime.state.get("research_tasks") or {}).get(task_id.strip())
    return task or f"No research task with id {task_id!r}. Use list_research to see them."


def _updated(task: ResearchTask, status: str) -> ResearchTask:
    now = _now()
    return {
        **task,
        "status": status,
        "last_checked_at": now,
        "last_updated_at": now if status != task["status"] else task["last_updated_at"],
    }


def _command(runtime: ToolRuntime, message: str, task: ResearchTask) -> Command:
    return Command(
        update={
            "messages": [ToolMessage(message, tool_call_id=runtime.tool_call_id)],
            "research_tasks": {task["task_id"]: task},
        }
    )


@tool
async def start_research(description: str, runtime: ToolRuntime) -> str | Command:
    """Start a research task that runs in the background and return its task id at once.

    The task keeps going while you and the user continue — even across later messages.
    Use it for long research, or when the user asked for it in the background. Don't wait
    for it: tell the user the task id and stop.

    Args:
        description: Everything the research agent needs, self-contained — it sees nothing of
            this conversation. State the question, the angles to cover, and what the final
            report should contain.
    """
    if not research_runner.ready:
        return "Background research is not available right now."
    cfg = (runtime.config or {}).get("configurable", {})
    try:
        task_id = research_runner.start(
            description, {**cfg, "parent_thread_id": cfg.get("thread_id")}
        )
    except RuntimeError as e:
        return str(e)
    now = _now()
    task: ResearchTask = {
        "task_id": task_id,
        "status": "running",
        "description": description[:200],
        "created_at": now,
        "last_checked_at": now,
        "last_updated_at": now,
    }
    return _command(runtime, f"Started research task. task_id: {task_id}", task)


@tool
async def check_research(task_id: str, runtime: ToolRuntime) -> str | Command:
    """Check a background research task: its status, what it has searched and fetched so far,
    and the final report once it has finished.

    Call it when the user asks how a task is going or for its result. If it is still running,
    report what it is doing now and stop — don't call it again in a loop.

    Args:
        task_id: The task id `start_research` returned.
    """
    tracked = _tracked(task_id, runtime)
    if isinstance(tracked, str):
        return tracked
    try:
        status, error, values = await research_runner.inspect(tracked["task_id"], tracked["status"])
    except Exception as e:  # noqa: BLE001 — a failed read is an answer, not a crash
        logger.warning("could not read research task %s", task_id, exc_info=True)
        return f"Could not read the research task: {e}"

    result: dict[str, Any] = {
        "task_id": tracked["task_id"],
        "status": status,
        "started_at": tracked["created_at"],
        **summarize_progress(values),
    }
    if status == "success":
        if stopped := _stopped_early(values):
            result["stopped_early"] = stopped
        else:
            result["report"] = _final_report(values)
    elif status == "error":
        result["error"] = error or "The research task failed."
    elif status == "interrupted":
        result["note"] = "The service restarted while this ran, so it stopped part-way."
    return _command(runtime, json.dumps(result), _updated(tracked, status))


@tool
async def cancel_research(task_id: str, runtime: ToolRuntime) -> str | Command:
    """Stop a background research task the user no longer wants.

    Args:
        task_id: The task id `start_research` returned.
    """
    tracked = _tracked(task_id, runtime)
    if isinstance(tracked, str):
        return tracked
    if not research_runner.cancel(tracked["task_id"]):
        return f"Research task {tracked['task_id']} is not running, so there is nothing to cancel."
    return _command(
        runtime, f"Cancelled research task {tracked['task_id']}.", _updated(tracked, "cancelled")
    )


@tool
async def list_research(runtime: ToolRuntime) -> str | Command:
    """List every background research task in this conversation with its current status.

    Use it when the user asks about all their tasks, or when you no longer have a task id.
    """
    tasks: dict[str, ResearchTask] = runtime.state.get("research_tasks") or {}
    if not tasks:
        return "There are no research tasks in this conversation."

    async def live(task: ResearchTask) -> tuple[ResearchTask, str]:
        try:
            status, _, _ = await research_runner.inspect(task["task_id"], task["status"])
        except Exception:  # noqa: BLE001 — fall back to the last status recorded
            status = task["status"]
        return task, status

    rows = await asyncio.gather(*(live(t) for t in tasks.values()))
    lines = [
        f"- {t['task_id']}: {status} — {t['description']!r} (started {t['created_at']})"
        for t, status in rows
    ]
    return Command(
        update={
            "messages": [ToolMessage("\n".join(lines), tool_call_id=runtime.tool_call_id)],
            "research_tasks": {t["task_id"]: _updated(t, status) for t, status in rows},
        }
    )


class BackgroundResearchMiddleware(AgentMiddleware):
    """Adds the background-research tools and the instructions for them. Only
    wired in when settings.BACKGROUND_RESEARCH is on (see agents/graph.py)."""

    state_schema = ResearchState
    tools = [start_research, check_research, cancel_research, list_research]

    @staticmethod
    def _with_prompt(request: ModelRequest) -> ModelRequest:
        existing = request.system_message
        text = existing.text if existing else ""
        return request.override(system_message=SystemMessage(f"{text}\n\n{_SYSTEM_PROMPT}".strip()))

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        return handler(self._with_prompt(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        return await handler(self._with_prompt(request))
