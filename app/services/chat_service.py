import asyncio
import json
import logging
import time

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.types import Command

from app.agents.llm import build_llm_with_fallback
from app.agents.middleware import TOOL_CALL_LIMIT_EVENT
from app.agents.user_flows import is_flow_tool
from app.flows.compiler import FLOW_NODE_EVENT
from app.agents.pruning import prune_tool_results
from app.agents.tools.sandbox_manager import reset as reset_sandbox
from app.clients import conversation_client
from app.core import observability
from app.core.config import settings
from app.core.conversation_lock import Lease, conversation_locks
from app.core.run_control import run_control
from app.agents.tool_output import tool_output_text
from app.agents import user_models
from app.schemas.chat import AVAILABLE_MODELS, DEFAULT_MODEL
from app.services.interrupted import INTERRUPTED_NO_TEXT, INTERRUPTED_NOTE

logger = logging.getLogger(__name__)

# Looked up in stream() before touching the graph at all — refuses a new
# user-initiated turn once the conversation's last-known context_tokens
# (see Conversation.context_tokens) already reached/exceeded the selected
# model's real context window. Only guards brand-new messages, not
# resume()/resume_clarify() — those continue an already-in-progress turn (a
# pending bash-approval or ask_user clarify interrupt), and blocking
# mid-flight risks the same kind of dangling-tool_call/orphaned-interrupt
# problems this session's interrupt-safety work went out of its way to avoid.
_MODEL_CONTEXT_WINDOW = {m["id"]: m["contextWindow"] for m in AVAILABLE_MODELS}

# Fraction of the real context window at which stream() proactively compacts
# old history (see _compact_conversation), instead of only ever hitting the
# hard block below. 0.8 leaves 20% of the window as headroom for the reply
# plus at least one more turn before the next check — not copied from
# anywhere, just a number with a stated reason, same spirit as the old
# COMPACT_ELIGIBILITY_TOKENS this replaces (git log 572e254).
COMPACT_THRESHOLD_RATIO = 0.8

# On a small window 20% is not much room, so the trigger also leaves a fixed
# reserve: the reply the model may write plus 65,536 tokens of headroom, the
# same shape as DeepSeek Harness's compaction trigger
# (floor(min(W * 0.8, W - O - 65,536))). Only models below ~400K tokens notice.
COMPACT_HEADROOM_TOKENS = 65_536
RESERVED_OUTPUT_TOKENS = 16_384


# The newest part of the conversation stays word for word when older history is
# condensed (DeepSeek Harness keeps the newest 16% of the usable window), so the
# model still has the exact last turn(s) instead of only a summary of them.
COMPACT_RETAIN_RATIO = 0.16


def split_for_compaction(messages: list, budget_tokens: int) -> tuple[list, list]:
    """(older messages to condense, newest messages to keep as they are).

    The kept part starts at a user message, never in the middle of a turn, so a
    tool call and its result are never separated. It is the longest such tail
    that fits in `budget_tokens`; if even the last turn does not fit, nothing is
    kept and everything is condensed."""
    for i, m in enumerate(messages):
        if i > 0 and isinstance(m, HumanMessage) and count_tokens_approximately(messages[i:]) <= budget_tokens:
            return messages[:i], messages[i:]
    return messages, []


def compact_threshold(context_window: int) -> int:
    """Total tokens above which stream() compacts before the next message."""
    reserve = RESERVED_OUTPUT_TOKENS + COMPACT_HEADROOM_TOKENS
    return max(context_window // 2, int(min(context_window * COMPACT_THRESHOLD_RATIO, context_window - reserve)))

# ---------------------------------------------------------------------------
# Thinking parser — splits <think>…</think> out of the content stream
# ---------------------------------------------------------------------------


class ThinkingParser:
    """State machine that routes streaming tokens to thinking_token vs token.

    None of the currently configured models (schemas/chat.py AVAILABLE_MODELS)
    actually need this: verified live that all of them report reasoning via
    OpenRouter's dedicated `reasoning` delta field (rescued into
    additional_kwargs by ThinkingChatOpenAI, handled above at the
    `additional_kwargs.get("reasoning")` check) — chunk.content never contains
    <think>/</think> for any of them, so this just passes content through
    unchanged. Kept for a future model that inlines <think>...</think> in
    content instead of using a dedicated field."""

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self):
        self.in_thinking = False
        self._buf = ""

    def feed(self, text: str) -> list[dict]:
        self._buf += text
        events: list[dict] = []
        tag = self._CLOSE if self.in_thinking else self._OPEN

        while True:
            idx = self._buf.find(tag)
            if idx != -1:
                content = self._buf[:idx]
                if content:
                    events.append(
                        {
                            "type": "thinking_token" if self.in_thinking else "token",
                            "content": content,
                        }
                    )
                self._buf = self._buf[idx + len(tag) :]
                self.in_thinking = not self.in_thinking
                tag = self._CLOSE if self.in_thinking else self._OPEN
            else:
                # Hold back any bytes that could be the start of the tag
                for i in range(1, len(tag)):
                    if self._buf.endswith(tag[:i]):
                        safe = self._buf[:-i]
                        if safe:
                            events.append(
                                {
                                    "type": (
                                        "thinking_token"
                                        if self.in_thinking
                                        else "token"
                                    ),
                                    "content": safe,
                                }
                            )
                        self._buf = self._buf[-i:]
                        return events
                if self._buf:
                    events.append(
                        {
                            "type": "thinking_token" if self.in_thinking else "token",
                            "content": self._buf,
                        }
                    )
                    self._buf = ""
                return events


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


# Tools whose result renders as its own inline block (viz / file chip), not a
# generic tool badge — suppress their tool_start/tool_chunk; tool_end turns
# the special JSON payload into a dedicated SSE event.
INLINE_RESULT_TOOLS = {"generate_visualization_svg", "present_file"}
TODO_TOOL = "write_todos"
# ask_user has its own dedicated clarify_request event (see _extract_hitl_events)
# instead of a generic tool badge — the interrupt it raises means its tool_end
# never fires in this streaming pass anyway (see resume_clarify's docstring).
HIDDEN_TOOLS = {"write_todos", "ask_user"}


def _nests(tool_name: str | None) -> bool:
    """A tool whose own steps are shown nested under its badge: a subagent (`task`) or one of the user's flows."""
    return tool_name == "task" or is_flow_tool(tool_name)


class ToolStartEventHandler:
    def handle(self, event: dict, task_run_id: str | None = None) -> list[dict]:
        if event["name"] == TODO_TOOL:
            # write_todos replaces the whole list each call — its input IS the
            # new state, so emit it directly instead of a generic tool badge.
            # No matching tool_end needed (still suppressed via HIDDEN_TOOLS):
            # there's nothing more useful in the output than what's here.
            raw_input = event["data"].get("input") or {}
            result: dict = {"type": "todo_update", "todos": raw_input.get("todos", [])}
            if task_run_id:
                result["task_run_id"] = task_run_id
            return [result]
        if event["name"] in INLINE_RESULT_TOOLS or event["name"] in HIDDEN_TOOLS:
            return []
        raw_input = dict(event["data"].get("input") or {})
        label = raw_input.pop("label", None)
        if label is None and is_flow_tool(event["name"]):
            flow_name = (event.get("metadata") or {}).get("flow_name")  # the name the user gave it, not flow_<slug>
            label = f"Flow: {flow_name}" if flow_name else None
        result = {"type": "tool_start", "name": event["name"], "label": label, "input": raw_input or None, "run_id": event.get("run_id", "")}
        if task_run_id:
            result["task_run_id"] = task_run_id
        return [result]


class ToolEndEventHandler:
    def handle(self, event: dict, task_run_id: str | None = None) -> list[dict]:
        if event["name"] in HIDDEN_TOOLS:
            return []
        raw = event["data"].get("output")
        if isinstance(raw, Command):
            # Tools that return a Command (e.g. the `task` subagent tool) wrap
            # their result in update={"messages": [ToolMessage(...)]} rather
            # than returning content directly.
            msgs = (raw.update or {}).get("messages") or []
            output = tool_output_text(msgs[0].content) if msgs and hasattr(msgs[0], "content") else str(raw)
        else:
            output = tool_output_text(raw.content) if hasattr(raw, "content") else str(raw)
        try:
            data = json.loads(output)
            if "__viz__" in data:
                viz_result: dict = {"type": "viz", "format": data["__viz__"], "code": data["code"], "title": data.get("title", "")}
                if task_run_id:
                    viz_result["task_run_id"] = task_run_id
                return [viz_result]
            if "__file__" in data:
                file_result: dict = {"type": "file", **data["__file__"]}
                if task_run_id:
                    file_result["task_run_id"] = task_run_id
                return [file_result]
        except Exception:
            pass
        # Viz tools suppress tool_start/tool_chunk, so FE has no badge yet.
        # Emit a synthetic tool_start first so FE can show the error output.
        events: list[dict] = []
        if event["name"] in INLINE_RESULT_TOOLS:
            raw_input = dict(event["data"].get("input") or {})
            label = raw_input.pop("label", None)
            start: dict = {"type": "tool_start", "name": event["name"], "label": label, "input": raw_input or None, "run_id": event.get("run_id", "")}
            if task_run_id:
                start["task_run_id"] = task_run_id
            events.append(start)
        end: dict = {"type": "tool_end", "name": event["name"], "output": output, "run_id": event.get("run_id", "")}
        # search_files hands its citations over as a ToolMessage artifact —
        # structured data for the UI that never enters the model's context.
        # Forwarded on the same event so the badge and its sources arrive
        # together, and so a subagent's trace (saved as these raw payloads)
        # keeps them for replay.
        artifact = getattr(raw, "artifact", None)
        if isinstance(artifact, dict) and artifact.get("citations"):
            end["citations"] = artifact["citations"]
        # view_image does the same with the pictures it opened: the ids of the
        # stored images, which the chat shows under the step.
        if isinstance(artifact, dict) and artifact.get("images"):
            end["images"] = artifact["images"]
        if task_run_id:
            end["task_run_id"] = task_run_id
        events.append(end)
        # A tool can hand files over the same way (draw_boxes: the picture it drew), shown
        # as the chips present_file makes, so the result is on screen whether or not the
        # model goes on to say so.
        if isinstance(artifact, dict):
            for f in artifact.get("files") or []:
                file_event: dict = {"type": "file", **f}
                if task_run_id:
                    file_event["task_run_id"] = task_run_id
                events.append(file_event)
        return events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    thread_id: str,
    user_id: str,
    thinking_effort: str = "high",
    model: str | None = None,
    subagent_model: str | None = None,
    web_search: bool = True,
    unattended: bool = False,
    model_info: dict | None = None,
    file_spaces: list[dict] | None = None,
) -> dict:
    # start_research is research done on the web too — only the tool that
    # begins a task is switched off; checking or cancelling one already
    # running stays available.
    disabled = [] if web_search else ["web_search", "web_fetch", "start_research"]
    if unattended:
        # ask_user interrupts for a human reply that will never come. Left
        # enabled, the run ends there with no answer, and a benchmark records
        # that as the agent failing the question rather than as the harness
        # having no way to respond.
        disabled = [*disabled, "ask_user"]
    return {
        "configurable": {
            "thread_id": thread_id,
            # Scopes the per-user memory store namespace — see
            # app/agents/memory.py's _user_memory_namespace.
            "user_id": user_id,
            "thinking_effort": thinking_effort,
            "model": model,
            "subagent_model": subagent_model,
            # Read by ToolToggleMiddleware (main agent + research subagent).
            "disabled_tools": disabled,
            # Read by _bash_needs_approval (app/agents/graph.py).
            "unattended": unattended,
            # The departments whose documents the workspace tools may read (app/core/spaces.py: for_agent), from the
            # token of the request that started this turn.
            "file_spaces": file_spaces or [],
            # What a model the user added looks like to the rest of the turn (user_models.info): whether it sees
            # images, its context window, and that the server's fallback models must not answer for it. Never its key.
            **(model_info or {}),
        },
        # Langfuse. Both are empty/absent unless tracing is configured, so
        # this is the only place in the request path that knows about it —
        # the graph, the middleware and the tools stay unaware.
        "callbacks": observability.callbacks(),
        "metadata": observability.trace_metadata(thread_id, user_id),
        "recursion_limit": 200,
    }


def _extract_hitl_events(state) -> list[str]:
    """Return serialised hitl_request/clarify_request SSE lines for any pending
    interrupts — two different shapes share the same graph.aget_state().interrupts
    list: HumanInTheLoopMiddleware's bash approval ("action_requests") and the
    ask_user tool's own raw interrupt() call ("question"), each resumed via a
    different endpoint (/chat/resume vs /chat/resume_clarify) since they expect
    different Command(resume=...) payload shapes.
    """
    events = []
    for interrupt in getattr(state, "interrupts", ()):
        value = interrupt.value
        if isinstance(value, dict) and "action_requests" in value:
            events.append(json.dumps({
                "type": "hitl_request",
                "actions": value["action_requests"],
                "review_configs": value.get("review_configs", []),
            }))
        elif isinstance(value, dict) and "question" in value:
            events.append(json.dumps({
                "type": "clarify_request",
                "question": value["question"],
                "options": value.get("options"),
            }))
    return events


async def _compact_conversation(graph, config: dict, model: str | None) -> int | None:
    """Replace the older prior messages with one summary, keeping the newest
    COMPACT_RETAIN_RATIO of the window verbatim. Called from stream()
    before a brand-new user message when conv.context_tokens + the new
    message would cross COMPACT_THRESHOLD_RATIO of the model's real context
    window — see stream() for the trigger check.

    Unlike the SummarizationMiddleware/SummarizationToolMiddleware pair this
    project removed on 2026-08-18 (git log 572e254), this is a plain function
    chat_service calls directly — no model-facing tool, no second eligibility
    check inside create_agent()'s own middleware chain. That distinction is
    the whole point: the removed version's actual bug was that its automatic
    path trusted token_counter(messages) (a local, approximate count that
    can't see the system prompt and undercounts non-English text) while its
    manual path trusted usage_metadata on the last AIMessage (the real
    provider count) — two different numbers that drifted apart, so a
    conversation eligible by one measure read as "nothing to compact yet" by
    the other. Here there is exactly one number for "how big is the history
    right now" — conv.context_tokens, the same provider-reported total
    _usage_event already feeds everywhere else in this file — so that
    specific drift can't recur. (stream()'s pre-check also factors in a local
    estimate of the brand-new message, which is unavoidably approximate since
    it hasn't been sent yet — but that estimate only decides *when* to call
    this function, never what this function measures once it runs.)

    Returns the new context_tokens for the replacement summary (from the
    summarization call's own real usage_metadata), or None if nothing was
    compacted (too little history yet, an interrupt is pending, or the model
    returned an empty summary) — callers should treat None as "state
    unchanged" and fall through to whatever check runs next.
    """
    state = await graph.aget_state(config)

    # Never touch state while a HITL/clarify interrupt is unresolved — the
    # exact landmine the old compact() hit (see this file's git history):
    # mutating `messages` here would silently orphan that interrupt's pending
    # tool_call, with no ToolMessage ever generated for it, breaking the
    # thread on the next real turn. stream() only calls this before a
    # brand-new message, where no interrupt should be pending — but a direct
    # API call bypassing the frontend's own guard could still race one, so
    # this checks for real rather than trusting the caller.
    if _extract_hitl_events(state):
        return None

    old_messages = state.values.get("messages", [])
    # Nothing worth condensing yet — e.g. a fresh thread, or one message so far.
    if len(old_messages) < 2:
        return None
    window = config.get("configurable", {}).get("model_context_window") or _MODEL_CONTEXT_WINDOW.get(model or DEFAULT_MODEL, 1_000_000)
    to_condense, kept = split_for_compaction(
        old_messages, int((window - RESERVED_OUTPUT_TOKENS) * COMPACT_RETAIN_RATIO))
    if len(to_condense) < 2:
        return None

    llm = build_llm_with_fallback(model)
    compact_request = [
        SystemMessage(content=(
            "You are condensing an earlier conversation so another model can "
            "resume the work with no loss of essential context. Condense the "
            "conversation below into a structured checkpoint. Use terse "
            "bullets, not prose paragraphs. Write exactly these sections, in "
            "order, using \"(none)\" for an empty one — never drop a section:\n\n"
            "## Primary Request and Intent\n"
            "## Key Technical Concepts\n"
            "## Files and Code\n"
            "## Errors and Fixes\n"
            "## Pending Work"
        )),
        # Bulk tool output adds little to a summary and can overflow the request.
        *prune_tool_results(to_condense),
        HumanMessage(content="Produce the checkpoint now, following exactly that structure."),
    ]
    # A direct llm.ainvoke(), not a graph run — deliberately bypasses
    # create_agent()'s tools/middleware entirely (including observability's
    # tracing callbacks on config; this call isn't traced). Keeping this
    # outside the agent loop is what makes "no second eligibility check" above
    # true: there's no middleware here that could decide independently
    # whether to run.
    summary = await llm.ainvoke(compact_request)
    summary_text = (summary.content or "").strip()
    if not summary_text:
        return None

    # Same framing dsh-compaction-basic (a different project's compaction
    # package) uses for its replacement message: a plain instruction, not a
    # claim that the model itself asked for this.
    checkpoint = (
        "This is an automatically generated checkpoint condensing an earlier "
        "span of the conversation to free up context. Treat it as established "
        "background and build on it without restating it.\n\n"
        f"<compacted-summary>\n{summary_text}\n</compacted-summary>"
    )

    # Messages LangGraph never assigned an id to can't be targeted by
    # RemoveMessage and are left in place — same residual-message risk the
    # original implementation had; not expected in practice since
    # create_agent()'s checkpointer assigns one to everything it persists.
    removals = [RemoveMessage(id=m.id) for m in old_messages if getattr(m, "id", None)]
    await graph.aupdate_state(
        config,
        # The kept messages are removed and added back after the summary, so the
        # order stays summary first; a copy without its id gets a new one. Tool
        # results keep their artifacts (stored image ids, sources) in the copy.
        {"messages": [*removals, HumanMessage(content=checkpoint),
                      *[m.model_copy(update={"id": None}) for m in kept]]},
        # as_node deliberately omitted — identical reasoning to this file's
        # other aupdate_state(config, {"context_tokens": ...}) call below:
        # as_node="model" reschedules the whole after_model middleware chain
        # as state.next, which conversation_service.get_messages() then reads
        # as "turn still pending" on every later reload.
    )

    usage = summary.usage_metadata or {}
    return usage.get("total_tokens") or None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


async def save_interrupted(graph, config: dict, text: str, thinking: str = "") -> bool:
    """Keep what was already on screen when the user pressed Stop.

    The answer being written when a run is cancelled belongs to no finished step, so LangGraph never
    checkpointed it: after a reload it was gone, and the next turn ("that is wrong, do it this way") reached a
    model that could not see what it had said. This adds it to the thread's messages as an AI message marked
    `interrupted`, with the reasoning streamed so far (shown on the page again after a reload; it is not sent back
    to the model). A tool call cut off while its arguments were still being written is dropped.

    Returns whether a message was added. Nothing is added when nothing had streamed, or when the state already
    ends in a finished answer (the run completed an instant before the cancellation arrived).
    """
    if not text.strip() and not thinking.strip():
        return False
    state = await graph.aget_state(config)
    messages = state.values.get("messages", [])
    if messages and isinstance(messages[-1], AIMessage) and not messages[-1].tool_calls:
        return False
    extra: dict = {"interrupted": True}
    if thinking.strip():
        extra["reasoning"] = thinking
    # as_node left out, as for the context_tokens write below: naming the model node would schedule its whole
    # after_model middleware chain again (see the long comment in _run_graph).
    content = text + INTERRUPTED_NOTE if text.strip() else INTERRUPTED_NO_TEXT
    await graph.aupdate_state(config, {"messages": [AIMessage(content=content, additional_kwargs=extra)]})
    return True


async def prune_thread(graph, thread_id: str) -> None:
    """Delete the thread's checkpoints except the newest, with the writes and blobs only they used.

    Nothing in the app reads checkpoint history, only the head (aget_state / aupdate_state), and every turn used
    to leave one checkpoint per step behind, each with a full copy of the message list. Best effort: a failed
    prune only leaves old rows in place.
    """
    if not settings.CHECKPOINT_KEEP_LATEST or getattr(graph, "checkpointer", None) is None:
        return
    try:
        await graph.checkpointer.aprune([thread_id], strategy="keep_latest")
    except Exception:  # noqa: BLE001
        logger.warning("could not prune checkpoints of %s", thread_id, exc_info=True)


class ChatService:
    def __init__(self) -> None:
        self._tool_start = ToolStartEventHandler()
        self._tool_end = ToolEndEventHandler()
        # thread_id -> the asyncio.Task actually driving that thread's current
        # graph run (see _run_graph's `_drive`) — lets stop() cancel a run
        # from a separate request without touching the ASGI/StreamingResponse
        # task that's yielding SSE lines for it.
        self._active_runs: dict[str, asyncio.Task] = {}
        # thread_id -> when that run began. Lets a cancel request that is only picked up some seconds after it was
        # made (see core/run_control.py) tell that it was meant for an earlier run of the same conversation.
        self._run_started: dict[str, float] = {}

    @staticmethod
    async def guarded(lease: Lease, stream):
        """Run `stream` (one of stream/resume/resume_clarify) while holding the conversation's lease."""
        lease.start_heartbeat()
        try:
            async for chunk in stream:
                yield chunk
        finally:
            lease.release()

    def active_runs(self) -> dict[str, float]:
        """{thread_id: when it began} for the runs going in this process."""
        return {t: self._run_started.get(t, 0.0) for t, task in self._active_runs.items() if not task.done()}

    async def drain(self, timeout: float) -> int:
        """Wait up to `timeout` seconds for the runs going in this process to finish (shutdown: a replica that is being
        replaced stops taking requests but lets its runs end). How many were still going at the deadline."""
        tasks = [t for t in self._active_runs.values() if not t.done()]
        if not tasks:
            return 0
        logger.info("waiting for %d chat run(s) to finish before shutting down", len(tasks))
        _, pending = await asyncio.wait(tasks, timeout=timeout)
        if pending:
            logger.warning("shutting down with %d chat run(s) still going", len(pending))
        return len(pending)

    async def stop(self, thread_id: str) -> bool:
        """Stop the run of this conversation wherever it is going: here, or in the replica that holds its lock (it is
        told through Redis, see core/run_control.py). False when nothing is running it."""
        if await self.stop_local(thread_id):
            return True
        owner = await conversation_locks.owner_of(thread_id)
        if owner and owner != run_control.instance:
            return await run_control.cancel_remote("run", thread_id, owner)
        return False

    async def stop_local(self, thread_id: str) -> bool:
        """Cancel the in-flight run for this thread, if this process has it.

        Cancelling the asyncio task stops token/tool-event streaming
        immediately. It does NOT, by itself, stop a `bash` call already in
        flight — that awaits OpenSandbox's own command execution, and
        cancelling the awaiting task only stops us from waiting on the
        result, not the sandbox-side execution itself. Killing the sandbox
        aborts that execution too. A fresh sandbox is created next time this
        thread runs a tool (ensure_sandbox); nothing on it persists once
        killed.
        """
        task = self._active_runs.get(thread_id)
        if task is None or task.done():
            return False
        task.cancel()
        await reset_sandbox(thread_id)
        return True

    @staticmethod
    def _task_tool_call_id(event: dict) -> str | None:
        """Extract a nesting tool's own tool_call_id from its on_tool_end event: from the Command a `task` returns, or
        the ToolMessage a flow does.

        This is the stable id (matches ToolMessage.tool_call_id in the persisted
        state) used to key a saved subagent trace — unlike run_id, which is a
        fresh UUID every time the graph replays/reloads.
        """
        raw = event["data"].get("output")
        if isinstance(raw, Command):
            msgs = (raw.update or {}).get("messages") or []
            if msgs and hasattr(msgs[0], "tool_call_id"):
                return msgs[0].tool_call_id
        return getattr(raw, "tool_call_id", None)

    @staticmethod
    def _usage_event(chunk) -> dict | None:
        # OpenRouter reports usage on a final, contentless chunk per LLM call.
        # A single agent turn may call the model multiple times (tool round-trips,
        # including hidden subagent calls), so the frontend collects one usage
        # entry per call rather than summing — each call's input_tokens already
        # includes all prior calls' context, so summing would double/triple-count
        # the repeated prefix.
        usage = chunk.usage_metadata
        if not usage:
            return None
        return {
            "type": "usage",
            "input_tokens": usage.get("input_tokens", 0) or 0,
            "output_tokens": usage.get("output_tokens", 0) or 0,
            "total_tokens": usage.get("total_tokens", 0) or 0,
        }

    def _handle_token(self, event: dict, parser: ThinkingParser, viz_indexes: set[int]) -> list[dict]:
        chunk = event["data"]["chunk"]
        events: list[dict] = []

        # Some OpenRouter models return reasoning in a dedicated field
        reasoning = chunk.additional_kwargs.get(
            "reasoning"
        ) or chunk.additional_kwargs.get("reasoning_content", "")
        if reasoning:
            events.append({"type": "thinking_token", "content": reasoning})

        # Regular content — route through <think> tag parser
        if chunk.content:
            events.extend(parser.feed(chunk.content))

        usage_event = self._usage_event(chunk)
        if usage_event:
            events.append(usage_event)

        # Tool call chunks — suppress for viz tools (they render as viz blocks, not badges)
        for tc in getattr(chunk, "tool_call_chunks", None) or []:
            name = tc.get("name") or ""
            index = tc.get("index", 0)
            args_delta = tc.get("args", "") or ""
            if name and name in INLINE_RESULT_TOOLS:
                viz_indexes.add(index)
            if name and name in HIDDEN_TOOLS:
                viz_indexes.add(index)
            if index in viz_indexes:
                continue
            if name or args_delta:
                events.append(
                    {
                        "type": "tool_chunk",
                        "index": index,
                        "name": name,
                        "args_delta": args_delta,
                    }
                )

        return events

    async def _run_graph(self, graph_input, config: dict, graph):
        """Yield SSE lines by streaming graph events, then emit any pending HITL interrupt.

        The actual graph run (`_drive` below) executes as its own asyncio.Task
        rather than being driven directly by this generator's `async for`.
        That's what makes it independently cancellable via stop(): cancelling
        `drive_task` only affects that task, leaving this generator (which is
        what the ASGI/StreamingResponse machinery actually iterates) free to
        keep running, notice the cancellation via the sentinel + `await
        drive_task` below, and still yield a clean `stopped`/`done` pair
        instead of the whole SSE response just dying mid-frame.
        """
        parser = ThinkingParser()
        viz_indexes: set[int] = set()
        hitl_lines: list[str] = []
        thread_id = config["configurable"]["thread_id"]
        # Maps every run_id inside a `task` subagent's execution tree back to
        # the run_id of the specific `task` call it descends from — lets the
        # UI nest each subagent's own tool calls (web_search, web_fetch, ...)
        # under the right "Delegating to sub-agent" badge instead of showing
        # them as flat, unrelated rows. Those tool calls are shown as-is —
        # they're complete, atomic events, safe to stream even with several
        # subagents running in parallel. Their raw model text/thinking tokens
        # are NOT shown: those arrive as many small deltas, and interleave
        # character-by-character into unreadable garbled text when multiple
        # subagents stream concurrently. Token usage is still surfaced either
        # way, for accurate cost accounting.
        subagent_task_root: dict[str, str] = {}
        # Accumulates each task's nested tool_start/tool_end payloads, saved to
        # Postgres once that task finishes (keyed by its tool_call_id) purely
        # so the user can still see them after a reload — never read back into
        # the model's own context on later turns.
        nested_events_by_task: dict[str, list[dict]] = {}
        # Last top-level (non-subagent) LLM call's total_tokens seen this run —
        # overwritten, not summed, since it's meant to approximate *current*
        # context size (see Conversation.context_tokens), not lifetime spend.
        # A turn may call the model multiple times (tool round-trips); the
        # last one reflects the fullest/most current view of the thread.
        last_context_tokens: int | None = None
        stopped = False
        # The words (and reasoning) the top-level model has streamed in the call that is still going. Cleared when
        # a call starts and when it ends (a finished call is in the checkpoint by itself); what is left when the user
        # stops the run is what save_interrupted keeps.
        partial = {"text": "", "thinking": ""}

        # Generated and sent before the run has produced anything, not read
        # back from Langfuse afterward — trace_as() (below) pins the actual
        # trace to this exact id, so the frontend already has what
        # create_score() needs to attach feedback (thumbs up/down) the
        # instant this message exists, not after a round trip to ask
        # Langfuse what id it picked. See app/core/observability.py.
        trace_id = observability.new_trace_id()
        yield f"data: {json.dumps({'type': 'trace_id', 'trace_id': trace_id})}\n\n"

        queue: asyncio.Queue = asyncio.Queue()
        _DONE = object()

        async def _drive():
            try:
                # A plain `with`, not `async with` — verified live against
                # the installed SDK: trace_as's underlying context manager
                # only supports the sync protocol, even though everything
                # inside this block is async. See trace_as's docstring.
                with observability.trace_as(trace_id):
                    async for ev in graph.astream_events(
                        graph_input, config=config, version="v2", durability=settings.CHECKPOINT_DURABILITY
                    ):
                        await queue.put(ev)
            finally:
                await queue.put(_DONE)

        drive_task = asyncio.ensure_future(_drive())
        self._active_runs[thread_id] = drive_task
        self._run_started[thread_id] = time.time()
        # The browser may go away mid-run without stopping it; the conversation stays locked until it is done.
        conversation_locks.hold_until(thread_id, drive_task)

        try:
            while True:
                event = await queue.get()
                if event is _DONE:
                    break
                run_id = event.get("run_id", "")
                parent_ids = event.get("parent_ids") or []
                is_task_start = event["event"] == "on_tool_start" and _nests(event.get("name"))
                is_task_end = event["event"] == "on_tool_end" and _nests(event.get("name"))
                task_run_id = next((subagent_task_root[pid] for pid in parent_ids if pid in subagent_task_root), None)

                if is_task_start:
                    subagent_task_root[run_id] = run_id
                elif task_run_id:
                    subagent_task_root[run_id] = task_run_id

                results: list[dict] | None = None

                if event["event"] == "on_chat_model_stream":
                    if task_run_id:
                        usage_event = self._usage_event(event["data"]["chunk"])
                        results = [usage_event] if usage_event else None
                    else:
                        results = self._handle_token(event, parser, viz_indexes) or None
                        for r in list(results or ()):
                            if r.get("type") == "token":
                                partial["text"] += r["content"]
                            elif r.get("type") == "thinking_token":
                                partial["thinking"] += r["content"]
                            if r.get("type") == "usage":
                                last_context_tokens = r["total_tokens"]
                                # Tell the frontend's gauge now, not at the end
                                # of the turn — a turn with many tool round-trips
                                # otherwise leaves it stale for minutes. Also
                                # persisted right away (best-effort), so a Stop,
                                # an error or a dropped connection mid-turn still
                                # leaves the real number for the next page load,
                                # which reads Conversation.context_tokens.
                                results.append({"type": "context_tokens", "tokens": last_context_tokens})
                                try:
                                    await conversation_client.set_context_tokens(thread_id, last_context_tokens)
                                except Exception:
                                    logger.warning("could not save context tokens", exc_info=True)
                elif event["event"] in ("on_chat_model_start", "on_chat_model_end") and not task_run_id:
                    partial["text"] = partial["thinking"] = ""
                elif event["event"] == "on_tool_start":
                    results = self._tool_start.handle(event, task_run_id=task_run_id)
                    if task_run_id:
                        nested_events_by_task.setdefault(task_run_id, []).extend(results)
                    elif is_task_start and is_flow_tool(event.get("name")) and results and results[0].get("label"):
                        # The flow's name is in no message (its call's arguments are the flow's inputs): kept first in
                        # its trace, so that a reload labels the badge the same way (conversation_service).
                        nested_events_by_task.setdefault(run_id, []).append({"type": "flow", "label": results[0]["label"]})
                elif event["event"] == "on_tool_end":
                    results = self._tool_end.handle(event, task_run_id=task_run_id)
                    if task_run_id:
                        nested_events_by_task.setdefault(task_run_id, []).extend(results)
                    elif is_task_end:
                        trace = nested_events_by_task.get(run_id)
                        tool_call_id = self._task_tool_call_id(event)
                        if trace and tool_call_id:
                            await conversation_client.save_subagent_trace(thread_id, tool_call_id, trace)
                elif event["event"] == "on_custom_event" and event.get("name") == FLOW_NODE_EVENT and task_run_id:
                    # A step of a flow the model called, nested under that flow's badge, and kept with its other
                    # nested events so a reload shows it too.
                    data = event["data"]
                    step = {"type": "flow_node", "node": data.get("node"), "node_type": data.get("type"),
                            "phase": data.get("phase"), "task_run_id": task_run_id}
                    if "output" in data:
                        step["output"] = data["output"]
                    results = [step]
                    nested_events_by_task.setdefault(task_run_id, []).append(step)
                elif event["event"] == "on_custom_event" and event.get("name") == TOOL_CALL_LIMIT_EVENT:
                    limit_event = {"type": "tool_limit", **event["data"]}
                    if task_run_id:
                        limit_event["task_run_id"] = task_run_id
                    results = [limit_event]
                    if task_run_id:
                        nested_events_by_task.setdefault(task_run_id, []).extend(results)

                if results:
                    for data in results:
                        yield f"data: {json.dumps(data)}\n\n"

            # _drive already finished (put _DONE) by this point, but its task
            # may not have transitioned to done/cancelled/exception state the
            # instant that happened — await it (not just check .cancelled())
            # to avoid that race, and to re-raise any real exception it hit.
            try:
                await drive_task
            except asyncio.CancelledError:
                stopped = True

            if not stopped:
                # After normal stream completion check for a pending HITL interrupt
                state = await graph.aget_state(config)
                hitl_lines = _extract_hitl_events(state)

                # Lets the message-feedback control survive a reload — the
                # live SSE trace_id event above already covers the message
                # while it's still on screen; conversation_service.py reads
                # this back the same way it already reads additional_kwargs
                # for other per-message extras (see serialize_messages).
                # Best-effort: a save failure here shouldn't break an
                # otherwise-successful turn, just mean feedback isn't
                # available for this one message after a reload.
                messages = state.values.get("messages", [])
                if messages and isinstance(messages[-1], AIMessage) and messages[-1].id:
                    try:
                        await conversation_client.save_message_trace(
                            thread_id, messages[-1].id, trace_id
                        )
                    except Exception:
                        logger.warning("could not save message trace", exc_info=True)

                if last_context_tokens is not None:
                    await conversation_client.set_context_tokens(thread_id, last_context_tokens)
                    # Mirrored into the checkpoint itself too (ContextTokensMiddleware
                    # in app/agents/middleware.py contributes this state key) — but
                    # ONLY when there's no pending interrupt. Verified live: calling
                    # aupdate_state(..., as_node=...) at all — even with as_node
                    # omitted, not just as_node="model" — silently clears
                    # state.interrupts for any interrupt still unresolved at that
                    # point (no error, no trace), orphaning its tool_call with no
                    # ToolMessage ever generated for it. hitl_lines was already
                    # captured above from the pre-corruption state, so this turn's
                    # hitl_request/clarify_request still reaches the frontend fine
                    # either way — but skip the checkpoint write so the *next*
                    # resume()/resume_clarify() still finds the real interrupt
                    # intact. The Postgres column is unaffected either way (plain
                    # SQL, no graph-state interaction) so it's still updated
                    # unconditionally.
                    #
                    # as_node deliberately omitted (not "model"): passing
                    # as_node="model" makes LangGraph treat this update as if the
                    # model node had just run, which reschedules the *entire*
                    # after_model middleware chain as state.next — even though
                    # that chain already ran for real earlier this turn. Verified
                    # live: a fully-completed, non-interrupted turn ended up with
                    # state.next == ('SoftHardToolCallLimitMiddleware[task].after_model',)
                    # after this call, which conversation_service.get_messages()
                    # reads as `is_pending=True`, making every single reload show
                    # a spurious "Response was interrupted" banner. Omitting
                    # as_node lets LangGraph infer "the last node that updated
                    # the state" instead (per aupdate_state's own docstring),
                    # which correctly leaves state.next empty for a truly-finished
                    # turn.
                    if not hitl_lines:
                        await graph.aupdate_state(config, {"context_tokens": last_context_tokens})
                    yield f"data: {json.dumps({'type': 'context_tokens', 'tokens': last_context_tokens})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            if self._active_runs.get(thread_id) is drive_task:
                del self._active_runs[thread_id]
                self._run_started.pop(thread_id, None)

        if stopped:
            try:
                await save_interrupted(graph, config, partial["text"], partial["thinking"])
            except Exception:  # noqa: BLE001 — losing the partial answer must not break the stop itself
                logger.warning("could not save the interrupted answer", exc_info=True)
            yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
        else:
            for line in hitl_lines:
                yield f"data: {line}\n\n"
        # Last, after every state write of this turn (save_interrupted, context_tokens); and not while another
        # request has a run going on the same thread, which has checkpoints of its own in flight.
        if thread_id not in self._active_runs:
            await prune_thread(graph, thread_id)
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    async def stream(
        self,
        thread_id: str,
        content: str,
        graph,
        user_id: str,
        thinking_effort: str = "high",
        model: str | None = None,
        subagent_model: str | None = None,
        web_search: bool = True,
        unattended: bool = False,
        file_spaces: list[dict] | None = None,
    ):
        try:
            model_info = await user_models.info(user_id, model)
        except user_models.UserModelError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return
        context_window = model_info.get("model_context_window") or _MODEL_CONTEXT_WINDOW.get(model)
        if context_window:
            conv = await conversation_client.get_conversation(thread_id)
            if conv:
                # Only this new-message estimate is a local approximation —
                # there's no provider-reported number for a message not sent
                # yet. conv.context_tokens is the same real, provider-reported
                # total used everywhere else in this file (see _usage_event).
                # See _compact_conversation's docstring for why mixing two
                # *real* measurements (not this kind of one-sided estimate)
                # is what actually broke the old implementation.
                new_tokens_estimate = count_tokens_approximately([HumanMessage(content=content)])
                threshold = compact_threshold(context_window)
                if conv.context_tokens + new_tokens_estimate > threshold:
                    compact_config = _make_config(
                        thread_id, user_id, model=model, subagent_model=subagent_model, model_info=model_info
                    )
                    new_total = await _compact_conversation(graph, compact_config, model)
                    if new_total is not None:
                        await conversation_client.set_context_tokens(thread_id, new_total)
                        conv = await conversation_client.get_conversation(thread_id)

            # Final safety net: still refuse if, even after an attempted
            # compaction above, the real total is at/over the hard limit —
            # e.g. compaction was skipped (interrupt pending, empty summary)
            # or a single message is itself larger than the window.
            if conv and conv.context_tokens >= context_window:
                message = (
                    f"This conversation has reached {model}'s context window "
                    f"({context_window:,} tokens). Start a new conversation to continue."
                )
                yield f"data: {json.dumps({'type': 'error', 'message': message})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

        config = _make_config(thread_id, user_id, thinking_effort, model, subagent_model,
                              web_search, unattended, model_info, file_spaces)
        async for chunk in self._run_graph(
            {"messages": [HumanMessage(content=content)]}, config, graph
        ):
            yield chunk

    async def resume(
        self,
        thread_id: str,
        decision: str,
        graph,
        user_id: str,
        model: str | None = None,
        subagent_model: str | None = None,
        web_search: bool = True,
        file_spaces: list[dict] | None = None,
    ):
        """Resume a bash-approval HITL interrupt. HumanInTheLoopMiddleware
        expects a list of one decision dict per pending action_request —
        the same decision ("approve"/"reject") is applied to every pending
        action in this batch (no per-action granularity in the UI)."""
        try:
            model_info = await user_models.info(user_id, model)
        except user_models.UserModelError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return
        config = _make_config(thread_id, user_id, model=model, subagent_model=subagent_model, web_search=web_search,
                              model_info=model_info, file_spaces=file_spaces)

        # Count pending action_requests so we send exactly N decisions
        n = 1
        state = await graph.aget_state(config)
        for interrupt in getattr(state, "interrupts", ()):
            value = interrupt.value
            if isinstance(value, dict) and "action_requests" in value:
                n = len(value["action_requests"])
                break

        decisions = [{"type": decision}] * n
        async for chunk in self._run_graph(
            Command(resume={"decisions": decisions}), config, graph
        ):
            yield chunk

    async def resume_clarify(
        self,
        thread_id: str,
        answer: str,
        graph,
        user_id: str,
        model: str | None = None,
        subagent_model: str | None = None,
        web_search: bool = True,
        file_spaces: list[dict] | None = None,
    ):
        """Resume an ask_user interrupt. Unlike resume() above, the Command's
        resume value is the raw answer string itself — ask_user's interrupt()
        call returns exactly whatever value Command(resume=...) carries, since
        it's a bare langgraph interrupt(), not a decisions-list protocol."""
        try:
            model_info = await user_models.info(user_id, model)
        except user_models.UserModelError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return
        config = _make_config(thread_id, user_id, model=model, subagent_model=subagent_model, web_search=web_search,
                              model_info=model_info, file_spaces=file_spaces)
        async for chunk in self._run_graph(Command(resume=answer), config, graph):
            yield chunk


chat_service = ChatService()
