import asyncio
import json
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.agents.middleware import TOOL_CALL_LIMIT_EVENT
from app.db import repository
from app.schemas.chat import AVAILABLE_MODELS

# Looked up in stream() before touching the graph at all — refuses a new
# user-initiated turn once the conversation's last-known context_tokens
# (see Conversation.context_tokens) already reached/exceeded the selected
# model's real context window. Only guards brand-new messages, not
# resume_clarify() — that continues an already-in-progress turn (a pending
# ask_user clarify interrupt), and blocking mid-flight risks the same kind
# of dangling-tool_call/orphaned-interrupt problems this session's
# interrupt-safety work went out of its way to avoid.
_MODEL_CONTEXT_WINDOW = {m["id"]: m["contextWindow"] for m in AVAILABLE_MODELS}

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


VIZ_TOOLS = {"generate_visualization_svg"}
TODO_TOOL = "write_todos"
# ask_user has its own dedicated clarify_request event (see _extract_clarify_events)
# instead of a generic tool badge — the interrupt it raises means its tool_end
# never fires in this streaming pass anyway (see resume_clarify's docstring).
HIDDEN_TOOLS = {"write_todos", "ask_user"}


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
        if event["name"] in VIZ_TOOLS or event["name"] in HIDDEN_TOOLS:
            return []
        raw_input = dict(event["data"].get("input") or {})
        label = raw_input.pop("label", None)
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
            output = msgs[0].content if msgs and hasattr(msgs[0], "content") else str(raw)
        else:
            output = raw.content if hasattr(raw, "content") else str(raw)
        try:
            data = json.loads(output)
            if "__viz__" in data:
                viz_result: dict = {"type": "viz", "format": data["__viz__"], "code": data["code"], "title": data.get("title", "")}
                if task_run_id:
                    viz_result["task_run_id"] = task_run_id
                return [viz_result]
        except Exception:
            pass
        # Viz tools suppress tool_start/tool_chunk, so FE has no badge yet.
        # Emit a synthetic tool_start first so FE can show the error output.
        events: list[dict] = []
        if event["name"] in VIZ_TOOLS:
            raw_input = dict(event["data"].get("input") or {})
            label = raw_input.pop("label", None)
            start: dict = {"type": "tool_start", "name": event["name"], "label": label, "input": raw_input or None, "run_id": event.get("run_id", "")}
            if task_run_id:
                start["task_run_id"] = task_run_id
            events.append(start)
        end: dict = {"type": "tool_end", "name": event["name"], "output": output, "run_id": event.get("run_id", "")}
        if task_run_id:
            end["task_run_id"] = task_run_id
        events.append(end)
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
) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
            # Scopes the per-user memory store namespace — see
            # app/agents/memory.py's _user_memory_namespace.
            "user_id": user_id,
            "thinking_effort": thinking_effort,
            "model": model,
            "subagent_model": subagent_model,
        },
        "recursion_limit": 200,
    }


def _extract_clarify_events(state) -> list[str]:
    """Return serialised clarify_request SSE lines for any pending interrupts —
    the ask_user tool's own raw interrupt() call ("question" in the interrupt
    value), resumed via /chat/resume_clarify with a bare Command(resume=<answer>).
    """
    events = []
    for interrupt in getattr(state, "interrupts", ()):
        value = interrupt.value
        if isinstance(value, dict) and "question" in value:
            events.append(json.dumps({
                "type": "clarify_request",
                "question": value["question"],
                "options": value.get("options"),
            }))
    return events


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ChatService:
    def __init__(self) -> None:
        self._tool_start = ToolStartEventHandler()
        self._tool_end = ToolEndEventHandler()
        # thread_id -> the asyncio.Task actually driving that thread's current
        # graph run (see _run_graph's `_drive`) — lets stop() cancel a run
        # from a separate request without touching the ASGI/StreamingResponse
        # task that's yielding SSE lines for it.
        self._active_runs: dict[str, asyncio.Task] = {}

    def stop(self, thread_id: str) -> bool:
        """Cancel the in-flight run for this thread, if any.

        Cancelling the asyncio task stops token/tool-event streaming
        immediately.
        """
        task = self._active_runs.get(thread_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    @staticmethod
    def _task_tool_call_id(event: dict) -> str | None:
        """Extract the `task` call's own tool_call_id from its on_tool_end event.

        This is the stable id (matches ToolMessage.tool_call_id in the persisted
        state) used to key a saved subagent trace — unlike run_id, which is a
        fresh UUID every time the graph replays/reloads.
        """
        raw = event["data"].get("output")
        if isinstance(raw, Command):
            msgs = (raw.update or {}).get("messages") or []
            if msgs and hasattr(msgs[0], "tool_call_id"):
                return msgs[0].tool_call_id
        return None

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
            if name and name in VIZ_TOOLS:
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
        """Yield SSE lines by streaming graph events, then emit any pending clarify interrupt.

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
        clarify_lines: list[str] = []
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

        queue: asyncio.Queue = asyncio.Queue()
        _DONE = object()

        async def _drive():
            try:
                async for ev in graph.astream_events(graph_input, config=config, version="v2"):
                    await queue.put(ev)
            finally:
                await queue.put(_DONE)

        drive_task = asyncio.ensure_future(_drive())
        self._active_runs[thread_id] = drive_task

        try:
            while True:
                event = await queue.get()
                if event is _DONE:
                    break
                run_id = event.get("run_id", "")
                parent_ids = event.get("parent_ids") or []
                is_task_start = event["event"] == "on_tool_start" and event.get("name") == "task"
                is_task_end = event["event"] == "on_tool_end" and event.get("name") == "task"
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
                        for r in results or ():
                            if r.get("type") == "usage":
                                last_context_tokens = r["total_tokens"]
                elif event["event"] == "on_tool_start":
                    results = self._tool_start.handle(event, task_run_id=task_run_id)
                    if task_run_id:
                        nested_events_by_task.setdefault(task_run_id, []).extend(results)
                elif event["event"] == "on_tool_end":
                    results = self._tool_end.handle(event, task_run_id=task_run_id)
                    if task_run_id:
                        nested_events_by_task.setdefault(task_run_id, []).extend(results)
                    elif is_task_end:
                        trace = nested_events_by_task.get(run_id)
                        tool_call_id = self._task_tool_call_id(event)
                        if trace and tool_call_id:
                            await repository.save_subagent_trace(thread_id, tool_call_id, trace)
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
                # After normal stream completion check for a pending clarify interrupt
                state = await graph.aget_state(config)
                clarify_lines = _extract_clarify_events(state)

                if last_context_tokens is not None:
                    await repository.set_context_tokens(thread_id, last_context_tokens)
                    # Mirrored into the checkpoint itself too (ContextTokensMiddleware
                    # in app/agents/middleware.py contributes this state key) — but
                    # ONLY when there's no pending interrupt. Verified live: calling
                    # aupdate_state(..., as_node="model") — even for a key that has
                    # nothing to do with messages — silently clears state.interrupts
                    # for any interrupt still unresolved at that point (no error, no
                    # trace), orphaning its tool_call with no ToolMessage ever
                    # generated for it. clarify_lines was already captured above from
                    # the pre-corruption state, so this turn's clarify_request still
                    # reaches the frontend fine either way — but skip the checkpoint
                    # write so the *next* resume_clarify() still finds the real
                    # interrupt intact. The Postgres column is unaffected either way
                    # (plain SQL, no graph-state interaction) so it's still updated
                    # unconditionally.
                    if not clarify_lines:
                        await graph.aupdate_state(config, {"context_tokens": last_context_tokens}, as_node="model")
                    yield f"data: {json.dumps({'type': 'context_tokens', 'tokens': last_context_tokens})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            if self._active_runs.get(thread_id) is drive_task:
                del self._active_runs[thread_id]

        if stopped:
            yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
        else:
            for line in clarify_lines:
                yield f"data: {line}\n\n"
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
    ):
        context_window = _MODEL_CONTEXT_WINDOW.get(model)
        if context_window:
            conv = await repository.get_conversation(thread_id)
            if conv and conv.context_tokens >= context_window:
                message = (
                    f"This conversation has reached {model}'s context window "
                    f"({context_window:,} tokens). Start a new conversation to continue."
                )
                yield f"data: {json.dumps({'type': 'error', 'message': message})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

        config = _make_config(thread_id, user_id, thinking_effort, model, subagent_model)
        async for chunk in self._run_graph(
            {"messages": [HumanMessage(content=content)]}, config, graph
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
    ):
        """Resume an ask_user interrupt. The Command's resume value is the raw
        answer string itself — ask_user's interrupt() call returns exactly
        whatever value Command(resume=...) carries, since it's a bare
        langgraph interrupt(), not a decisions-list protocol."""
        config = _make_config(thread_id, user_id, model=model, subagent_model=subagent_model)
        async for chunk in self._run_graph(Command(resume=answer), config, graph):
            yield chunk


chat_service = ChatService()
