import json
from uuid import uuid4
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.types import Command

from app.agents.middleware import TOOL_CALL_LIMIT_EVENT
from app.db import repository

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
    content instead of using a dedicated field. If that model only emits the
    closing tag with no opening one, use ImplicitThinkingParser instead."""

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


class ImplicitThinkingParser:
    """State machine for models that start in thinking mode implicitly (no
    opening tag) and signal the end of thinking with a single </think> —
    e.g. some DeepSeek-R1 deployments. Unlike ThinkingParser, there's no
    <think> to wait for: every token is thinking_token until </think> is
    seen once, then everything after is the final answer. Not wired into
    _run_graph() by default — swap in for ThinkingParser if a configured
    model streams reasoning this way.

    reasoning_enabled must reflect whether reasoning was actually requested
    for this turn (e.g. thinking_effort != "none"). If reasoning is off, the
    model streams straight to its answer with no </think> at all — starting
    in_thinking=True in that case would misclassify the entire response as
    thinking_token forever, since the closing tag never arrives.
    """

    _CLOSE = "</think>"

    def __init__(self, reasoning_enabled: bool = True):
        self.in_thinking = reasoning_enabled
        self._buf = ""

    def feed(self, text: str) -> list[dict]:
        self._buf += text
        events: list[dict] = []

        if not self.in_thinking:
            if self._buf:
                events.append({"type": "token", "content": self._buf})
                self._buf = ""
            return events

        idx = self._buf.find(self._CLOSE)
        if idx != -1:
            content = self._buf[:idx]
            if content:
                events.append({"type": "thinking_token", "content": content})
            rest = self._buf[idx + len(self._CLOSE):]
            self.in_thinking = False
            self._buf = ""
            if rest:
                events.append({"type": "token", "content": rest})
            return events

        # Hold back any bytes that could be the start of </think>
        for i in range(1, len(self._CLOSE)):
            if self._buf.endswith(self._CLOSE[:i]):
                safe = self._buf[:-i]
                if safe:
                    events.append({"type": "thinking_token", "content": safe})
                self._buf = self._buf[-i:]
                return events

        if self._buf:
            events.append({"type": "thinking_token", "content": self._buf})
            self._buf = ""
        return events


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


# Mirrors SummarizationToolMiddleware's own eligibility gate: ~50% of
# graph.py's SummarizationMiddleware(trigger=("tokens", 60000)) — same
# formula the frontend's chat-input.tsx pill uses for its amber threshold.
# Checked ourselves in compact() *before* touching the graph at all, so an
# ineligible click costs zero LLM calls instead of one: even the "Nothing to
# compact yet" outcome normally still pays for the mandatory model call that
# follows any tool execution (tools -> model is unconditional in
# create_agent's graph). Worst case if this pre-check is ever imprecise
# (e.g. edge cases in how the library computes its "effective" message list)
# is a false negative that skips a real compaction — SummarizationToolMiddleware's
# own gate is still fully intact as the actual source of truth whenever we
# do proceed, so this can only under-trigger, never mis-compact.
COMPACT_ELIGIBILITY_TOKENS = 30000

VIZ_TOOLS = {"generate_visualization_svg"}
TODO_TOOL = "write_todos"
# ask_user has its own dedicated clarify_request event (see _extract_hitl_events)
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


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ChatService:
    def __init__(self) -> None:
        self._tool_start = ToolStartEventHandler()
        self._tool_end = ToolEndEventHandler()

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
        """Yield SSE lines by streaming graph events, then emit any HITL interrupt."""
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
        try:
            async for event in graph.astream_events(graph_input, config=config, version="v2"):
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

            # After normal stream completion check for pending HITL interrupt
            state = await graph.aget_state(config)
            hitl_lines = _extract_hitl_events(state)

            if last_context_tokens is not None:
                await repository.set_context_tokens(thread_id, last_context_tokens)
                # Mirrored into the checkpoint itself too (ContextTokensMiddleware
                # in app/agents/middleware.py contributes this state key) — but
                # ONLY when there's no pending interrupt. Verified live: calling
                # aupdate_state(..., as_node="model") — even for a key that has
                # nothing to do with messages — silently clears state.interrupts
                # for any interrupt still unresolved at that point (no error, no
                # trace), orphaning its tool_call with no ToolMessage ever
                # generated for it. hitl_lines was already captured above from
                # the pre-corruption state, so this turn's hitl_request/
                # clarify_request still reaches the frontend fine either way —
                # but skip the checkpoint write so the *next* resume() still
                # finds the real interrupt intact. The Postgres column is
                # unaffected either way (plain SQL, no graph-state interaction)
                # so it's still updated unconditionally.
                if not hitl_lines:
                    await graph.aupdate_state(config, {"context_tokens": last_context_tokens}, as_node="model")
                yield f"data: {json.dumps({'type': 'context_tokens', 'tokens': last_context_tokens})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        for line in hitl_lines:
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
        config = _make_config(thread_id, user_id, thinking_effort, model, subagent_model)
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
    ):
        config = _make_config(thread_id, user_id, model=model, subagent_model=subagent_model)

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
    ):
        """Resume an ask_user interrupt. Unlike resume() above, this Command's
        resume value is the raw answer string itself — ask_user's interrupt()
        call returns exactly whatever value Command(resume=...) carries, since
        it's a bare langgraph interrupt() rather than HumanInTheLoopMiddleware's
        decisions-list protocol."""
        config = _make_config(thread_id, user_id, model=model, subagent_model=subagent_model)
        async for chunk in self._run_graph(Command(resume=answer), config, graph):
            yield chunk

    async def compact(
        self,
        thread_id: str,
        graph,
        user_id: str,
        model: str | None = None,
        subagent_model: str | None = None,
    ):
        """Force-execute the compact_conversation tool (see
        SummarizationToolMiddleware in graph.py) without waiting for the model
        to decide to call it — triggered by the frontend's Compact button.

        LangGraph's agent graph routes model -> tools whenever the last
        message is an AIMessage with pending tool_calls (verified via
        agent.get_graph()). Seeding that state directly, tagged as if the
        `model` node produced it via as_node="model", makes the graph resume
        straight into executing the tool — the same mechanism LangGraph uses
        for manual state edits/time-travel, not a hack around the library.

        The trigger AIMessage + its ToolMessage result + the model's
        follow-up wrap-up reply are scrubbed from persisted history right
        after the run (see below) — they're an artifact of *how* compaction
        is invoked (has to go through real tool-calling for
        SummarizationToolMiddleware's engine to run), not a natural part of
        the conversation. Left in place, they'd sit in state["messages"]
        forever and get replayed to the model on every future turn, and to
        the user on every reload — confusing, since compact_conversation
        isn't even bound to the model outside of this one graph/run
        (app.state.compact_graph — see build_graph()'s docstring). The
        `_summarization_event` cutoff/summary this run recorded is untouched
        by the scrub (it's a separate state key, not part of messages), so
        the actual context-size reduction still fully applies going forward.
        """
        config = _make_config(thread_id, user_id, model=model, subagent_model=subagent_model)

        prior_state = await graph.aget_state(config)
        prior_values = prior_state.values
        prior_messages = prior_values.get("messages", [])

        # Refuse while a HITL/clarify interrupt is already pending — verified
        # live (minimal repro graph, not guessed): calling aupdate_state(...,
        # as_node="model") on top of an unresolved interrupt does NOT error,
        # it silently abandons the original pending tool_call (no ToolMessage
        # ever generated for it, no re-raised interrupt for it either) and
        # starts processing the newly-injected trigger instead. The orphaned
        # tool_call then has no matching ToolMessage, which OpenAI-compatible
        # APIs reject outright on the *next* real turn — breaking the thread
        # until manually fixed. The frontend already disables the Compact
        # button while pendingHitl/pendingClarify is set, but that's a
        # client-side guard only (direct API calls, or a race between click
        # and state sync, both bypass it) — this is the actual backing check.
        # Re-emitting the same hitl_request/clarify_request the frontend
        # would already have from the interrupted turn keeps it in sync
        # rather than just erroring silently.
        pending_lines = _extract_hitl_events(prior_state)
        if pending_lines:
            for line in pending_lines:
                yield f"data: {line}\n\n"
            yield f"data: {json.dumps({'type': 'error', 'message': 'Cannot compact while a previous action is still awaiting approval.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # Short-circuit before touching the graph at all if we're clearly
        # under SummarizationToolMiddleware's own eligibility gate — see
        # COMPACT_ELIGIBILITY_TOKENS. Without this, an ineligible click still
        # runs the tool for real (cheap — no LLM call) but then the graph's
        # tools -> model edge is unconditional, so a mandatory *real* LLM
        # call follows regardless of what the tool returned, just to have
        # the model announce a no-op. context_tokens is already exactly this
        # same "current usage" approximation (see ContextTokensMiddleware),
        # reused here for free since we already fetched state this call.
        current_context_tokens = prior_values.get("context_tokens", 0)
        if current_context_tokens < COMPACT_ELIGIBILITY_TOKENS:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Nothing to compact yet — conversation is within the token budget.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # SummarizationToolMiddleware._is_eligible_for_compaction only trusts
        # usage_metadata reported on the *last* AIMessage in state (see
        # _should_summarize_based_on_reported_tokens in langchain's
        # summarization middleware) — it has no computed-total fallback the
        # way the auto-trigger path does. Since this trigger message becomes
        # the new last AIMessage the moment it's written, leaving its
        # usage_metadata unset makes eligibility evaluate false unconditionally,
        # regardless of the real conversation size (verified live: eligible at
        # 30k+ real tokens still returned "nothing to compact" until this was
        # fixed). Carrying over the real last reply's usage/provider metadata
        # fixes that without misrepresenting anything — this trigger message
        # is a stand-in for "current usage state", not a real model reply.
        last_ai = next((m for m in reversed(prior_messages) if isinstance(m, AIMessage)), None)

        trigger_id = str(uuid4())
        tool_call_id = str(uuid4())
        await graph.aupdate_state(
            config,
            {
                "messages": [
                    AIMessage(
                        id=trigger_id,
                        content="",
                        tool_calls=[{"name": "compact_conversation", "args": {}, "id": tool_call_id}],
                        usage_metadata=last_ai.usage_metadata if last_ai else None,
                        response_metadata=last_ai.response_metadata if last_ai else {},
                    )
                ]
            },
            as_node="model",
        )
        # None input resumes execution from the current checkpoint state
        # (the tool call just seeded above) rather than sending new input.
        async for chunk in self._run_graph(None, config, graph):
            yield chunk

        state = await graph.aget_state(config)
        if state.interrupts:
            # Vanishingly unlikely (the follow-up wrap-up reply would have to
            # itself decide to call bash right after acknowledging the
            # compact), but if it happens, this new interrupt is real and
            # unresolved — skip the scrub rather than risk corrupting it the
            # same way (see the as_node="model" note below). Leaves the
            # trigger/tool/wrap-up messages in history uncleaned this one
            # time; harmless, just the cosmetic issue this scrub exists to
            # avoid.
            return
        messages = state.values.get("messages", [])
        to_remove = [trigger_id]
        tool_msg_index = next(
            (i for i, m in enumerate(messages) if isinstance(m, ToolMessage) and m.tool_call_id == tool_call_id),
            None,
        )
        if tool_msg_index is not None:
            to_remove.append(messages[tool_msg_index].id)
            follow_up_index = tool_msg_index + 1
            if follow_up_index < len(messages) and isinstance(messages[follow_up_index], AIMessage):
                to_remove.append(messages[follow_up_index].id)
        # trigger_id always found (we just wrote it) unless the run errored
        # before applying the update at all — RemoveMessage on an id that
        # was never part of state is a no-op, not an error, so this is safe
        # to call unconditionally.
        #
        # as_node is required here — verified live: aupdate_state on a real
        # multi-node graph (model + tools, not a trivial single-node test
        # graph) raises InvalidUpdateError("Ambiguous update, specify
        # as_node") for a bare messages-only update with no as_node, since
        # more than one node could plausibly have produced it. "model" is
        # correct: confirmed live it removes exactly the target messages,
        # leaves everything else untouched, and leaves state.next empty
        # (nothing pending) afterward. Safe to use even though we don't
        # re-enter the graph after this — this method already returned
        # early above if a HITL/clarify interrupt was pending, so there's
        # nothing left for as_node="model"'s routing recomputation to
        # disturb here.
        await graph.aupdate_state(
            config,
            {"messages": [RemoveMessage(id=mid) for mid in to_remove]},
            as_node="model",
        )


chat_service = ChatService()
