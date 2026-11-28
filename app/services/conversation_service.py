from fastapi import HTTPException
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

import logging

from app.agents.tool_output import tool_output_text
from app.agents.async_research import describe_tasks, research_runner
from app.agents.tools.sandbox_manager import reset as reset_sandbox
from app.clients import conversation_client
from app.db import attachment_repository, feedback_repository
from app.services.interrupted import INTERRUPTED_NO_TEXT, INTERRUPTED_NOTE

logger = logging.getLogger(__name__)


# What _compact_conversation's summary message starts with — it is a HumanMessage, but not something the user wrote.
SUMMARY_PREFIX = "Here is a summary of the conversation to date:"


async def _get_owned_conversation(conversation_id: str, user_id: str):
    conv = await conversation_client.get_conversation(conversation_id)
    if conv is None or conv.user_id != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv

# ── serialization ─────────────────────────────────────────────────────────────


def _replay_subagent_trace(trace: list[dict], task_tool_call_id: str) -> list[dict]:
    """Reconstruct tool parts from a saved subagent trace (the raw tool_start/
    tool_end SSE payloads, in original stream order) so they render nested
    under the `task` badge after reload, same as they did live. A flow's trace
    is replayed the same way, its steps (flow_node) included."""
    parts: list[dict] = []
    by_run_id: dict[str, dict] = {}
    todos_part: dict | None = None
    batch_id = 0
    prev_type = None
    for ev in trace:
        if ev["type"] == "tool_start":
            if prev_type != "tool_start":
                batch_id += 1
            tool: dict = {
                "name": ev["name"],
                "label": ev.get("label"),
                "input": ev.get("input"),
                "status": "done",
                "run_id": ev.get("run_id"),
                "task_run_id": task_tool_call_id,
                "parent_run_id": f"sub-{task_tool_call_id}-{batch_id}",
            }
            by_run_id[ev.get("run_id")] = tool
            parts.append({"type": "tool", "tool": tool})
        elif ev["type"] == "tool_end":
            tool = by_run_id.get(ev.get("run_id"))
            if tool:
                tool["output"] = ev.get("output")
                if ev.get("citations"):
                    tool["citations"] = ev["citations"]
                if ev.get("images"):
                    tool["images"] = ev["images"]
        elif ev["type"] == "todo_update":
            # write_todos replaces the whole list each call — keep updating
            # the same part in place rather than stacking a new one per call,
            # matching how the live stream shows a single updating widget.
            if todos_part is None:
                todos_part = {"type": "todos", "todos": ev.get("todos", []), "task_run_id": task_tool_call_id}
                parts.append(todos_part)
            else:
                todos_part["todos"] = ev.get("todos", [])
        elif ev["type"] == "viz":
            parts.append({
                "type": "viz", "format": ev["format"], "code": ev["code"], "title": ev.get("title", ""),
                "task_run_id": task_tool_call_id,
            })
        elif ev["type"] == "flow_node" and ev.get("phase") == "end":
            # A step of a flow the model called (chat_service's flow_node events); its start adds nothing once done.
            parts.append({
                "type": "flow_step", "node": ev.get("node"), "node_type": ev.get("node_type"), "status": "done",
                "output": ev.get("output"), "task_run_id": task_tool_call_id,
            })
        prev_type = ev["type"]
    return parts


def serialize_messages(
    messages: list,
    subagent_traces: dict[str, list[dict]] | None = None,
    message_traces: dict[str, str] | None = None,
    feedback_by_trace: dict[str, dict] | None = None,
) -> list[dict]:
    subagent_traces = subagent_traces or {}
    message_traces = message_traces or {}
    feedback_by_trace = feedback_by_trace or {}
    tool_outputs: dict[str, str] = {
        msg.tool_call_id: tool_output_text(msg.content)
        for msg in messages
        if isinstance(msg, ToolMessage)
    }
    # Citations live on the ToolMessage's artifact (see search_files), which
    # the checkpoint keeps — so sources reappear on reload exactly as they
    # streamed, without re-running the search.
    tool_citations: dict[str, list] = {
        msg.tool_call_id: msg.artifact["citations"]
        for msg in messages
        if isinstance(msg, ToolMessage)
        and isinstance(getattr(msg, "artifact", None), dict)
        and msg.artifact.get("citations")
    }

    # The images view_image opened, from the same artifact — the chat shows them
    # under the step after a reload as it did while streaming.
    tool_images: dict[str, list] = {
        msg.tool_call_id: msg.artifact["images"]
        for msg in messages
        if isinstance(msg, ToolMessage)
        and isinstance(getattr(msg, "artifact", None), dict)
        and msg.artifact.get("images")
    }

    # The files a tool handed over (see ToolEndEventHandler): shown as chips after its step.
    tool_files: dict[str, list] = {
        msg.tool_call_id: msg.artifact["files"]
        for msg in messages
        if isinstance(msg, ToolMessage)
        and isinstance(getattr(msg, "artifact", None), dict)
        and msg.artifact.get("files")
    }

    result: list[dict] = []
    pending_parts: list[dict] = []
    pending_usage_calls: list[dict] = []
    todos_part: dict | None = None  # this turn's own todos part, updated in place — see write_todos below
    # The id chat_service.py saved a trace under for this turn's final
    # AIMessage (see save_message_trace) — set on every AIMessage below so
    # it ends up holding the *last* one by the time flush() fires, same
    # message chat_service.py itself keyed the save on.
    pending_trace_id: str | None = None

    def flush():
        nonlocal todos_part, pending_trace_id
        if pending_parts:
            entry: dict = {"role": "assistant", "parts": list(pending_parts)}
            if pending_usage_calls:
                entry["usage"] = list(pending_usage_calls)
            if pending_trace_id:
                entry["trace_id"] = pending_trace_id
                fb = feedback_by_trace.get(pending_trace_id)
                if fb:
                    entry["rating"] = fb["rating"]
                    entry["comment"] = fb["comment"]
            result.append(entry)
            pending_parts.clear()
        pending_usage_calls.clear()
        todos_part = None
        pending_trace_id = None

    for msg in messages:
        if isinstance(msg, HumanMessage):
            flush()
            content = str(msg.content)
            if content.startswith(SUMMARY_PREFIX):
                result.append({
                    "role": "system",
                    "parts": [{"type": "text", "content": "Earlier messages in this conversation were summarized."}],
                })
                continue
            result.append(
                {
                    "role": "user",
                    "parts": [{"type": "text", "content": content}],
                }
            )
        elif isinstance(msg, AIMessage):
            msg_trace_id = message_traces.get(msg.id) if msg.id else None
            if msg_trace_id:
                pending_trace_id = msg_trace_id
            interrupted = bool(msg.additional_kwargs.get("interrupted"))
            reasoning = msg.additional_kwargs.get("reasoning", "")
            if reasoning:
                pending_parts.append({"type": "thinking", "content": reasoning, "isStreaming": False,
                                      **({"interrupted": True} if interrupted else {})})
            if msg.content:
                text = str(msg.content)
                if interrupted:                      # the markers are for the model, not for the page
                    if text == INTERRUPTED_NO_TEXT:
                        text = ""
                    elif text.endswith(INTERRUPTED_NOTE):
                        text = text[: -len(INTERRUPTED_NOTE)]
                if text:
                    pending_parts.append({"type": "text", "content": text, **({"interrupted": True} if interrupted else {})})
            usage = msg.usage_metadata
            if usage:
                pending_usage_calls.append(
                    {
                        "input_tokens": usage.get("input_tokens", 0) or 0,
                        "output_tokens": usage.get("output_tokens", 0) or 0,
                        "total_tokens": usage.get("total_tokens", 0) or 0,
                    }
                )
            tool_calls = msg.tool_calls or []
            # Tools in the same AIMessage were called in parallel — share a batch key
            batch_id = msg.id if len(tool_calls) > 1 else None
            for tc in tool_calls:
                if tc["name"] == "write_todos":
                    todos = (tc["args"] or {}).get("todos", [])
                    if todos_part is None:
                        todos_part = {"type": "todos", "todos": todos}
                        pending_parts.append(todos_part)
                    else:
                        todos_part["todos"] = todos
                    continue
                raw_output = tool_outputs.get(tc["id"], "")
                try:
                    import json as _json
                    data = _json.loads(raw_output)
                    if "__viz__" in data:
                        pending_parts.append({"type": "viz", "format": data["__viz__"], "code": data["code"], "title": data.get("title", "")})
                        continue
                    if "__file__" in data:
                        pending_parts.append({"type": "file", **data["__file__"]})
                        continue
                except Exception:
                    pass
                tool_input = dict(tc["args"] or {})
                tool_label = tool_input.pop("label", None)
                tool_part: dict = {
                    "name": tc["name"],
                    "label": tool_label,
                    "input": tool_input or None,
                    "output": raw_output,
                    "status": "done",
                }
                if batch_id:
                    tool_part["parent_run_id"] = batch_id
                if tc["id"] in tool_citations:
                    tool_part["citations"] = tool_citations[tc["id"]]
                if tc["id"] in tool_images:
                    tool_part["images"] = tool_images[tc["id"]]
                pending_parts.append({"type": "tool", "tool": tool_part})
                for f in tool_files.get(tc["id"], []):
                    pending_parts.append({"type": "file", **f})

                trace = subagent_traces.get(tc["id"])
                if trace:
                    tool_part["run_id"] = tc["id"]
                    if not tool_part["label"]:
                        # a flow's badge: its name was saved with its steps (chat_service)
                        tool_part["label"] = next((e.get("label") for e in trace if e.get("type") == "flow"), None)
                    pending_parts.extend(_replay_subagent_trace(trace, tc["id"]))

    flush()
    return result


# ── CRUD ──────────────────────────────────────────────────────────────────────


async def list_conversations(user_id: str):
    return await conversation_client.list_conversations(user_id)


async def create_conversation(title: str, user_id: str):
    return await conversation_client.create_conversation(title, user_id)


async def delete_conversation(graph, conversation_id: str, user_id: str) -> None:
    await _get_owned_conversation(conversation_id, user_id)
    await reset_sandbox(conversation_id)
    try:
        await attachment_repository.delete_by_thread(conversation_id)
    except Exception:  # noqa: BLE001 — leftover images must not block deleting the conversation
        logger.warning("could not delete attachments of %s", conversation_id, exc_info=True)
    await conversation_client.delete_conversation(conversation_id)


async def update_title(conversation_id: str, title: str, user_id: str):
    await _get_owned_conversation(conversation_id, user_id)
    await conversation_client.update_conversation_title(conversation_id, title)


async def get_research_tasks(graph, conversation_id: str, user_id: str) -> list[dict]:
    """The background research tasks started in a conversation, with live
    progress — what the chat's side panel polls. The task ids live in the
    conversation's own agent state (see app/agents/async_research.py); their
    progress is read from each task's own thread."""
    await _get_owned_conversation(conversation_id, user_id)
    if not research_runner.ready:
        return []
    state = await graph.aget_state({"configurable": {"thread_id": conversation_id}})
    return await describe_tasks((state.values or {}).get("research_tasks") or {})


async def get_context_breakdown(graph, conversation_id: str, user_id: str) -> dict:
    """What the conversation's context is made of, for the chat's context panel.

    `total` is the provider-reported size of the last model call (the same
    figure as the gauge). The system prompt, skills listing and tool schemas
    are measured by ContextBreakdownMiddleware on that call; messages are what
    is left, so the parts always add up to `total`. `deferred` is what the
    tools held back by ToolSearchMiddleware would have cost — not part of the
    total. A conversation with no model call since the breakdown was
    introduced has the total only."""
    conv = await _get_owned_conversation(conversation_id, user_id)
    total = conv.context_tokens or 0
    state = await graph.aget_state({"configurable": {"thread_id": conversation_id}})
    b = (state.values or {}).get("context_breakdown")
    if not b:
        return {"total": total, "system_prompt": None, "skills": None, "tools": None, "mcp_tools": None,
                "messages": None, "tool_detail": [], "deferred": {"tokens": 0, "detail": []}}
    system, skills = b.get("system", 0), b.get("skills", 0)
    tools, mcp = b.get("tools", 0), b.get("mcp_tools", 0)
    return {
        "total": total,
        "system_prompt": system,
        "skills": skills,
        "tools": tools,
        "mcp_tools": mcp,
        "messages": max(0, total - system - skills - tools - mcp),
        "tool_detail": b.get("tool_detail", []),
        "deferred": b.get("deferred") or {"tokens": 0, "detail": []},
    }


async def get_messages(graph, conversation_id: str, user_id: str) -> dict:
    """Composes two data sources owned by two different services: LangGraph
    checkpoint state (local — stays with the Chat/Agent service) and
    subagent_traces (remote — jarvis-conversation-service). See Chapter 7's
    API composition pattern, which this is a small preview of."""
    await _get_owned_conversation(conversation_id, user_id)
    config = {"configurable": {"thread_id": conversation_id}}
    state = await graph.aget_state(config)
    messages = state.values.get("messages", []) if state.values else []
    subagent_traces = await conversation_client.get_subagent_traces(conversation_id)
    message_traces = await conversation_client.get_message_traces(conversation_id)
    feedback_by_trace = await feedback_repository.get_feedback_bulk(list(message_traces.values()))
    return {
        "messages": serialize_messages(messages, subagent_traces, message_traces, feedback_by_trace),
        "is_pending": bool(state.next),
    }


# ── branching ─────────────────────────────────────────────────────────────────


def _is_user_turn_start(msg) -> bool:
    return isinstance(msg, HumanMessage) and not str(msg.content).startswith(SUMMARY_PREFIX)


def messages_for_branch(messages: list, keep_turns: int) -> list:
    """The messages of the first `keep_turns` turns — a turn being a message the user wrote and everything up to
    the next one. Raises HTTPException when that is not a point a conversation can be continued from: more turns
    than there are, or a cut that does not end on a finished answer (a tool call without its result, or an
    answer still being written, cannot be resumed in a new thread)."""
    if keep_turns < 0:
        raise HTTPException(status_code=400, detail="keep_turns must not be negative")
    starts = [i for i, m in enumerate(messages) if _is_user_turn_start(m)]
    if keep_turns > len(starts):
        raise HTTPException(status_code=400, detail="This conversation has fewer turns than that")
    kept = messages[: starts[keep_turns]] if keep_turns < len(starts) else list(messages)
    if not kept:
        return []
    last = kept[-1]
    if not isinstance(last, AIMessage) or last.tool_calls:
        raise HTTPException(status_code=400, detail="Cannot branch here: that turn has not finished")
    return kept


def _node_that_ends_the_run(graph) -> str | None:
    """A node whose edge leads to the end of the graph. Writing state `as_node` that one makes LangGraph treat the
    new thread as a run that has just finished (state.next empty). Without it a thread whose state was only ever
    written by hand reads as 'about to run the model', and the page would show a spurious interrupted banner."""
    for edge in graph.get_graph().edges:
        if edge.target == "__end__":
            return edge.source
    return None


async def fork_conversation(graph, conversation_id: str, user_id: str, keep_turns: int) -> dict:
    """A new conversation that starts as a copy of the first `keep_turns` turns of this one, to continue
    differently from there. The original is left as it is.

    Built from the messages in the original's latest checkpoint, so no checkpoint history is needed. What is not
    carried over: the sandbox (the branch gets its own, empty one — files the earlier turns made there are not in
    it) and the todo list / background research tasks, which belong to the original's present, not to the point
    branched from."""
    conv = await _get_owned_conversation(conversation_id, user_id)
    state = await graph.aget_state({"configurable": {"thread_id": conversation_id}})
    values = state.values or {}
    kept = messages_for_branch(values.get("messages", []), keep_turns)

    title = (conv.title or "New conversation")[:80] + " (branch)"
    new = await conversation_client.create_conversation(title, user_id)
    try:
        if kept:
            update: dict = {"messages": kept}
            if values.get("loaded_tools"):
                update["loaded_tools"] = values["loaded_tools"]
            await graph.aupdate_state(
                {"configurable": {"thread_id": new["id"]}}, update, as_node=_node_that_ends_the_run(graph)
            )
            tokens = count_tokens_approximately(kept)
            await conversation_client.set_context_tokens(new["id"], tokens)
            new["context_tokens"] = tokens
        if conv.last_model or conv.last_subagent_model:
            await conversation_client.touch_conversation(new["id"], conv.last_model, conv.last_subagent_model)
            new["last_model"], new["last_subagent_model"] = conv.last_model, conv.last_subagent_model
    except Exception:
        # Do not leave an empty conversation behind in the user's list (it is one this call just made).
        await conversation_client.delete_conversation(new["id"])
        raise
    return new
