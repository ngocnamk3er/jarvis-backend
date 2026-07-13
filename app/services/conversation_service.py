import shutil
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from app.core.config import settings
from app.db import repository
from app.agents.tools.sandbox_manager import stop_container

# ── serialization ─────────────────────────────────────────────────────────────


def _replay_subagent_trace(trace: list[dict], task_tool_call_id: str) -> list[dict]:
    """Reconstruct tool parts from a saved subagent trace (the raw tool_start/
    tool_end SSE payloads, in original stream order) so they render nested
    under the `task` badge after reload, same as they did live."""
    parts: list[dict] = []
    by_run_id: dict[str, dict] = {}
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
        prev_type = ev["type"]
    return parts


def serialize_messages(messages: list, subagent_traces: dict[str, list[dict]] | None = None) -> list[dict]:
    subagent_traces = subagent_traces or {}
    tool_outputs: dict[str, str] = {
        msg.tool_call_id: str(msg.content)
        for msg in messages
        if isinstance(msg, ToolMessage)
    }

    result: list[dict] = []
    pending_parts: list[dict] = []
    pending_usage_calls: list[dict] = []

    def flush():
        if pending_parts:
            entry: dict = {"role": "assistant", "parts": list(pending_parts)}
            if pending_usage_calls:
                entry["usage"] = list(pending_usage_calls)
            result.append(entry)
            pending_parts.clear()
        pending_usage_calls.clear()

    for msg in messages:
        if isinstance(msg, HumanMessage):
            flush()
            content = str(msg.content)
            if content.startswith("Here is a summary of the conversation to date:"):
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
            reasoning = msg.additional_kwargs.get("reasoning", "")
            if reasoning:
                pending_parts.append({"type": "thinking", "content": reasoning, "isStreaming": False})
            if msg.content:
                pending_parts.append({"type": "text", "content": str(msg.content)})
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
                raw_output = tool_outputs.get(tc["id"], "")
                try:
                    import json as _json
                    data = _json.loads(raw_output)
                    if "__viz__" in data:
                        pending_parts.append({"type": "viz", "format": data["__viz__"], "code": data["code"], "title": data.get("title", "")})
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
                pending_parts.append({"type": "tool", "tool": tool_part})

                trace = subagent_traces.get(tc["id"])
                if trace:
                    tool_part["run_id"] = tc["id"]
                    pending_parts.extend(_replay_subagent_trace(trace, tc["id"]))

    flush()
    return result


# ── CRUD ──────────────────────────────────────────────────────────────────────


async def list_conversations(pool):
    return await repository.list_conversations(pool)


async def create_conversation(pool, title: str):
    return await repository.create_conversation(pool, title)


async def delete_conversation(pool, graph, conversation_id: str) -> None:
    stop_container(conversation_id)

    sandbox_dir = Path(settings.SANDBOX_DATA_DIR) / conversation_id
    if sandbox_dir.exists():
        shutil.rmtree(sandbox_dir, ignore_errors=True)

    await repository.delete_conversation(pool, conversation_id)


async def update_title(pool, conversation_id: str, title: str):
    await repository.update_conversation_title(pool, conversation_id, title)


async def get_messages(graph, conversation_id: str, pool) -> dict:
    config = {"configurable": {"thread_id": conversation_id}}
    state = await graph.aget_state(config)
    messages = state.values.get("messages", []) if state.values else []
    subagent_traces = await repository.get_subagent_traces(pool, conversation_id)
    return {
        "messages": serialize_messages(messages, subagent_traces),
        "is_pending": bool(state.next),
    }
