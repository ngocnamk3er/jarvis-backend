"""Shared helpers for offloading long tool outputs out of the model's context.

`ToolOutputOffloadMiddleware` (middleware.py) rewrites the model-facing copy of
a long ToolMessage into a short stub and stashes the full text in the LangGraph
store; the `recall_tool_output` tool (tools/recall_tool_output.py) reads it back.
Both sides agree on the store namespace and the ref format here.
"""

from __future__ import annotations

import hashlib

from langchain_core.messages import ToolMessage

# Store namespace: (PREFIX, thread_id). Per-conversation — a ref only means
# something within the conversation that produced it.
_NS_PREFIX = "tool_offload"

# Marker line the stub starts with, so the model (and our own re-scan) can tell
# a stub apart from a real tool result.
STUB_MARKER = "[tool-output-offloaded]"


def store_namespace(thread_id: str) -> tuple[str, str]:
    return (_NS_PREFIX, thread_id or "default")


def ref_for(tool_call_id: str) -> str:
    """Short, stable id for a tool result — derived from its tool_call_id so a
    graph replay re-derives the same ref instead of duplicating."""
    return "to_" + hashlib.sha1((tool_call_id or "").encode()).hexdigest()[:10]


def message_text(content) -> str:
    """ToolMessage.content is usually a str; tolerate the content-blocks list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def is_stub(message: ToolMessage) -> bool:
    return message_text(message.content).startswith(STUB_MARKER)


def render_stub(*, name: str, ref: str, text: str, head: int, tail: int) -> str:
    """The compact placeholder the model sees instead of the full output."""
    n = len(text)
    approx_tokens = n // 4
    body = text if n <= head + tail else (
        text[:head].rstrip()
        + f"\n... [{n - head - tail:,} chars omitted] ...\n"
        + text[-tail:].lstrip()
    )
    return (
        f"{STUB_MARKER} tool={name!r}  size={n:,} chars (~{approx_tokens:,} tokens)\n"
        f"--- preview ---\n{body}\n--- end preview ---\n"
        f"The full output was moved out of context to keep it small. "
        f'If you need all of it, call recall_tool_output("{ref}").'
    )
