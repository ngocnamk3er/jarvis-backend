"""Trimming oversized tool results before they are condensed into a summary.

A tool result that is mostly bulk (a page of web text, a long listing) adds
little to a summary but can make the summarizing request itself too big. This
keeps the start and the end of each over-long text result and drops the middle,
with a marker saying how much. Only the copy sent to the summarizer is trimmed;
nothing in the saved conversation changes.

The 8,192 / 4,096 / 1,024 character figures are the defaults of DeepSeek
Harness's compaction-tool-result-pruner, kept so the two behave alike.
"""

from langchain_core.messages import BaseMessage, ToolMessage

THRESHOLD_CHARS = 8_192
HEAD_CHARS = 4_096
TAIL_CHARS = 1_024


def _trim(text: str) -> str:
    if len(text) <= THRESHOLD_CHARS:
        return text
    cut = len(text) - HEAD_CHARS - TAIL_CHARS
    return f"{text[:HEAD_CHARS]}\n\n[... {cut:,} characters pruned from the middle ...]\n\n{text[-TAIL_CHARS:]}"


def prune_tool_results(messages: list[BaseMessage]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            c = m.content
            if isinstance(c, str):
                trimmed = _trim(c)
            elif isinstance(c, list):
                trimmed = [{**b, "text": _trim(b["text"])} if isinstance(b, dict) and b.get("type") == "text"
                           and isinstance(b.get("text"), str) else b for b in c]
            else:
                trimmed = c
            if trimmed != c:
                m = m.model_copy(update={"content": trimmed})
        out.append(m)
    return out
