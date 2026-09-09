"""`recall_tool_output` — bring an offloaded tool result back into context.

When a tool result was long and is a few turns old, ToolOutputOffloadMiddleware
replaces it in the model's view with a stub that ends
`... call recall_tool_output("to_xxxxxxxxxx")`. This tool reads the full text
back out of the LangGraph store for that ref.
"""

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.agents.tool_offload import store_namespace


@tool
async def recall_tool_output(ref: str, config: RunnableConfig) -> str:
    """Retrieve the full text of an earlier tool output that was moved out of
    context to keep it small.

    Use the ref shown in a `[tool-output-offloaded] ...` stub (looks like
    `to_1a2b3c4d5e`). Only call this when the preview in the stub isn't enough
    and you actually need the whole output — it brings the full text back into
    context for this turn.

    Args:
        ref: The `to_...` id from the offloaded-output stub.
    """
    try:
        from langgraph.config import get_store

        store = get_store()
    except (RuntimeError, ValueError):
        store = None
    if store is None:
        return "Error: no store available to recall from."

    thread_id = config.get("configurable", {}).get("thread_id", "default")
    ref = ref.strip().strip('"').strip("'")

    item = await store.aget(store_namespace(thread_id), ref)
    if item is None:
        return (
            f"Error: no offloaded output found for ref {ref!r}. It may be from a "
            "different conversation, or the ref is mistyped — copy it exactly "
            "from the '[tool-output-offloaded]' stub."
        )

    value = item.value or {}
    text = value.get("content", "")
    tool_name = value.get("tool", "tool")
    return f"Full output of {tool_name!r} (ref {ref}):\n\n{text}"
