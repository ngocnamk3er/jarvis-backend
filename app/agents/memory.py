"""Per-user persistent memory: a StoreBackend namespaced by user_id, shared
between MemoryMiddleware (reads/injects into the system prompt) and
FilesystemMiddleware (gives the model write_file/edit_file to update it) in
graph.py."""

from langgraph.config import get_config
from deepagents.backends.store import StoreBackend


def _user_memory_namespace(_runtime) -> tuple[str, str]:
    # Read from configurable rather than `_runtime.context` — same pattern as
    # ThinkingChatOpenAI._get_request_payload in llm.py — so chat_service.py
    # only has to thread user_id through _make_config's configurable dict,
    # not through a separate context_schema.
    user_id = get_config().get("configurable", {}).get("user_id") or "anonymous"
    return (user_id, "memory")


memory_backend = StoreBackend(namespace=_user_memory_namespace)
