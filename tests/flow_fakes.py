"""A scripted model and fake tools for the flow tests. The compiler, the runner and the API around them are the real ones."""

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.config import get_config


class EchoModel(BaseChatModel):
    """Answers `[<model>] <the prompt>`, or what `replies` says for a prompt containing a key; streams it in two halves.
    Records each call: the model the run's configurable named, and the messages."""

    replies: dict[str, str] = {}
    calls: list = []

    @property
    def _llm_type(self):
        return "echo"

    def _answer(self, messages) -> str:
        cfg = get_config().get("configurable", {})
        prompt = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
        system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
        self.calls.append({"model": cfg.get("model"), "prompt": prompt, "system": system,
                           "no_fallback": cfg.get("no_fallback")})
        for key, reply in self.replies.items():
            if key in prompt:
                return reply
        return f"[{cfg.get('model')}] {prompt}"

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(self._answer(messages)))])

    async def _astream(self, messages, stop=None, run_manager=None, **kw):
        text = self._answer(messages)
        for part in (text[: len(text) // 2], text[len(text) // 2:]):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=part))
            if run_manager:
                await run_manager.on_llm_new_token(part, chunk=chunk)
            yield chunk


tool_calls: list = []


@tool
async def web_search(query: str, label: str, config: RunnableConfig) -> str:
    """Search the web."""
    cfg = config.get("configurable", {})
    tool_calls.append({"tool": "web_search", "query": query, "label": label, "user_id": cfg.get("user_id"),
                       "sandbox": cfg.get("sandbox")})
    return f"results for {query}"


@tool
async def top_words(text: str, top_k: int, config: RunnableConfig, lower: bool = False) -> str:
    """The most frequent words of a text."""
    tool_calls.append({"tool": "top_words", "text": text, "top_k": top_k, "lower": lower})
    words = text.lower().split() if lower else text.split()
    return json.dumps(sorted(set(words))[:top_k])


@tool
async def broken(query: str, config: RunnableConfig) -> str:
    """Always fails."""
    raise RuntimeError("upstream is down")


TOOLS = {t.name: t for t in (web_search, top_words, broken)}


def flow(nodes: list[dict], edges: list[tuple[str, str]]) -> dict:
    return {"schema_version": 1, "nodes": nodes, "edges": [{"source": s, "target": t} for s, t in edges]}


def node(id: str, type: str, **config) -> dict:
    return {"id": id, "type": type, "config": config}


def news_flow() -> dict:
    """input -> search -> summarise -> output"""
    return flow(
        [node("in", "input", fields=[{"name": "topic"}]),
         node("s", "tool", tool="web_search", args={"query": "{{in.topic}} news"}),
         node("sum", "llm", prompt="Summarise: {{s.result}}"),
         node("out", "output", template="Summary: {{sum}}")],
        [("in", "s"), ("s", "sum"), ("sum", "out")],
    )
