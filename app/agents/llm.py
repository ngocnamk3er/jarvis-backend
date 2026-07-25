from langchain_openai import ChatOpenAI
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.messages import AIMessageChunk
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

from app.core.config import settings


def enable_llm_cache(path: str = ".langchain_cache.db") -> None:
    """Enable SQLite-backed LLM response cache. Same prompt → same response, no API call."""
    set_llm_cache(SQLiteCache(database_path=path))


# Tried via each model's own .with_fallbacks() chain (see build_llm_with_fallback
# below) if the primary errors (rate-limited, down, etc.).
_FALLBACK_MODELS = [m.strip() for m in settings.OPENROUTER_FALLBACK_MODELS.split(",") if m.strip()]


class ThinkingChatOpenAI(ChatOpenAI):
    """ChatOpenAI with two additions:
    1. Reads `thinking_effort` and `model` from LangGraph's configurable at
       request time so the caller can steer reasoning depth and model per-turn.
    2. Rescues OpenRouter's `reasoning` delta field that LangChain drops,
       storing it in additional_kwargs so chat_service emits thinking_token events.
    """

    # False on fallback-chain instances (build_llm_with_fallback) so they keep
    # their own fixed model instead of also picking up the ambient per-turn
    # override — get_config() is read the same regardless of which instance
    # in the chain is executing, so without this every fallback would just
    # retry the same (already-failing) user-selected model instead of
    # actually diversifying.
    honor_model_override: bool = True

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        try:
            from langgraph.config import get_config
            cfg = get_config().get("configurable", {})
        except RuntimeError:
            cfg = {}

        effort = cfg.get("thinking_effort", "high")
        if "extra_body" not in kwargs:
            kwargs["extra_body"] = {"reasoning": {"effort": effort, "exclude": False}}

        model_override = cfg.get("model")
        if model_override and self.honor_model_override and "model" not in kwargs:
            kwargs["model"] = model_override

        return super()._get_request_payload(input_, stop=stop, **kwargs)

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        gen_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if gen_chunk is None:
            return None

        choices = chunk.get("choices") or []
        if choices and isinstance(gen_chunk.message, AIMessageChunk):
            delta = choices[0].get("delta") or {}
            reasoning = delta.get("reasoning")
            if reasoning:
                gen_chunk.message.additional_kwargs["reasoning"] = reasoning

        return gen_chunk


def build_llm(model: str | None = None, *, honor_model_override: bool = True) -> ThinkingChatOpenAI:
    return ThinkingChatOpenAI(
        model=model or settings.OPENROUTER_MODEL,
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
        streaming=True,
        temperature=0,
        # OpenRouter only reports token usage on the final stream chunk when asked;
        # ChatOpenAI auto-enables this for api.openai.com but not custom base URLs.
        stream_usage=True,
        honor_model_override=honor_model_override,
    )


def build_llm_with_fallback(model: str | None = None):
    """build_llm(), plus a LangChain .with_fallbacks() chain: if the primary
    model errors (rate-limited, down, etc.), retries on the next model in
    OPENROUTER_FALLBACK_MODELS, in order, within the same call.

    Used for both the root agent and every subagent — RESEARCH_SUBAGENT is
    wired as a CompiledSubAgent (see subagents.py) specifically so it can
    take a .with_fallbacks() Runnable here instead of a plain model; passing
    one through the raw SubAgent "model" field breaks, since deepagents'
    resolve_model() unconditionally treats non-BaseChatModel objects as
    provider-prefixed strings (verified: raises AttributeError: 'count').
    """
    primary_id = model or settings.OPENROUTER_MODEL
    primary = build_llm(model)
    fallbacks = [
        build_llm(m, honor_model_override=False) for m in _FALLBACK_MODELS if m != primary_id
    ]
    return primary.with_fallbacks(fallbacks) if fallbacks else primary
