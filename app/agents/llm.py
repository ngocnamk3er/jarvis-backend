from langchain_openai import ChatOpenAI
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.messages import AIMessageChunk
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
import hashlib

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

    # True for the research subagent's own model (see subagents.py). Reads
    # "subagent_model" from configurable instead of "model" — LangGraph's
    # ensure_config merges the root run's configurable into the subagent's
    # own config (verified live), so without a distinct key the subagent
    # would always just inherit whatever model the main agent is using,
    # with no way to pick a different one for it specifically. Falls back to
    # "model" when "subagent_model" isn't set, so subagents still inherit
    # the main model by default until the user explicitly overrides it.
    is_subagent_model: bool = False

    # Whether to send OpenRouter's `reasoning` body field. An own endpoint (OpenAI, vLLM...) may reject a body field it
    # does not know, so it is off there unless the user asked for it.
    send_reasoning: bool = True

    # True for the instance that talks to a user's own endpoint (see _delegate_for): it uses its own model and
    # endpoint, never the per-turn override, and is not a fallback.
    delegate: bool = False

    def _configurable(self) -> dict:
        """LangGraph's per-run configurable, or {} outside a run."""
        try:
            from langgraph.config import get_config
            return get_config().get("configurable", {})
        except RuntimeError:
            return {}

    def _model_override(self) -> str | None:
        """The model this turn should actually use, or None to keep self.model."""
        if not self.honor_model_override:
            return None
        cfg = self._configurable()
        if self.is_subagent_model:
            return cfg.get("subagent_model") or cfg.get("model")
        return cfg.get("model")

    def _get_invocation_params(self, stop=None, **kwargs):
        """Report the model the request will actually use, not the one this
        instance happens to be constructed with.

        LangChain builds these params and hands them to every callback
        *before* _get_request_payload() below swaps in the per-turn model, and
        _get_ls_params() derives ls_model_name from them in turn. Without this
        override a tracer records OPENROUTER_MODEL for every single turn no
        matter which model the user picked — and, worse, prices the call
        against that model. Verified against Langfuse: a run served by
        qwen3.5-9b was recorded as deepseek-r1-0528-qwen3-8b:free.

        Fallback instances need nothing here: each carries its own model and
        has honor_model_override False, so _model_override() returns None and
        the name they report is already the one that answered.
        """
        params = super()._get_invocation_params(stop=stop, **kwargs)
        override = self._model_override()
        if override and "model" not in kwargs:
            params["model"] = override
        return params

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        effort = self._configurable().get("thinking_effort", "high")
        if "extra_body" not in kwargs and self.send_reasoning:
            kwargs["extra_body"] = {"reasoning": {"effort": effort, "exclude": False}}

        model_override = self._model_override()
        if model_override and "model" not in kwargs:
            kwargs["model"] = model_override

        return super()._get_request_payload(input_, stop=stop, **kwargs)

    # ── a user's own model ──────────────────────────────────────────────────
    #
    # When the turn's model is `user:<uuid>`, the call is handed to another instance built for that model's name,
    # endpoint and key. The key is looked up here, at call time, not carried in the run's config.

    async def _custom_for_turn(self):
        """The user's model this call is for, or None. Raises if the chosen model is gone or unreadable."""
        from app.agents import user_models

        override = self._model_override()
        if not user_models.is_user_model(override):
            return None
        cfg = self._configurable()
        custom = await user_models.resolve(cfg.get("user_id") or "", override)
        if custom is None:
            raise user_models.UserModelError("The model you picked was removed. Choose another one.")
        return custom

    def _refuse_fallback(self) -> None:
        """A fallback model must not answer a turn the user put on their own endpoint."""
        if not self.honor_model_override and not self.delegate and self._configurable().get("no_fallback"):
            raise RuntimeError("Your own endpoint failed; the server's fallback models are not used for it.")

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        self._refuse_fallback()
        custom = await self._custom_for_turn()
        if custom is None:
            async for chunk in super()._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
                yield chunk
            return
        async for chunk in _delegate_for(custom)._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
            yield chunk

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self._refuse_fallback()
        custom = await self._custom_for_turn()
        if custom is None:
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        return await _delegate_for(custom)._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

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


_delegates: dict[tuple, "ThinkingChatOpenAI"] = {}


def _delegate_for(custom) -> "ThinkingChatOpenAI":
    """The instance that calls a user's model; kept so its HTTP clients are reused between turns."""
    key = (custom.base_url, hashlib.sha256(custom.api_key.encode()).hexdigest(), custom.model, custom.send_reasoning)
    llm = _delegates.get(key)
    if llm is None:
        llm = ThinkingChatOpenAI(
            model=custom.model,
            api_key=custom.api_key or "none",          # some servers (Ollama) want no key, the client wants a string
            base_url=custom.base_url,
            streaming=True,
            temperature=0,
            stream_usage=True,
            honor_model_override=False,
            delegate=True,
            send_reasoning=custom.send_reasoning,
        )
        if len(_delegates) > 64:
            _delegates.clear()
        _delegates[key] = llm
    return llm


def build_llm(
    model: str | None = None, *, honor_model_override: bool = True, is_subagent_model: bool = False
) -> ThinkingChatOpenAI:
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
        is_subagent_model=is_subagent_model,
    )


def build_llm_with_fallback(model: str | None = None, *, is_subagent_model: bool = False):
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
    primary = build_llm(model, is_subagent_model=is_subagent_model)
    fallbacks = [
        build_llm(m, honor_model_override=False, is_subagent_model=is_subagent_model)
        for m in _FALLBACK_MODELS
        if m != primary_id
    ]
    return primary.with_fallbacks(fallbacks) if fallbacks else primary
