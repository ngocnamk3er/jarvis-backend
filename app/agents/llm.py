from langchain_openai import ChatOpenAI
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.messages import AIMessageChunk

from app.core.config import settings


class ThinkingChatOpenAI(ChatOpenAI):
    """ChatOpenAI with two additions:
    1. Reads `thinking_effort` and `model` from LangGraph's configurable at
       request time so the caller can steer reasoning depth and model per-turn.
    2. Rescues OpenRouter's `reasoning` delta field that LangChain drops,
       storing it in additional_kwargs so chat_service emits thinking_token events.
    """

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
        if model_override and "model" not in kwargs:
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


def build_llm(model: str | None = None) -> ThinkingChatOpenAI:
    return ThinkingChatOpenAI(
        model=model or settings.OPENROUTER_MODEL,
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
        streaming=True,
    )
