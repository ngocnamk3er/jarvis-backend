from langchain_openai import ChatOpenAI
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.messages import AIMessageChunk

from app.core.config import settings


class ThinkingChatOpenAI(ChatOpenAI):
    """ChatOpenAI that rescues OpenRouter's `reasoning` delta field.

    LangChain's _convert_delta_to_message_chunk ignores unknown delta keys,
    so `reasoning` is dropped before we can read it. We override
    _convert_chunk_to_generation_chunk to fish it out of the raw dict
    (after model_dump()) and store it in additional_kwargs so chat_service
    can emit it as a thinking_token SSE event.
    """

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
