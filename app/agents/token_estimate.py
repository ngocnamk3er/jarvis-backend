"""The ~4-characters-per-token estimate used for prompt and tool-schema sizes
— the same approximation count_tokens_approximately uses for the new-message
estimate in chat_service.py."""


def tokens(text: str) -> int:
    return -(-len(text) // 4)


# Per-request scratch space in `request.state`: middleware further out record
# what they added or held back (skills listing, deferred tools) so
# ContextBreakdownMiddleware, which is innermost, can report it separately.
EXTRA_KEY = "_context_extra"
