from pydantic import BaseModel
from typing import Literal

# Pricing/context sourced from openrouter.ai model pages.
# Sorted by input price ascending — cheapest first.
#
# Removed after testing against the same multi-step research prompt:
# - tencent/hy3-preview: repeatedly announced an action ("Let me fetch...")
#   without ever issuing the tool call, leaving the turn silently incomplete.
# - openai/gpt-5.4-nano: ignored the "don't call web_search more than twice
#   on the same topic" prompt guidance entirely — delegated to a research
#   subagent AND kept searching itself, 46 tool calls for one question,
#   exhausting the shared Tavily search quota for everyone.
AVAILABLE_MODELS = [
    {"id": "deepseek/deepseek-v4-flash",  "name": "DeepSeek Flash",     "desc": "Fast, cost-effective MoE",        "inputPrice": "$0.09",  "outputPrice": "$0.18",  "context": "1M",  "size": "13B active / 284B total", "default": True},
    {"id": "qwen/qwen3.5-9b",             "name": "Qwen 3.5 9B",        "desc": "Compact, efficient multimodal",   "inputPrice": "$0.10",  "outputPrice": "$0.15",  "context": "262K", "size": "9B"},
    {"id": "qwen/qwen3.7-plus",           "name": "Qwen 3.7+",          "desc": "High quality",                     "inputPrice": "$0.32",  "outputPrice": "$1.28",  "context": "1M",  "size": "Undisclosed"},
    {"id": "deepseek/deepseek-v4-pro",    "name": "DeepSeek Pro",       "desc": "Advanced reasoning, coding, agents", "inputPrice": "$0.435", "outputPrice": "$0.87",  "context": "1M", "size": "49B active / 1.6T total"},
    {"id": "anthropic/claude-opus-4.8",   "name": "Claude Opus 4.8",    "desc": "Frontier reasoning",               "inputPrice": "$1.70",  "outputPrice": "$25.00", "context": "1M", "size": "Undisclosed"},
]

DEFAULT_MODEL = next(m["id"] for m in AVAILABLE_MODELS if m.get("default"))


class ChatRequest(BaseModel):
    thread_id: str
    content: str
    thinking_effort: Literal["low", "medium", "high", "xhigh"] = "high"
    model: str = DEFAULT_MODEL


class ResumeRequest(BaseModel):
    thread_id: str
    decision: Literal["approve", "reject"]
    # Without this, resume used to silently fall back to DEFAULT_MODEL for
    # the rest of the turn — confirmed live: a conversation selected as
    # qwen3.5-9b, once a HITL approval triggered /chat/resume, kept running
    # (and wrote the final synthesized answer) on deepseek-v4-flash instead.
    model: str = DEFAULT_MODEL
