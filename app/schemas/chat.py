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
    {"id": "deepseek/deepseek-v4-flash",  "name": "DeepSeek Flash",     "desc": "Fast, cost-effective MoE",        "inputPrice": "$0.09",  "outputPrice": "$0.18",  "context": "1M",  "default": True},
    {"id": "qwen/qwen3.7-plus",           "name": "Qwen 3.7+",          "desc": "High quality",                     "inputPrice": "$0.32",  "outputPrice": "$1.28",  "context": "1M"},
    {"id": "deepseek/deepseek-v4-pro",    "name": "DeepSeek Pro",       "desc": "Advanced reasoning, coding, agents", "inputPrice": "$0.435", "outputPrice": "$0.87",  "context": "1M"},
    {"id": "anthropic/claude-opus-4.8",   "name": "Claude Opus 4.8",    "desc": "Frontier reasoning",               "inputPrice": "$1.70",  "outputPrice": "$25.00", "context": "1M"},
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
