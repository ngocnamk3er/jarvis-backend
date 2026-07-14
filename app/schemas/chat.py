from pydantic import BaseModel
from typing import Literal

# Pricing/context sourced from openrouter.ai model pages.
# Sorted by input price ascending — cheapest first, which is also the default.
AVAILABLE_MODELS = [
    {"id": "tencent/hy3-preview",         "name": "HY3 Preview",        "desc": "Lightweight, cheap",               "inputPrice": "$0.063", "outputPrice": "$0.21",  "context": "262K"},
    {"id": "deepseek/deepseek-v4-flash",  "name": "DeepSeek Flash",     "desc": "Fast, cost-effective MoE",        "inputPrice": "$0.09",  "outputPrice": "$0.18",  "context": "1M"},
    {"id": "openai/gpt-5.4-nano",         "name": "GPT-5.4 Nano",       "desc": "Speed-critical, high-volume",      "inputPrice": "$0.20",  "outputPrice": "$1.25",  "context": "400K"},
    {"id": "qwen/qwen3.7-plus",           "name": "Qwen 3.7+",          "desc": "High quality",                     "inputPrice": "$0.32",  "outputPrice": "$1.28",  "context": "1M"},
    {"id": "deepseek/deepseek-v4-pro",    "name": "DeepSeek Pro",       "desc": "Advanced reasoning, coding, agents", "inputPrice": "$0.435", "outputPrice": "$0.87",  "context": "1M"},
    {"id": "anthropic/claude-opus-4.8",   "name": "Claude Opus 4.8",    "desc": "Frontier reasoning",               "inputPrice": "$1.70",  "outputPrice": "$25.00", "context": "1M"},
]

DEFAULT_MODEL = AVAILABLE_MODELS[0]["id"]


class ChatRequest(BaseModel):
    thread_id: str
    content: str
    thinking_effort: Literal["low", "medium", "high", "xhigh"] = "high"
    model: str = DEFAULT_MODEL


class ResumeRequest(BaseModel):
    thread_id: str
    decision: Literal["approve", "reject"]
