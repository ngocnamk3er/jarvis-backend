from pydantic import BaseModel
from typing import Literal

AVAILABLE_MODELS = [
    {"id": "qwen/qwen3.7-plus",           "name": "Qwen 3.7+"},
    {"id": "deepseek/deepseek-v4-flash",  "name": "DeepSeek Flash"},
    {"id": "deepseek/deepseek-v4-pro",    "name": "DeepSeek Pro"},
    {"id": "openai/gpt-5.4-nano",         "name": "GPT-5 Nano"},
    {"id": "tencent/hy3-preview",         "name": "HY3 Preview"},
]

DEFAULT_MODEL = "deepseek/deepseek-v4-flash"


class ChatRequest(BaseModel):
    thread_id: str
    content: str
    thinking_effort: Literal["low", "medium", "high", "xhigh"] = "high"
    model: str = DEFAULT_MODEL


class ResumeRequest(BaseModel):
    thread_id: str
    decision: Literal["approve", "reject"]
