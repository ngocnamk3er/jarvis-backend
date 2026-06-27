from pydantic import BaseModel
from typing import Literal


class ChatRequest(BaseModel):
    thread_id: str
    content: str
    thinking_effort: Literal["low", "medium", "high", "xhigh"] = "high"
