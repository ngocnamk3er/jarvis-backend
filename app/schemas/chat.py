from pydantic import BaseModel


class ChatRequest(BaseModel):
    thread_id: str
    content: str
