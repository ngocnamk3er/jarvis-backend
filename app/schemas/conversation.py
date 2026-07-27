from datetime import datetime
from pydantic import BaseModel


class ConversationOut(BaseModel):
    id: str
    title: str
    last_model: str | None = None
    last_subagent_model: str | None = None
    created_at: datetime
    updated_at: datetime


class ConversationCreate(BaseModel):
    title: str = "New conversation"
