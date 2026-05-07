from datetime import datetime

from pydantic import BaseModel


class ChatCreate(BaseModel):
    title: str | None = None


class ChatResponse(BaseModel):
    id: str
    title: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class MessageCreate(BaseModel):
    content: str
    model: str


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    model: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatDetailResponse(ChatResponse):
    messages: list[MessageResponse]