from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.db.database import async_session_maker
from app.schemas.chat import (
    ChatCreate,
    ChatDetailResponse,
    ChatResponse,
    MessageCreate,
    MessageResponse,
)
from app.services.chat import ChatService
from app.services.llm import get_client

router = APIRouter(prefix="/chats", tags=["chats"])


async def get_db():
    async with async_session_maker() as session:
        yield session


def _service(db: AsyncSession = Depends(get_db)) -> ChatService:
    return ChatService(db)


@router.get("/models", tags=["models"])
async def list_models():
    client = get_client()
    models = await client.models.list()
    return [{"id": m.id} for m in models.data if "accounts/fireworks/models" in m.id]


@router.post("", response_model=ChatResponse)
async def create_chat(body: ChatCreate, svc: ChatService = Depends(_service)):
    chat = await svc.create_chat(body.title)
    return chat


@router.get("", response_model=list[ChatResponse])
async def list_chats(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, le=100),
    svc: ChatService = Depends(_service),
):
    return await svc.list_chats(offset, limit)


@router.get("/{chat_id}", response_model=ChatDetailResponse)
async def get_chat(chat_id: str, svc: ChatService = Depends(_service)):
    chat = await svc.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.delete("/{chat_id}", status_code=204)
async def delete_chat(chat_id: str, svc: ChatService = Depends(_service)):
    deleted = await svc.delete_chat(chat_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat not found")


@router.post("/{chat_id}/messages", response_model=MessageResponse)
async def send_message(chat_id: str, body: MessageCreate, svc: ChatService = Depends(_service)):
    result = await svc.send_message(chat_id, body)
    if result is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return result


@router.post("/{chat_id}/messages/stream")
async def send_message_stream(chat_id: str, body: MessageCreate, svc: ChatService = Depends(_service)):
    chat = await svc.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return EventSourceResponse(svc.stream_message(chat_id, body))
