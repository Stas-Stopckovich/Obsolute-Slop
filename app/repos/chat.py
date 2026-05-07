from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Chat, Message


class ChatRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, title: str | None) -> Chat:
        chat = Chat(title=title)
        self.session.add(chat)
        await self.session.commit()
        await self.session.refresh(chat)
        return chat

    async def get(self, chat_id: str) -> Chat | None:
        result = await self.session.execute(
            select(Chat).where(Chat.id == chat_id).options(selectinload(Chat.messages))
        )
        return result.scalar_one_or_none()

    async def list(self, offset: int = 0, limit: int = 20) -> list[Chat]:
        result = await self.session.execute(
            select(Chat).offset(offset).limit(limit).order_by(Chat.created_at.desc())
        )
        return result.scalars().all()

    async def delete(self, chat_id: str) -> bool:
        chat = await self.get(chat_id)
        if not chat:
            return False
        await self.session.execute(delete(Chat).where(Chat.id == chat_id))
        await self.session.commit()
        return True

    async def save_message(self, chat_id: str, role: str, content: str, model: str | None = None) -> Message:
        msg = Message(chat_id=chat_id, role=role, content=content, model=model)
        self.session.add(msg)
        await self.session.commit()
        await self.session.refresh(msg)
        return msg

    async def get_history(self, chat_id: str) -> list[Message]:
        result = await self.session.execute(
            select(Message).where(Message.chat_id == chat_id).order_by(Message.created_at)
        )
        return result.scalars().all()
