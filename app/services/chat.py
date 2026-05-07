import json
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import async_session_maker
from app.repos.chat import ChatRepo
from app.repos.vector import VectorRepo
from app.schemas.chat import MessageCreate
from app.services.llm import stream_chat
from app.services.rag import RagService

SYSTEM_PROMPT = ""


class ChatService:
    def __init__(self, session: AsyncSession):
        self.repo = ChatRepo(session)
        self.rag = RagService(VectorRepo(session))
        self.session = session

    async def create_chat(self, title: str | None):
        return await self.repo.create(title)

    async def get_chat(self, chat_id: str):
        return await self.repo.get(chat_id)

    async def list_chats(self, offset: int = 0, limit: int = 20):
        return await self.repo.list(offset, limit)

    async def delete_chat(self, chat_id: str) -> bool:
        return await self.repo.delete(chat_id)

    async def send_message(self, chat_id: str, body: MessageCreate):
        chat = await self.repo.get(chat_id)
        if not chat:
            return None

        await self.repo.save_message(chat_id, "user", body.content)

        history = await self.repo.get_history(chat_id)
        system_prompt = await self.rag.get_enriched_system_prompt(SYSTEM_PROMPT, body.content)
        messages_payload = self._build_messages(history, system_prompt)

        full_response = ""
        async for chunk in stream_chat(messages_payload, body.model):
            full_response += chunk

        assistant_msg = await self.repo.save_message(chat_id, "assistant", full_response, body.model)
        return assistant_msg

    async def stream_message(self, chat_id: str, body: MessageCreate) -> AsyncGenerator:
        chat = await self.repo.get(chat_id)
        if not chat:
            return

        await self.repo.save_message(chat_id, "user", body.content)

        history = await self.repo.get_history(chat_id)
        system_prompt = await self.rag.get_enriched_system_prompt(SYSTEM_PROMPT, body.content)
        messages_payload = self._build_messages(history, system_prompt)

        full_response = ""
        try:
            async for chunk in stream_chat(messages_payload, body.model):
                full_response += chunk
                yield {"data": json.dumps({"content": chunk}, ensure_ascii=False)}
        finally:
            async with async_session_maker() as save_session:
                repo = ChatRepo(save_session)
                await repo.save_message(chat_id, "assistant", full_response, body.model)
                yield {"data": json.dumps({"done": True}, ensure_ascii=False)}

    def _build_messages(self, history: list, system_prompt: str = SYSTEM_PROMPT) -> list[dict]:
        messages = [{"role": "system", "content": system_prompt}]
        for m in history:
            messages.append({"role": m.role, "content": m.content})
        return messages
