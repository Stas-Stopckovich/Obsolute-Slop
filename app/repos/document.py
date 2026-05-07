from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document


class DocumentRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, filename: str, content: str) -> Document:
        doc = Document(filename=filename, content=content)
        self.session.add(doc)
        await self.session.commit()
        await self.session.refresh(doc)
        return doc

    async def get(self, doc_id: str) -> Document | None:
        result = await self.session.execute(select(Document).where(Document.id == doc_id))
        return result.scalar_one_or_none()

    async def list(self, offset: int = 0, limit: int = 20) -> list[Document]:
        result = await self.session.execute(
            select(Document).offset(offset).limit(limit).order_by(Document.created_at.desc())
        )
        return result.scalars().all()

    async def delete(self, doc_id: str) -> bool:
        doc = await self.get(doc_id)
        if not doc:
            return False
        await self.session.execute(delete(Document).where(Document.id == doc_id))
        await self.session.commit()
        return True
