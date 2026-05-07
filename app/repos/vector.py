import struct

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk


def _embedding_to_bytes(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def _bytes_to_embedding(data: bytes) -> list[float]:
    count = len(data) // 4
    return list(struct.unpack(f"{count}f", data))


class VectorRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def store_chunks(self, document_id: str, chunks: list[dict]) -> None:
        for chunk_data in chunks:
            chunk = Chunk(
                document_id=document_id,
                content=chunk_data["content"],
                embedding=_embedding_to_bytes(chunk_data["embedding"]) if chunk_data.get("embedding") else None,
                chunk_index=chunk_data["index"],
            )
            self.session.add(chunk)
        await self.session.commit()

    async def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        result = await self.session.execute(select(Chunk).where(Chunk.embedding.isnot(None)))
        all_chunks = result.scalars().all()

        if not all_chunks:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        scored = []
        for chunk in all_chunks:
            chunk_vec = np.array(_bytes_to_embedding(chunk.embedding), dtype=np.float32)
            chunk_norm = np.linalg.norm(chunk_vec)
            if chunk_norm == 0:
                continue
            similarity = float(np.dot(query_vec, chunk_vec) / (query_norm * chunk_norm))
            scored.append({"content": chunk.content, "document_id": chunk.document_id, "score": similarity})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    async def delete_by_document(self, document_id: str) -> None:
        await self.session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        await self.session.commit()
