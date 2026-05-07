import io

import pdfplumber

from app.repos.document import DocumentRepo
from app.repos.vector import VectorRepo
from app.services.embeddings import embed_batch

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def _parse_pdf(content: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def _parse_txt(content: bytes) -> str:
    return content.decode("utf-8", errors="replace")


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        start += size - overlap
    return chunks


class DocumentsService:
    def __init__(self, doc_repo: DocumentRepo, vec_repo: VectorRepo):
        self.doc_repo = doc_repo
        self.vec_repo = vec_repo

    async def upload(self, filename: str, content: bytes) -> dict:
        if filename.endswith(".pdf"):
            text = _parse_pdf(content)
        else:
            text = _parse_txt(content)

        doc = await self.doc_repo.create(filename, text)

        chunks = _chunk_text(text)
        if not chunks:
            return {"id": doc.id, "filename": doc.filename, "chunks": 0}

        embeddings = await embed_batch(chunks)

        chunk_data = [
            {"content": chunk, "embedding": emb, "index": i}
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings, strict=True))
        ]
        await self.vec_repo.store_chunks(doc.id, chunk_data)

        return {"id": doc.id, "filename": doc.filename, "chunks": len(chunks)}

    async def list_documents(self, offset: int = 0, limit: int = 20):
        return await self.doc_repo.list(offset, limit)

    async def delete(self, doc_id: str) -> bool:
        await self.vec_repo.delete_by_document(doc_id)
        return await self.doc_repo.delete(doc_id)
