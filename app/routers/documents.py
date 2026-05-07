from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import async_session_maker
from app.repos.document import DocumentRepo
from app.repos.vector import VectorRepo
from app.schemas.document import DocumentQuery, DocumentQueryResponse, DocumentResponse
from app.services.documents import DocumentsService
from app.services.llm import get_client
from app.services.rag import RagService

router = APIRouter(prefix="/documents", tags=["documents"])


async def get_db():
    async with async_session_maker() as session:
        yield session


def _doc_service(db: AsyncSession = Depends(get_db)) -> DocumentsService:
    return DocumentsService(DocumentRepo(db), VectorRepo(db))


@router.post("", response_model=DocumentResponse)
async def upload_document(file: UploadFile, db: AsyncSession = Depends(get_db)):
    content = await file.read()
    svc = DocumentsService(DocumentRepo(db), VectorRepo(db))
    result = await svc.upload(file.filename or "untitled.txt", content)
    doc = await DocumentRepo(db).get(result["id"])
    if not doc:
        raise HTTPException(status_code=500, detail="Document creation failed")
    return doc


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, le=100),
    db: AsyncSession = Depends(get_db),
):
    return await DocumentRepo(db).list(offset, limit)


@router.delete("/{doc_id}", status_code=204)
async def delete_document(doc_id: str, db: AsyncSession = Depends(get_db)):
    svc = DocumentsService(DocumentRepo(db), VectorRepo(db))
    deleted = await svc.delete(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")


@router.post("/query", response_model=DocumentQueryResponse)
async def query_documents(body: DocumentQuery, db: AsyncSession = Depends(get_db)):
    rag = RagService(VectorRepo(db))
    context = await rag.build_context(body.query, body.top_k)

    if not context:
        return DocumentQueryResponse(answer="No relevant documents found.", sources=[])

    system_prompt = f"Answer the question using the provided context.\n\nContext:\n{context}"

    client = get_client()
    response = await client.chat.completions.create(
        model="accounts/fireworks/models/glm-5p1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": body.query},
        ],
    )

    results = await rag.search(body.query, body.top_k)
    sources = list({r["document_id"] for r in results})

    return DocumentQueryResponse(
        answer=response.choices[0].message.content,
        sources=sources,
    )
