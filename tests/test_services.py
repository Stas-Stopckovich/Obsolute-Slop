from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk
from app.repos.vector import VectorRepo, _bytes_to_embedding, _embedding_to_bytes
from app.services.documents import _chunk_text
from app.services.rag import RagService


def test_chunk_text_basic():
    text = "word " * 600
    chunks = _chunk_text(text, size=100, overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) <= 100


def test_chunk_text_small_text():
    chunks = _chunk_text("hello world", size=500, overlap=100)
    assert len(chunks) == 1
    assert chunks[0] == "hello world"


def test_chunk_text_empty():
    chunks = _chunk_text("")
    assert chunks == []


def test_embedding_roundtrip():
    original = [0.1, -0.2, 0.3, 0.0, 1.0]
    packed = _embedding_to_bytes(original)
    unpacked = _bytes_to_embedding(packed)
    assert len(unpacked) == len(original)
    for a, b in zip(original, unpacked, strict=True):
        assert abs(a - b) < 1e-6


async def test_vector_repo_search(db_session: AsyncSession):
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    repo = VectorRepo(db_session)

    chunk1 = Chunk(document_id="doc1", content="first chunk", embedding=_embedding_to_bytes(vec1), chunk_index=0)
    chunk2 = Chunk(document_id="doc1", content="second chunk", embedding=_embedding_to_bytes(vec2), chunk_index=1)
    db_session.add_all([chunk1, chunk2])
    await db_session.commit()

    results = await repo.search([1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0]["content"] == "first chunk"
    assert results[0]["score"] > results[1]["score"]


async def test_vector_repo_search_empty(db_session: AsyncSession):
    repo = VectorRepo(db_session)
    results = await repo.search([0.1] * 768, top_k=5)
    assert results == []


async def test_vector_repo_store_chunks(db_session: AsyncSession):
    repo = VectorRepo(db_session)
    chunks_data = [
        {"content": "chunk zero", "embedding": [0.1] * 768, "index": 0},
        {"content": "chunk one", "embedding": [0.2] * 768, "index": 1},
    ]
    await repo.store_chunks("doc1", chunks_data)

    results = await repo.search([0.15] * 768, top_k=2)
    assert len(results) == 2


async def test_vector_repo_delete_by_document(db_session: AsyncSession):
    repo = VectorRepo(db_session)
    chunk = Chunk(document_id="doc1", content="to delete", embedding=_embedding_to_bytes([0.1] * 768), chunk_index=0)
    db_session.add(chunk)
    await db_session.commit()

    await repo.delete_by_document("doc1")
    results = await repo.search([0.1] * 768, top_k=5)
    assert results == []


async def test_rag_get_enriched_system_prompt(db_session: AsyncSession, mocker):
    vec_repo = VectorRepo(db_session)
    chunk = Chunk(
        document_id="doc1",
        content="Present Simple is used for habits",
        embedding=_embedding_to_bytes([1.0, 0.0, 0.0]),
        chunk_index=0,
    )
    db_session.add(chunk)
    await db_session.commit()

    mocker.patch("app.services.rag.embed_text", return_value=[1.0, 0.0, 0.0])
    rag = RagService(vec_repo)
    result = await rag.get_enriched_system_prompt("Base prompt", "What is Present Simple?")
    assert "Base prompt" in result
    assert "Present Simple" in result


async def test_rag_get_enriched_system_prompt_no_context(db_session: AsyncSession, mocker):
    vec_repo = VectorRepo(db_session)
    mocker.patch("app.services.rag.embed_text", return_value=[0.0] * 768)
    rag = RagService(vec_repo)
    result = await rag.get_enriched_system_prompt("Base prompt", "irrelevant query")
    assert result == "Base prompt"
