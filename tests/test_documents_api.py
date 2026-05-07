from httpx import AsyncClient


async def test_upload_txt(client: AsyncClient, mock_embed):
    res = await client.post(
        "/documents",
        files={"file": ("test.txt", b"Hello world this is a test document", "text/plain")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["filename"] == "test.txt"
    assert "id" in data


async def test_upload_md(client: AsyncClient, mock_embed):
    res = await client.post(
        "/documents",
        files={"file": ("notes.md", b"# Notes\nSome markdown content here", "text/markdown")},
    )
    assert res.status_code == 200
    assert res.json()["filename"] == "notes.md"


async def test_upload_pdf(client: AsyncClient, mock_embed, mocker):
    mocker.patch("app.services.documents._parse_pdf", return_value="PDF text content")
    res = await client.post(
        "/documents",
        files={"file": ("doc.pdf", b"%PDF-1.4 fake content", "application/pdf")},
    )
    assert res.status_code == 200
    assert res.json()["filename"] == "doc.pdf"


async def test_list_documents(client: AsyncClient, mock_embed):
    await client.post("/documents", files={"file": ("a.txt", b"Content A", "text/plain")})
    await client.post("/documents", files={"file": ("b.txt", b"Content B", "text/plain")})
    res = await client.get("/documents")
    assert res.status_code == 200
    assert len(res.json()) >= 2


async def test_list_documents_pagination(client: AsyncClient, mock_embed):
    for i in range(5):
        await client.post("/documents", files={"file": (f"doc{i}.txt", f"Content {i}".encode(), "text/plain")})
    res = await client.get("/documents", params={"offset": 0, "limit": 2})
    assert res.status_code == 200
    assert len(res.json()) == 2


async def test_delete_document(client: AsyncClient, mock_embed):
    create = await client.post(
        "/documents",
        files={"file": ("to_delete.txt", b"Delete me", "text/plain")},
    )
    doc_id = create.json()["id"]
    res = await client.delete(f"/documents/{doc_id}")
    assert res.status_code == 204


async def test_delete_document_not_found(client: AsyncClient):
    res = await client.delete("/documents/nonexistent")
    assert res.status_code == 404


async def test_query_documents_no_data(client: AsyncClient, mock_embed):
    res = await client.post("/documents/query", json={"query": "What is Present Simple?"})
    assert res.status_code == 200
    data = res.json()
    assert data["answer"] == "No relevant documents found."
    assert data["sources"] == []


async def test_query_documents_with_data(client: AsyncClient, mock_embed, mock_get_client):
    await client.post(
        "/documents",
        files={"file": ("english.txt", b"Present Simple is used for habits and routines", "text/plain")},
    )
    res = await client.post("/documents/query", json={"query": "What is Present Simple?"})
    assert res.status_code == 200
    data = res.json()
    assert data["answer"] == "Test answer"
    assert len(data["sources"]) > 0
