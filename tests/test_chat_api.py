from httpx import AsyncClient


async def test_health(client: AsyncClient):
    res = await client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


async def test_create_chat(client: AsyncClient):
    res = await client.post("/chats", json={"title": "Test chat"})
    assert res.status_code == 200
    data = res.json()
    assert data["title"] == "Test chat"
    assert "id" in data
    assert "created_at" in data


async def test_create_chat_no_title(client: AsyncClient):
    res = await client.post("/chats", json={})
    assert res.status_code == 200
    assert res.json()["title"] is None


async def test_list_chats(client: AsyncClient):
    await client.post("/chats", json={"title": "Chat 1"})
    await client.post("/chats", json={"title": "Chat 2"})
    res = await client.get("/chats")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 2


async def test_list_chats_pagination(client: AsyncClient):
    for i in range(5):
        await client.post("/chats", json={"title": f"Chat {i}"})
    res = await client.get("/chats", params={"offset": 0, "limit": 2})
    assert res.status_code == 200
    assert len(res.json()) == 2


async def test_get_chat(client: AsyncClient):
    create = await client.post("/chats", json={"title": "My chat"})
    chat_id = create.json()["id"]
    res = await client.get(f"/chats/{chat_id}")
    assert res.status_code == 200
    assert res.json()["title"] == "My chat"
    assert res.json()["messages"] == []


async def test_get_chat_not_found(client: AsyncClient):
    res = await client.get("/chats/nonexistent")
    assert res.status_code == 404


async def test_delete_chat(client: AsyncClient):
    create = await client.post("/chats", json={"title": "To delete"})
    chat_id = create.json()["id"]
    res = await client.delete(f"/chats/{chat_id}")
    assert res.status_code == 204
    res = await client.get(f"/chats/{chat_id}")
    assert res.status_code == 404


async def test_delete_chat_not_found(client: AsyncClient):
    res = await client.delete("/chats/nonexistent")
    assert res.status_code == 404


async def test_send_message(client: AsyncClient, mock_stream_chat, mock_embed):
    create = await client.post("/chats", json={"title": "Chat"})
    chat_id = create.json()["id"]

    res = await client.post(
        f"/chats/{chat_id}/messages",
        json={"content": "Hello", "model": "test-model"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["role"] == "assistant"
    assert data["content"] == "Hello from test"


async def test_send_message_chat_not_found(client: AsyncClient, mock_stream_chat, mock_embed):
    res = await client.post(
        "/chats/nonexistent/messages",
        json={"content": "Hello", "model": "test-model"},
    )
    assert res.status_code == 404


async def test_send_message_stream(client: AsyncClient, mock_stream_chat, mock_embed):
    create = await client.post("/chats", json={"title": "Chat"})
    chat_id = create.json()["id"]

    res = await client.post(
        f"/chats/{chat_id}/messages/stream",
        json={"content": "Hello", "model": "test-model"},
    )
    assert res.status_code == 200
    assert "text/event-stream" in res.headers.get("content-type", "")


async def test_get_chat_with_messages(client: AsyncClient, mock_stream_chat, mock_embed):
    create = await client.post("/chats", json={"title": "Chat"})
    chat_id = create.json()["id"]

    await client.post(
        f"/chats/{chat_id}/messages",
        json={"content": "Hello", "model": "test-model"},
    )

    res = await client.get(f"/chats/{chat_id}")
    assert res.status_code == 200
    messages = res.json()["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"
    assert messages[1]["role"] == "assistant"


async def test_list_models(client: AsyncClient, mock_get_client):
    res = await client.get("/chats/models")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 1
    assert data[0]["id"] == "accounts/fireworks/models/test-model"
