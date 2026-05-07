from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db.database import Base
from app.main import app
from app.routers.chat import get_db as chat_get_db
from app.routers.documents import get_db as doc_get_db

test_engine = create_async_engine(
    "sqlite+aiosqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
test_session_maker = async_sessionmaker(test_engine, expire_on_commit=False)


async def _override_get_db():
    async with test_session_maker() as session:
        yield session


@pytest_asyncio.fixture(autouse=True)
async def prepare_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession]:
    async with test_session_maker() as session:
        yield session


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient]:
    app.dependency_overrides[chat_get_db] = _override_get_db
    app.dependency_overrides[doc_get_db] = _override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


EMBED_RETURN = [0.1] * 768


@pytest.fixture
def mock_stream_chat(mocker):
    async def _fake_stream(messages, model):
        yield "Hello"
        yield " from"
        yield " test"

    mocker.patch("app.services.chat.stream_chat", side_effect=_fake_stream)


@pytest.fixture
def mock_embed(mocker):
    async def _fake_embed(text):
        return EMBED_RETURN

    async def _fake_batch(texts):
        return [EMBED_RETURN for _ in texts]

    mocker.patch("app.services.rag.embed_text", side_effect=_fake_embed)
    mocker.patch("app.services.embeddings.embed_text", side_effect=_fake_embed)
    mocker.patch("app.services.documents.embed_batch", side_effect=_fake_batch)
    mocker.patch("app.services.embeddings.embed_batch", side_effect=_fake_batch)


@pytest.fixture
def mock_get_client(mocker):
    fake_client = mocker.AsyncMock()
    fake_client.chat.completions.create.return_value = mocker.Mock(
        choices=[mocker.Mock(message=mocker.Mock(content="Test answer"))]
    )
    fake_client.models.list.return_value = mocker.Mock(
        data=[mocker.Mock(id="accounts/fireworks/models/test-model")]
    )
    mocker.patch("app.services.llm.get_client", return_value=fake_client)
    mocker.patch("app.routers.documents.get_client", return_value=fake_client)
    mocker.patch("app.routers.chat.get_client", return_value=fake_client)
    return fake_client
