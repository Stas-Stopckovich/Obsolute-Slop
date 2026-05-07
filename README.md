# Obsolute Slop

RAG-powered chat assistant with document upload. Upload PDF/TXT/MD files, ask questions — the assistant finds relevant context and streams responses in real-time.

## Stack

- **FastAPI** — async REST API with SSE streaming
- **SQLite** (aiosqlite) — database + vector storage (BLOB embeddings)
- **Fireworks AI** — LLM completions + embeddings via OpenAI-compatible API
- **nomic-embed-text-v1.5** — 768-dimensional text embeddings
- **Nginx** — reverse proxy with SSE support
- **Vanilla JS SPA** — single-file frontend, no build step

## Quick Start

```bash
# Clone
git clone https://github.com/nordbro/obsolute-slop.git
cd obsolute-slop

# Install (requires uv)
uv sync

# Configure
cp .env.example .env
# Edit .env: set LLM_API_KEY

# Run
uv run uvicorn app.main:app --reload
```

Open http://localhost:8000

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/chats` | Create chat |
| GET | `/chats` | List chats (paginated) |
| GET | `/chats/{id}` | Get chat with messages |
| DELETE | `/chats/{id}` | Delete chat |
| POST | `/chats/{id}/messages` | Send message (sync) |
| POST | `/chats/{id}/messages/stream` | Send message (SSE stream) |
| GET | `/chats/models` | List available LLM models |
| POST | `/documents` | Upload document (PDF/TXT/MD) |
| GET | `/documents` | List documents (paginated) |
| DELETE | `/documents/{id}` | Delete document + chunks |
| POST | `/documents/query` | RAG Q&A over documents |

## Architecture

```
Browser ──► Nginx (:80) ──► FastAPI (:8000) ──► SQLite (data.db)
                                     │
                                     ├── Repos (SQL queries)
                                     ├── Services (business logic)
                                     └── Fireworks API (LLM + embeddings)
```

**RAG Pipeline:**
1. Upload document → parse (PDF/TXT) → chunk (500 words, 100 overlap)
2. Embed chunks via Fireworks → store as BLOB in SQLite
3. User asks question → embed query → cosine similarity search → top-k chunks
4. Inject context into system prompt → stream LLM response via SSE

## Nginx Setup

```bash
sudo cp nginx/obsolute-slop.conf /etc/nginx/conf.d/
# Remove default server block from /etc/nginx/nginx.conf
sudo nginx -t && sudo systemctl reload nginx
```

## Development

```bash
uv sync --group dev

# Lint
uv run ruff check .

# Test (33 tests, 91% coverage)
uv run pytest tests/ -v --cov=app
```

CI runs on push/PR to main via GitHub Actions.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_BASE_URL` | OpenAI-compatible API base URL | `https://api.openai.com/v1` |
| `LLM_API_KEY` | API key for LLM + embeddings | (required) |

## License

MIT
