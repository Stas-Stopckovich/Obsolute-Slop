from app.repos.vector import VectorRepo
from app.services.embeddings import embed_text

RAG_SYSTEM_SUFFIX = "\n\nRelevant context from uploaded documents:\n{context}"


class RagService:
    def __init__(self, vec_repo: VectorRepo):
        self.vec_repo = vec_repo

    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_embedding = await embed_text(query)
        return await self.vec_repo.search(query_embedding, top_k)

    async def build_context(self, query: str, top_k: int = 5) -> str:
        results = await self.search(query, top_k)
        if not results:
            return ""
        return "\n---\n".join(r["content"] for r in results)

    async def get_enriched_system_prompt(self, base_prompt: str, query: str) -> str:
        context = await self.build_context(query)
        if not context:
            return base_prompt
        return base_prompt + RAG_SYSTEM_SUFFIX.format(context=context)
