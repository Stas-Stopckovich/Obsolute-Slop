import os
from collections.abc import AsyncGenerator

from openai import AsyncOpenAI


def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("LLM_API_KEY", ""),
    )


async def stream_chat(messages: list[dict], model: str) -> AsyncGenerator[str]:
    client = get_client()
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    async for chunk in response:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content
