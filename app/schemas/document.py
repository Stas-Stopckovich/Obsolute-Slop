from datetime import datetime

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: str
    filename: str
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentQuery(BaseModel):
    query: str
    top_k: int = 5


class DocumentQueryResponse(BaseModel):
    answer: str
    sources: list[str]
