from pydantic import BaseModel
from typing import List

class DocumentRequest(BaseModel):
    id: int
    text: str
    category: str

class DocumentBatchRequest(BaseModel):
    documents: List[DocumentRequest]

class StructuredRecommendRequest(BaseModel):
    query: str
    top_k: int = 5

    genre: str | None = None
    exclude_genre: str | None = None

    year_from: int | None = None
    year_to: int | None = None

    min_vote_average: float | None = None
    min_vote_count: int | None = None

class AgentRecommendRequest(BaseModel):
    message: str

class AgentUserRecommendRequest(BaseModel):
    user_id: int
    message: str

class LocalAgentUserRecommendRequest(BaseModel):
    user_id: int
    prompt: str




