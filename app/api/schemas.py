"""
API request and response schemas.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Chat request model."""
    query: str = Field(..., description="User query/question")


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str = Field(..., description="Generated response")

