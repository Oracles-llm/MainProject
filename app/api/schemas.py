"""
API request and response schemas.
"""

from typing import Literal

from pydantic import BaseModel, Field


ChatMode = Literal["normal", "thinking"]


class ChatRequest(BaseModel):
    """Chat request model."""
    query: str = Field(..., description="User query/question")
    mode: ChatMode = Field("normal", description="Answering mode: normal or thinking")


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str = Field(..., description="Generated response")

