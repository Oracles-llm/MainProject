"""
Embedding service for generating text embeddings.
"""

from app.embeddings.embedder import Embedder, get_embedder
from app.embeddings.models import (
    EmbeddingTaskType,
    EmbeddingConfig,
    EmbeddingResult
)

__all__ = [
    "Embedder",
    "get_embedder",
    "EmbeddingTaskType",
    "EmbeddingConfig",
    "EmbeddingResult"
]

