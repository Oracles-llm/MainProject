"""
Embedding service for generating text embeddings.
Supports both custom interface and LangChain compatibility.
"""

from app.embeddings.embedder import (
    Embedder,
    get_embedder,
    LangChainGeminiEmbeddings,
    get_langchain_embeddings
)
from app.embeddings.models import (
    EmbeddingTaskType,
    EmbeddingConfig,
    EmbeddingResult
)

__all__ = [
    "Embedder",
    "get_embedder",
    "LangChainGeminiEmbeddings",
    "get_langchain_embeddings",
    "EmbeddingTaskType",
    "EmbeddingConfig",
    "EmbeddingResult"
]

