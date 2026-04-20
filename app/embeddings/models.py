"""
Embedding models and configuration.
Defines embedding task types and configuration structures.
"""

from enum import Enum
from typing import Optional, List
from dataclasses import dataclass


class EmbeddingTaskType(str, Enum):
    """Embedding task types used by the embedding layer."""
    
    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
    CLASSIFICATION = "CLASSIFICATION"
    CLUSTERING = "CLUSTERING"
    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    
    task_type: EmbeddingTaskType = EmbeddingTaskType.RETRIEVAL_DOCUMENT
    output_dimensionality: Optional[int] = None
    
    def to_embed_config(self):
        """Convert to Gemini EmbedContentConfig."""
        from google.genai import types
        
        config_dict = {
            "task_type": self.task_type.value
        }
        
        if self.output_dimensionality:
            config_dict["output_dimensionality"] = self.output_dimensionality
            
        return types.EmbedContentConfig(**config_dict)


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""
    
    embeddings: List[List[float]]
    model: str
    task_type: str
    
    @property
    def dimension(self) -> int:
        """Get the dimension of embeddings."""
        if self.embeddings and self.embeddings[0]:
            return len(self.embeddings[0])
        return 0

