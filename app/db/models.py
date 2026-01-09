"""
Data models for Qdrant vector database.
Defines structures for documents, embeddings, and metadata.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Document:
    """Document model for storing text and metadata."""
    
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    chunk_index: Optional[int] = None
    created_at: Optional[datetime] = None
    
    def to_payload(self) -> Dict[str, Any]:
        """Convert document to Qdrant payload format."""
        payload = {
            "text": self.text,
            "source": self.source,
            "chunk_index": self.chunk_index,
            **self.metadata
        }
        if self.created_at:
            payload["created_at"] = self.created_at.isoformat()
        return payload


@dataclass
class VectorPoint:
    """Vector point model for Qdrant."""
    
    id: str
    vector: List[float]
    payload: Dict[str, Any]
    
    def to_point_struct(self):
        """Convert to Qdrant PointStruct."""
        from qdrant_client.models import PointStruct
        return PointStruct(
            id=self.id,
            vector=self.vector,
            payload=self.payload
        )


@dataclass
class SearchResult:
    """Search result model."""
    
    id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_qdrant_result(cls, result: Dict[str, Any]) -> "SearchResult":
        """Create SearchResult from Qdrant search result."""
        payload = result.get("payload", {})
        return cls(
            id=str(result["id"]),
            score=result["score"],
            text=payload.get("text", ""),
            metadata={k: v for k, v in payload.items() if k != "text"}
        )

