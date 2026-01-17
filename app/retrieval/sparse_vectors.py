"""
Sparse vector generation for BM25/keyword search in Qdrant using fastembed.
"""

from typing import Dict, List, Optional
from app.core.logging import get_logger

logger = get_logger(__name__)


class SparseVectorGenerator:
    """Generate sparse vectors for BM25-based keyword search using fastembed."""
    
    def __init__(self, model_name: str = "Qdrant/bm25"):
        """
        Initialize sparse vector generator using fastembed.
        
        Args:
            model_name: Name of the sparse embedding model (default: "Qdrant/bm25")
        """
        try:
            from fastembed import SparseTextEmbedding
            self._model = SparseTextEmbedding(model_name)
            self._initialized = True
            logger.info(f"SparseVectorGenerator initialized with model: {model_name}")
        except ImportError:
            logger.error("fastembed not installed. Install with: pip install fastembed")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize SparseTextEmbedding: {e}")
            raise
    
    def _extract_sparse_dict(self, embedding) -> Dict[int, float]:
        """
        Extract sparse vector dictionary from embedding object.
        Handles both array-based structure (indices/values) and dictionary structure.
        
        Args:
            embedding: Sparse embedding object from fastembed
        
        Returns:
            Dictionary mapping token_id -> BM25 score
        """
        try:
            if hasattr(embedding, 'indices') and hasattr(embedding, 'values'):
                indices = embedding.indices
                values = embedding.values
                if isinstance(indices, (list, tuple)) and isinstance(values, (list, tuple)):
                    return {int(idx): float(val) for idx, val in zip(indices, values)}
        except Exception:
            pass
        
        obj = embedding.as_object()
        
        if isinstance(obj, dict):
            if 'indices' in obj and 'values' in obj:
                try:
                    import numpy as np
                    indices = obj['indices']
                    values = obj['values']
                    
                    if isinstance(indices, np.ndarray) and isinstance(values, np.ndarray):
                        return {int(idx): float(val) for idx, val in zip(indices, values)}
                    elif isinstance(indices, (list, tuple)) and isinstance(values, (list, tuple)):
                        return {int(idx): float(val) for idx, val in zip(indices, values)}
                except Exception as e:
                    logger.debug(f"Error processing indices/values arrays: {e}")
            
            numeric_dict = {}
            for k, v in obj.items():
                if k in ('indices', 'values'):
                    continue
                try:
                    if isinstance(k, int):
                        numeric_dict[k] = float(v)
                    elif isinstance(k, str) and k.isdigit():
                        numeric_dict[int(k)] = float(v)
                except (ValueError, TypeError):
                    continue
            
            if numeric_dict:
                return numeric_dict
        
        return {}
    
    def generate_sparse_vector(self, text: str) -> Dict[int, float]:
        """
        Generate BM25 sparse vector for a document.
        
        Args:
            text: Document text
        
        Returns:
            Dictionary mapping token_id -> BM25 score
        """
        if not self._initialized:
            raise ValueError("SparseVectorGenerator not initialized")
        
        try:
            sparse_embedding = next(self._model.embed([text]))
            return self._extract_sparse_dict(sparse_embedding)
        except Exception as e:
            logger.error(f"Failed to generate sparse vector: {e}")
            raise
    
    def generate_query_sparse_vector(self, query: str) -> Dict[int, float]:
        """
        Generate sparse vector for a query.
        
        Args:
            query: Query text
        
        Returns:
            Dictionary mapping token_id -> BM25 score
        """
        return self.generate_sparse_vector(query)
    
    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        """
        Generate sparse vectors for multiple documents.
        
        Args:
            texts: List of document texts
        
        Returns:
            List of sparse vectors as dictionaries
        """
        if not self._initialized:
            raise ValueError("SparseVectorGenerator not initialized")
        
        try:
            sparse_embeddings = list(self._model.embed(texts))
            return [
                self._extract_sparse_dict(embedding)
                for embedding in sparse_embeddings
            ]
        except Exception as e:
            logger.error(f"Failed to generate sparse vectors: {e}")
            raise
    
    def is_fitted(self) -> bool:
        """Check if the generator is initialized (always True for fastembed)."""
        return self._initialized

