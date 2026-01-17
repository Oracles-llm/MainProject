"""
Utility functions for initializing sparse vector generators.
"""

from typing import Optional
from app.retrieval.sparse_vectors import SparseVectorGenerator
from app.core.logging import get_logger

logger = get_logger(__name__)


def get_sparse_vector_generator(model_name: str = "Qdrant/bm25") -> Optional[SparseVectorGenerator]:
    """
    Get a sparse vector generator instance using fastembed.
    
    Args:
        model_name: Name of the sparse embedding model (default: "Qdrant/bm25")
    
    Returns:
        SparseVectorGenerator instance, or None if initialization fails
    """
    try:
        logger.info(f"Initializing sparse vector generator with model: {model_name}")
        sparse_gen = SparseVectorGenerator(model_name=model_name)
        logger.info("Sparse vector generator initialized successfully")
        return sparse_gen
    except Exception as e:
        logger.error(f"Failed to initialize sparse vector generator: {e}")
        return None


def load_sparse_vector_generator_from_qdrant(
    model_name: str = "Qdrant/bm25"
) -> Optional[SparseVectorGenerator]:
    """
    Get a sparse vector generator (no need to load from Qdrant with fastembed).
    
    Args:
        model_name: Name of the sparse embedding model (default: "Qdrant/bm25")
    
    Returns:
        SparseVectorGenerator instance, or None if initialization fails
    """
    return get_sparse_vector_generator(model_name)

