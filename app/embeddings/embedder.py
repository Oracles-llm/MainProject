"""
Embedding service for generating text embeddings.
Currently uses Gemini Embedding API, designed to be easily switchable to self-hosted models.
"""

from typing import List, Optional, Union
import os
from google import genai
from google.genai import types

from app.core.config import settings
from app.core.logging import get_logger
from app.embeddings.models import (
    EmbeddingTaskType,
    EmbeddingConfig,
    EmbeddingResult
)

logger = get_logger(__name__)


class Embedder:
    """Embedding service using Gemini API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "models/embedding-001",
        default_task_type: EmbeddingTaskType = EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        default_output_dimensionality: Optional[int] = None
    ):
        """
        Initialize the embedder.
        
        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model: Embedding model name (default: models/embedding-001, can use gemini-embedding-001)
            default_task_type: Default task type for embeddings
            default_output_dimensionality: Default output dimension (128-3072, recommended: 768)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY must be provided either as parameter or environment variable"
            )
        
        self.model = model
        self.default_task_type = default_task_type
        self.default_output_dimensionality = default_output_dimensionality
        
        genai.configure(api_key=self.api_key)
        self.client = genai.Client()
        
        logger.info(f"Embedder initialized with model: {model}")
    
    def embed(
        self,
        texts: Union[str, List[str]],
        task_type: Optional[EmbeddingTaskType] = None,
        output_dimensionality: Optional[int] = None,
        config: Optional[EmbeddingConfig] = None
    ) -> EmbeddingResult:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of text strings
            task_type: Task type for embedding (defaults to instance default)
            output_dimensionality: Output dimension (defaults to instance default)
            config: Optional EmbeddingConfig object (overrides other params)
        
        Returns:
            EmbeddingResult containing embeddings and metadata
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            raise ValueError("At least one text is required")
        
        if config:
            embed_config = config.to_embed_config()
            task_type = config.task_type
        else:
            task_type = task_type or self.default_task_type
            output_dimensionality = output_dimensionality or self.default_output_dimensionality
            
            config_dict = {"task_type": task_type.value}
            if output_dimensionality:
                config_dict["output_dimensionality"] = output_dimensionality
            embed_config = types.EmbedContentConfig(**config_dict)
        
        try:
            logger.debug(f"Generating embeddings for {len(texts)} text(s) with task_type: {task_type.value}")
            
            response = self.client.models.embed_content(
                model=self.model,
                contents=texts,
                config=embed_config
            )
            
            embeddings = [list(embedding.values) for embedding in response.embeddings]
            
            logger.debug(f"Generated embeddings with dimension: {len(embeddings[0]) if embeddings else 0}")
            
            return EmbeddingResult(
                embeddings=embeddings,
                model=self.model,
                task_type=task_type.value
            )
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def embed_documents(
        self,
        texts: Union[str, List[str]],
        output_dimensionality: Optional[int] = None
    ) -> EmbeddingResult:
        """
        Generate embeddings optimized for documents (RETRIEVAL_DOCUMENT).
        
        Args:
            texts: Single text string or list of text strings
            output_dimensionality: Output dimension (optional)
        
        Returns:
            EmbeddingResult containing embeddings
        """
        return self.embed(
            texts=texts,
            task_type=EmbeddingTaskType.RETRIEVAL_DOCUMENT,
            output_dimensionality=output_dimensionality
        )
    
    def embed_query(
        self,
        text: str,
        output_dimensionality: Optional[int] = None
    ) -> List[float]:
        """
        Generate embedding optimized for search queries (RETRIEVAL_QUERY).
        
        Args:
            text: Query text string
            output_dimensionality: Output dimension (optional)
        
        Returns:
            Single embedding vector as list of floats
        """
        result = self.embed(
            texts=text,
            task_type=EmbeddingTaskType.RETRIEVAL_QUERY,
            output_dimensionality=output_dimensionality
        )
        return result.embeddings[0] if result.embeddings else []
    
    def embed_batch(
        self,
        texts: List[str],
        task_type: Optional[EmbeddingTaskType] = None,
        output_dimensionality: Optional[int] = None,
        batch_size: int = 100
    ) -> EmbeddingResult:
        """
        Generate embeddings for a large batch of texts.
        
        Args:
            texts: List of text strings
            task_type: Task type for embedding
            output_dimensionality: Output dimension
            batch_size: Number of texts to process per batch
        
        Returns:
            EmbeddingResult containing all embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.debug(f"Processing batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
            
            result = self.embed(
                texts=batch,
                task_type=task_type,
                output_dimensionality=output_dimensionality
            )
            all_embeddings.extend(result.embeddings)
        
        task_type = task_type or self.default_task_type
        return EmbeddingResult(
            embeddings=all_embeddings,
            model=self.model,
            task_type=task_type.value
        )


def get_embedder() -> Embedder:
    """
    Get a default embedder instance using settings.
    
    Returns:
        Configured Embedder instance
    """
    return Embedder(
        api_key=settings.GEMINI_API_KEY,
        model=settings.EMBEDDING_MODEL
    )

