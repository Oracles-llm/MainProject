"""
Embedding service for generating text embeddings.
Supports Gemini API and local llama.cpp embedding models.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union
import os

try:
    from google import genai
    from google.genai import types
    GEMINI_EMBEDDINGS_AVAILABLE = True
except ImportError:
    genai = None
    types = None
    GEMINI_EMBEDDINGS_AVAILABLE = False

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    Llama = None
    LLAMA_CPP_AVAILABLE = False

try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    TextEmbedding = None
    FASTEMBED_AVAILABLE = False

try:
    from langchain_core.embeddings import Embeddings as LangChainEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    LangChainEmbeddings = object

from app.core.config import settings
from app.core.logging import get_logger
from app.embeddings.models import (
    EmbeddingTaskType,
    EmbeddingConfig,
    EmbeddingResult
)

logger = get_logger(__name__)


class Embedder:
    """Embedding service using either Gemini API or local llama.cpp models."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "models/embedding-001",
        provider: Optional[str] = None,
        model_path: Optional[str] = None,
        n_ctx: Optional[int] = None,
        n_threads: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        verbose: Optional[bool] = None,
        cache_dir: Optional[str] = None,
        use_cuda: Optional[bool] = None,
        default_task_type: EmbeddingTaskType = EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        default_output_dimensionality: Optional[int] = None
    ):
        """
        Initialize the embedder.
        
        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model: Embedding model name
            provider: Embedding provider ("gemini", "fastembed", or "local")
            model_path: Local GGUF path for llama.cpp embedding models
            n_ctx: Context window for local embedding model
            n_threads: CPU threads for local embedding model
            n_gpu_layers: GPU offload layers for local embedding model
            verbose: Enable verbose local model logging
            cache_dir: Optional fastembed cache directory
            use_cuda: Whether fastembed should use CUDA providers
            default_task_type: Default task type for embeddings
            default_output_dimensionality: Default output dimension (128-3072, recommended: 768)
        """
        self.provider = (provider or settings.EMBEDDING_PROVIDER).lower()
        self.model = model
        self.model_path = model_path or settings.EMBEDDING_MODEL_PATH
        self.n_ctx = n_ctx if n_ctx is not None else settings.EMBEDDING_N_CTX
        self.n_threads = n_threads if n_threads is not None else settings.EMBEDDING_N_THREADS
        self.n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else settings.EMBEDDING_N_GPU_LAYERS
        self.verbose = verbose if verbose is not None else settings.EMBEDDING_VERBOSE
        self.cache_dir = cache_dir or settings.EMBEDDING_CACHE_DIR
        self.use_cuda = use_cuda if use_cuda is not None else settings.EMBEDDING_USE_CUDA
        self.default_task_type = default_task_type
        self.default_output_dimensionality = default_output_dimensionality

        if self.provider == "gemini":
            self.api_key = api_key or os.getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY
            self._init_gemini()
        elif self.provider == "fastembed":
            self.api_key = None
            self._init_fastembed()
        elif self.provider == "local":
            self.api_key = None
            self._init_local()
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

        logger.info(
            "Embedder initialized with provider=%s model=%s model_path=%s",
            self.provider,
            self.model,
            self.model_path,
        )

    def _init_gemini(self) -> None:
        """Initialize Gemini embedding client."""
        if not GEMINI_EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "google-genai is not installed. Install with: pip install google-genai"
            )
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY must be provided either as parameter or environment variable"
            )
        self.client = genai.Client(api_key=self.api_key)
        self.llama = None

    def _init_local(self) -> None:
        """Initialize local llama.cpp embedding client."""
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is not installed. Install with: pip install llama-cpp-python"
            )
        if not self.model_path:
            raise ValueError(
                "EMBEDDING_MODEL_PATH must be set when EMBEDDING_PROVIDER=local"
            )
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Embedding model file not found: {model_path}")

        self.llama = Llama(
            model_path=str(model_path),
            embedding=True,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose,
        )
        self.client = None
        self.fastembed = None

    def _init_fastembed(self) -> None:
        """Initialize local fastembed dense embedding client."""
        if not FASTEMBED_AVAILABLE:
            raise ImportError(
                "fastembed is not installed. Install with: pip install fastembed"
            )

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.use_cuda else None
        self.fastembed = TextEmbedding(
            model_name=self.model,
            cache_dir=self.cache_dir,
            threads=self.n_threads,
            providers=providers,
            cuda=self.use_cuda,
            lazy_load=False,
        )
        self.client = None
        self.llama = None

    def _embed_local_texts(
        self,
        texts: List[str],
        output_dimensionality: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings with the local llama.cpp model.

        Uses one text at a time to avoid decode failures on multi-input batches
        that some GGUF embedding models exhibit.
        """
        embeddings: List[List[float]] = []

        for text in texts:
            raw_embedding = self.llama.embed(text, normalize=False, truncate=True)

            # llama_cpp may return either a single vector or a list containing one vector.
            if raw_embedding and isinstance(raw_embedding[0], (list, tuple)):
                vector = list(raw_embedding[0])
            else:
                vector = list(raw_embedding)

            if output_dimensionality:
                vector = vector[:output_dimensionality]

            embeddings.append(vector)

        return embeddings
    
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

        output_dimensionality = output_dimensionality or self.default_output_dimensionality
        
        if config:
            task_type = config.task_type
            embed_config = config.to_embed_config() if self.provider == "gemini" else None
        else:
            task_type = task_type or self.default_task_type
            embed_config = None
            if self.provider == "gemini":
                config_dict = {"task_type": task_type.value}
                if output_dimensionality:
                    config_dict["output_dimensionality"] = output_dimensionality
                embed_config = types.EmbedContentConfig(**config_dict)
        
        try:
            logger.debug(
                "Generating embeddings for %d text(s) with provider=%s task_type=%s",
                len(texts),
                self.provider,
                task_type.value,
            )

            if self.provider == "gemini":
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=texts,
                    config=embed_config
                )
                embeddings = [list(embedding.values) for embedding in response.embeddings]
            elif self.provider == "fastembed":
                if task_type == EmbeddingTaskType.RETRIEVAL_QUERY:
                    raw_embeddings = list(self.fastembed.query_embed(texts))
                else:
                    raw_embeddings = list(self.fastembed.passage_embed(texts))

                embeddings = [list(embedding.tolist()) for embedding in raw_embeddings]
                if output_dimensionality:
                    embeddings = [embedding[:output_dimensionality] for embedding in embeddings]
            else:
                embeddings = self._embed_local_texts(
                    texts=texts,
                    output_dimensionality=output_dimensionality,
                )
            
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


class LangChainGeminiEmbeddings(LangChainEmbeddings):
    """LangChain-compatible embedding wrapper for project embedder."""
    
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        task_type_for_documents: EmbeddingTaskType = EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        task_type_for_queries: EmbeddingTaskType = EmbeddingTaskType.RETRIEVAL_QUERY
    ):
        """
        Initialize LangChain-compatible embeddings.
        
        Args:
            embedder: Optional Embedder instance (creates new one if not provided)
            model: Model name (used if creating new embedder)
            api_key: API key (used if creating new embedder)
            task_type_for_documents: Task type for document embeddings
            task_type_for_queries: Task type for query embeddings
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Install with: pip install langchain-core"
            )
        
        super().__init__()
        self._embedder = embedder or Embedder(
            api_key=api_key or settings.GEMINI_API_KEY,
            model=model or settings.EMBEDDING_MODEL
        )
        self.task_type_for_documents = task_type_for_documents
        self.task_type_for_queries = task_type_for_queries
        
        logger.info("LangChain project embeddings initialized")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed search documents (LangChain interface).
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        # Reuse the embedder batching helper for provider-specific batching behavior.
        result = self._embedder.embed_batch(
            texts=texts,
            task_type=self.task_type_for_documents,
            batch_size=100,
        )
        return result.embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text (LangChain interface).
        
        Args:
            text: Query text string
        
        Returns:
            Embedding vector
        """
        result = self._embedder.embed(
            texts=text,
            task_type=self.task_type_for_queries
        )
        return result.embeddings[0] if result.embeddings else []
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_documents."""
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self.embed_documents,
            texts
        )
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async version of embed_query."""
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self.embed_query,
            text
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


def get_langchain_embeddings(
    embedder: Optional[Embedder] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None
) -> LangChainGeminiEmbeddings:
    """
    Get a LangChain-compatible embeddings instance.
    
    Args:
        embedder: Optional Embedder instance
        model: Optional model name
        api_key: Optional API key
    
    Returns:
        LangChainGeminiEmbeddings instance
    """
    return LangChainGeminiEmbeddings(
        embedder=embedder,
        model=model,
        api_key=api_key
    )

