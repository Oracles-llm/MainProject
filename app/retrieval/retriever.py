"""
Vector-based document retriever using embeddings and Qdrant.
Implements retrieval with LangChain compatibility.
"""

from typing import List, Optional, Dict, Any, Union
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document as LangChainDocument
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from app.core.config import settings
from app.core.logging import get_logger
from app.db.qdrant_client import QdrantDB, get_qdrant_db, CollectionDimensionMismatchError
from app.db.models import SearchResult
from app.embeddings import get_embedder, Embedder
from app.retrieval.sparse_vectors import SparseVectorGenerator

logger = get_logger(__name__)


class VectorRetriever:
    """Vector-based document retriever using embeddings and Qdrant."""
    
    def __init__(
        self,
        qdrant_client: Optional[QdrantDB] = None,
        embedder: Optional[Embedder] = None,
        collection_name: Optional[str] = None,
        k: int = 10,
        score_threshold: Optional[float] = None,
        use_hybrid_search: bool = True,
        sparse_vector_generator: Optional[SparseVectorGenerator] = None
    ):
        """
        Initialize the vector retriever.
        
        Args:
            qdrant_client: QdrantDB client instance (defaults to global qdrant_db)
            embedder: Embedder instance (defaults to get_embedder())
            collection_name: Collection name to search in
            k: Number of documents to retrieve (default: 10)
            score_threshold: Minimum similarity score threshold (optional, defaults to HYBRID_SEARCH_SCORE_THRESHOLD from env)
            use_hybrid_search: Whether to use hybrid search (BM25 + semantic)
            sparse_vector_generator: SparseVectorGenerator instance for BM25 (optional)
        """
        self.qdrant = qdrant_client or get_qdrant_db()
        self.embedder = embedder or get_embedder()
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.k = k
        self.score_threshold = score_threshold if score_threshold is not None else settings.HYBRID_SEARCH_SCORE_THRESHOLD
        self.use_hybrid_search = use_hybrid_search
        self.sparse_gen = sparse_vector_generator
        
        logger.info(f"VectorRetriever initialized with k={k}, collection={self.collection_name}, hybrid_search={use_hybrid_search}, score_threshold={self.score_threshold}")
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None,
        use_hybrid_search: Optional[bool] = None
    ) -> List[SearchResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text string
            k: Number of documents to retrieve (overrides instance default)
            score_threshold: Minimum similarity score (overrides instance default)
            filter: Optional Qdrant filter for metadata filtering
            use_hybrid_search: Override instance-level hybrid search setting
        
        Returns:
            List of SearchResult objects
        """
        k = k if k is not None else self.k
        # Use provided threshold, then instance default, then settings default
        if score_threshold is None:
            score_threshold = self.score_threshold if self.score_threshold is not None else settings.HYBRID_SEARCH_SCORE_THRESHOLD
        use_hybrid = use_hybrid_search if use_hybrid_search is not None else self.use_hybrid_search
        
        try:
            logger.debug(f"Retrieving documents for query: {query[:50]}...")

            existing_size = self.qdrant.get_collection_vector_size(self.collection_name)
            if existing_size is not None and existing_size != settings.EMBEDDING_DIMENSION:
                raise CollectionDimensionMismatchError(
                    f"Collection '{self.collection_name}' is storing {existing_size}-dim vectors, "
                    f"but the current embedding model produces {settings.EMBEDDING_DIMENSION}-dim vectors. "
                    f"Recreate the collection and re-ingest documents."
                )
            
            query_embedding = self.embedder.embed_query(query)
            
            if not query_embedding:
                logger.warning("Failed to generate query embedding")
                return []
            
            qdrant_filter = None
            if filter:
                try:
                    from qdrant_client.models import Filter, FieldCondition, MatchValue
                    
                    conditions = []
                    for key, value in filter.items():
                        conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value))
                        )
                    if conditions:
                        qdrant_filter = Filter(must=conditions)
                except ImportError:
                    logger.warning("Could not import Qdrant filter models, skipping filter")
            
            # Hybrid search: combine dense and sparse vectors
            if use_hybrid and self.sparse_gen:
                try:
                    query_sparse = self.sparse_gen.generate_query_sparse_vector(query)
                    if query_sparse:
                        if score_threshold is not None:
                            logger.debug(f"Using hybrid search (BM25 + semantic) with score_threshold={score_threshold}")
                        else:
                            logger.debug("Using hybrid search (BM25 + semantic) without score threshold")
                        results = self.qdrant.hybrid_search(
                            query_vector=query_embedding,
                            query_sparse_vector=query_sparse,
                            limit=k,
                            collection_name=self.collection_name,
                            score_threshold=score_threshold,
                            filter=qdrant_filter
                        )
                    else:
                        logger.warning("Failed to generate sparse vector, falling back to dense-only")
                        results = self.qdrant.search(
                            query_vector=query_embedding,
                            limit=k,
                            collection_name=self.collection_name,
                            score_threshold=score_threshold,
                            filter=qdrant_filter
                        )
                except Exception as e:
                    logger.warning(f"Hybrid search failed, falling back to dense-only: {e}")
                    results = self.qdrant.search(
                        query_vector=query_embedding,
                        limit=k,
                        collection_name=self.collection_name,
                        score_threshold=score_threshold,
                        filter=qdrant_filter
                    )
            else:
                # Fallback to dense-only search
                results = self.qdrant.search(
                    query_vector=query_embedding,
                    limit=k,
                    collection_name=self.collection_name,
                    score_threshold=score_threshold,
                    filter=qdrant_filter
                )
            
            search_results = [
                SearchResult.from_qdrant_result(result) for result in results
            ]
            
            logger.debug(f"Retrieved {len(search_results)} documents")
            return search_results
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise
    
    def retrieve_as_strings(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Retrieve documents and return only text content.
        
        Args:
            query: Query text string
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score
            filter: Optional metadata filter
        
        Returns:
            List of document text strings
        """
        results = self.retrieve(query, k=k, score_threshold=score_threshold, filter=filter)
        return [result.text for result in results]
    
    def retrieve_with_metadata(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with full metadata.
        
        Args:
            query: Query text string
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score
            filter: Optional metadata filter
        
        Returns:
            List of dictionaries with id, text, score, and metadata
        """
        results = self.retrieve(query, k=k, score_threshold=score_threshold, filter=filter)
        return [
            {
                "id": result.id,
                "text": result.text,
                "score": result.score,
                "metadata": result.metadata
            }
            for result in results
        ]


class LangChainVectorRetriever(BaseRetriever):
    """LangChain-compatible retriever wrapper for vector search."""
    
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        **kwargs
    ):
        """
        Initialize LangChain retriever.
        
        Args:
            vector_retriever: VectorRetriever instance to wrap
            **kwargs: Additional arguments for BaseRetriever
        """
        super().__init__(**kwargs)
        self.vector_retriever = vector_retriever
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[LangChainDocument]:
        """
        Get relevant documents for a query (LangChain interface).
        
        Args:
            query: Query string
            run_manager: Callback manager
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score
            filter: Optional metadata filter
        
        Returns:
            List of LangChain Document objects
        """
        results = self.vector_retriever.retrieve(
            query=query,
            k=k,
            score_threshold=score_threshold,
            filter=filter
        )
        
        return [
            LangChainDocument(
                page_content=result.text,
                metadata={
                    "id": result.id,
                    "score": result.score,
                    **result.metadata
                }
            )
            for result in results
        ]
    
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[LangChainDocument]:
        """Async version of _get_relevant_documents."""
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._get_relevant_documents(
                query,
                run_manager=run_manager,
                k=k,
                score_threshold=score_threshold,
                filter=filter
            )
        )


def get_retriever(
    qdrant_client: Optional[QdrantDB] = None,
    embedder: Optional[Embedder] = None,
    collection_name: Optional[str] = None,
    k: int = 10,
    score_threshold: Optional[float] = None,
    langchain_compatible: bool = False,
    use_hybrid_search: bool = True,
    sparse_vector_generator: Optional[SparseVectorGenerator] = None
) -> Union[VectorRetriever, LangChainVectorRetriever]:
    """
    Get a retriever instance.
    
    Args:
        qdrant_client: Optional QdrantDB client
        embedder: Optional Embedder instance
        collection_name: Optional collection name
        k: Number of documents to retrieve
        score_threshold: Optional minimum similarity score
        langchain_compatible: If True, returns LangChain-compatible retriever
        use_hybrid_search: Whether to use hybrid search (BM25 + semantic)
        sparse_vector_generator: Optional SparseVectorGenerator for hybrid search
    
    Returns:
        VectorRetriever or LangChainVectorRetriever instance
    """
    vector_retriever = VectorRetriever(
        qdrant_client=qdrant_client,
        embedder=embedder,
        collection_name=collection_name,
        k=k,
        score_threshold=score_threshold,
        use_hybrid_search=use_hybrid_search,
        sparse_vector_generator=sparse_vector_generator
    )
    
    if langchain_compatible:
        return LangChainVectorRetriever(vector_retriever)
    return vector_retriever

