"""
RAG (Retrieval-Augmented Generation) Service.
High-performance, scalable service for RAG workflows.
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from app.core.logging import get_logger
from app.retrieval import VectorRetriever, get_retriever, get_reranker, BaseReranker
from app.llm import LLMClient, get_llm_client
from app.db.models import SearchResult
from app.retrieval.reranker import RerankResult

logger = get_logger(__name__)


class RerankStrategy(str, Enum):
    """Reranking strategy options."""
    NONE = "none"
    BM25 = "bm25"


@dataclass
class RAGRequest:
    """RAG request parameters."""
    query: str
    k: int = 10
    rerank_top_k: Optional[int] = None
    use_reranking: bool = False
    rerank_strategy: RerankStrategy = RerankStrategy.NONE
    score_threshold: Optional[float] = None
    filter: Optional[Dict[str, Any]] = None
    chat_history: Optional[List[Tuple[str, str]]] = None
    system_prompt: Optional[str] = None
    streaming: bool = False


@dataclass
class RAGResponse:
    """RAG response with metadata."""
    answer: str
    query: str
    retrieved_documents: List[SearchResult]
    reranked_documents: Optional[List[RerankResult]] = None
    used_documents: List[SearchResult] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.used_documents is None:
            self.used_documents = self.retrieved_documents
        if self.metadata is None:
            self.metadata = {}


class RAGService:
    """
    High-performance RAG service that orchestrates retrieval, reranking, and generation.
    
    Designed for scalability and performance:
    - Dependency injection for all components
    - Configurable retrieval and reranking
    - Support for streaming responses
    - Comprehensive error handling
    - Performance monitoring
    """
    
    def __init__(
        self,
        retriever: Optional[VectorRetriever] = None,
        llm_client: Optional[LLMClient] = None,
        default_k: int = 10,
        default_rerank_top_k: Optional[int] = None,
        default_use_reranking: bool = False,
        default_rerank_strategy: RerankStrategy = RerankStrategy.NONE
    ):
        """
        Initialize RAG service.
        
        Args:
            retriever: VectorRetriever instance (creates new if not provided)
            llm_client: LLMClient instance (creates new if not provided)
            default_k: Default number of documents to retrieve
            default_rerank_top_k: Default number of top documents after reranking (defaults to RERANKER_TOP_K from env)
            default_use_reranking: Whether to use reranking by default
            default_rerank_strategy: Default reranking strategy
        """
        from app.core.config import settings
        
        self.retriever = retriever or get_retriever(k=default_k)
        self.llm_client = llm_client or get_llm_client()
        self.default_k = default_k
        self.default_rerank_top_k = default_rerank_top_k if default_rerank_top_k is not None else settings.RERANKER_TOP_K
        self.default_use_reranking = default_use_reranking
        self.default_rerank_strategy = default_rerank_strategy
        
        logger.info(f"RAGService initialized with default_rerank_top_k={self.default_rerank_top_k}")
    
    def query(
        self,
        query: str,
        k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        use_reranking: Optional[bool] = None,
        rerank_strategy: Optional[RerankStrategy] = None,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        system_prompt: Optional[str] = None
    ) -> RAGResponse:
        """
        Execute a RAG query with retrieval and generation.
        
        Args:
            query: User query string
            k: Number of documents to retrieve
            rerank_top_k: Number of top documents after reranking
            use_reranking: Whether to rerank results
            rerank_strategy: Reranking strategy to use
            score_threshold: Minimum similarity score
            filter: Optional metadata filter
            chat_history: Optional conversation history
            system_prompt: Optional custom system prompt
        
        Returns:
            RAGResponse with answer and metadata
        """
        k = k if k is not None else self.default_k
        rerank_top_k = rerank_top_k if rerank_top_k is not None else self.default_rerank_top_k
        use_reranking = use_reranking if use_reranking is not None else self.default_use_reranking
        rerank_strategy = rerank_strategy or self.default_rerank_strategy
        
        try:
            logger.debug(f"Processing RAG query: {query[:50]}...")
            
            retrieved_docs = self.retriever.retrieve(
                query=query,
                k=k,
                score_threshold=score_threshold,
                filter=filter
            )

            logger.debug(f"Retrieved documents: {retrieved_docs}")
            
            if not retrieved_docs:
                logger.warning(f"No documents retrieved for query: {query}")
                fallback_answer = self.llm_client.chat(
                    user_query=query,
                    chat_history=chat_history,
                    system_prompt=system_prompt
                )
                return RAGResponse(
                    answer=fallback_answer,
                    query=query,
                    retrieved_documents=[],
                    metadata={
                        "retrieval_count": 0,
                        "used_count": 0,
                        "reranked": False,
                        "rerank_strategy": None,
                        "fallback_to_chat": True
                    }
                )
            
            reranked_docs = None
            used_docs = retrieved_docs
            
            if use_reranking and len(retrieved_docs) > 0:
                try:
                    reranker = get_reranker(method=rerank_strategy.value)
                    reranked_results = reranker.rerank(
                        query=query,
                        results=retrieved_docs,
                        top_k=rerank_top_k
                    )
                    
                    reranked_docs = reranked_results
                    used_docs = [
                        SearchResult(
                            id=r.id,
                            score=r.final_score,
                            text=r.text,
                            metadata=r.metadata
                        )
                        for r in reranked_results
                    ]
                    
                    logger.debug(f"Reranked {len(retrieved_docs)} docs to {len(used_docs)}")
                except Exception as e:
                    logger.warning(f"Reranking failed, using original results: {e}")
            
            context_docs = [doc.text for doc in used_docs]
            
            answer = self.llm_client.rag(
                query=query,
                context_documents=context_docs,
                chat_history=chat_history,
                system_prompt=system_prompt
            )
            
            logger.debug(f"Generated answer with length: {len(answer)}")
            
            return RAGResponse(
                answer=answer,
                query=query,
                retrieved_documents=retrieved_docs,
                reranked_documents=reranked_docs,
                used_documents=used_docs,
                metadata={
                    "retrieval_count": len(retrieved_docs),
                    "used_count": len(used_docs),
                    "reranked": use_reranking,
                    "rerank_strategy": rerank_strategy.value if use_reranking else None
                }
            )
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}", exc_info=True)
            raise
    
    def query_with_request(self, request: RAGRequest) -> RAGResponse:
        """
        Execute RAG query using RAGRequest object.
        
        Args:
            request: RAGRequest object with all parameters
        
        Returns:
            RAGResponse with answer and metadata
        """
        return self.query(
            query=request.query,
            k=request.k,
            rerank_top_k=request.rerank_top_k,
            use_reranking=request.use_reranking,
            rerank_strategy=request.rerank_strategy,
            score_threshold=request.score_threshold,
            filter=request.filter,
            chat_history=request.chat_history,
            system_prompt=request.system_prompt
        )
    
    def stream(
        self,
        query: str,
        k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        use_reranking: Optional[bool] = None,
        rerank_strategy: Optional[RerankStrategy] = None,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Stream RAG response tokens.
        
        Args:
            query: User query string
            k: Number of documents to retrieve
            rerank_top_k: Number of top documents after reranking
            use_reranking: Whether to rerank results
            rerank_strategy: Reranking strategy
            score_threshold: Minimum similarity score
            filter: Optional metadata filter
            chat_history: Optional conversation history
            system_prompt: Optional custom system prompt
        
        Yields:
            Token strings from LLM generation
        """
        k = k if k is not None else self.default_k
        rerank_top_k = rerank_top_k if rerank_top_k is not None else self.default_rerank_top_k
        use_reranking = use_reranking if use_reranking is not None else self.default_use_reranking
        rerank_strategy = rerank_strategy or self.default_rerank_strategy
        
        try:
            retrieved_docs = self.retriever.retrieve(
                query=query,
                k=k,
                score_threshold=score_threshold,
                filter=filter
            )
            
            if not retrieved_docs:
                fallback_answer = self.llm_client.chat(
                    user_query=query,
                    chat_history=chat_history,
                    system_prompt=system_prompt
                )
                yield fallback_answer
                return
            
            used_docs = retrieved_docs
            
            if use_reranking and len(retrieved_docs) > 0:
                try:
                    reranker = get_reranker(method=rerank_strategy.value)
                    reranked_results = reranker.rerank(
                        query=query,
                        results=retrieved_docs,
                        top_k=rerank_top_k
                    )
                    used_docs = [
                        SearchResult(
                            id=r.id,
                            score=r.final_score,
                            text=r.text,
                            metadata=r.metadata
                        )
                        for r in reranked_results
                    ]
                except Exception as e:
                    logger.warning(f"Reranking failed, using original results: {e}")
            
            context_docs = [doc.text for doc in used_docs]
            
            from app.llm.prompts import format_context_documents, build_rag_prompt_string
            
            context_str = format_context_documents(context_docs)
            prompt = build_rag_prompt_string(
                user_query=query,
                context=context_str,
                chat_history=chat_history,
                system_prompt=system_prompt
            )
            
            stop_sequences = [
                "\n\nUser:",
                "\nUser:",
                "User:",
                "\n\nAssistant:",
                "\nAssistant:",
                "Assistant:"
            ]
            
            for token in self.llm_client.stream(prompt, stop=stop_sequences):
                yield token
                
        except Exception as e:
            logger.error(f"RAG stream failed: {e}", exc_info=True)
            raise
    
    def batch_query(
        self,
        queries: List[str],
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[RAGResponse]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of query strings
            k: Number of documents to retrieve per query
            score_threshold: Minimum similarity score
            filter: Optional metadata filter
        
        Returns:
            List of RAGResponse objects
        """
        responses = []
        
        for query in queries:
            try:
                response = self.query(
                    query=query,
                    k=k,
                    score_threshold=score_threshold,
                    filter=filter,
                    use_reranking=False
                )
                responses.append(response)
            except Exception as e:
                logger.error(f"Batch query failed for query '{query}': {e}")
                responses.append(RAGResponse(
                    answer=f"Error processing query: {str(e)}",
                    query=query,
                    retrieved_documents=[],
                    metadata={"error": str(e)}
                ))
        
        return responses


def get_rag_service(
    retriever: Optional[VectorRetriever] = None,
    llm_client: Optional[LLMClient] = None,
    default_k: int = 10,
    default_rerank_top_k: Optional[int] = None,
    use_hybrid_search: bool = True
) -> RAGService:
    """
    Get a RAG service instance.
    
    Args:
        retriever: Optional VectorRetriever instance
        llm_client: Optional LLMClient instance
        default_k: Default number of documents to retrieve
        default_rerank_top_k: Default number of top documents after reranking (defaults to RERANKER_TOP_K from env)
        use_hybrid_search: Whether to enable hybrid search (BM25 + semantic)
    
    Returns:
        RAGService instance
    """
    from app.core.config import settings
    
    if retriever is None and use_hybrid_search:
        from app.retrieval import get_sparse_vector_generator
        sparse_gen = get_sparse_vector_generator(model_name="Qdrant/bm25")
        if sparse_gen:
            from app.retrieval import get_retriever
            retriever = get_retriever(
                k=default_k,
                use_hybrid_search=True,
                sparse_vector_generator=sparse_gen
            )
            logger.info("RAG service initialized with hybrid search enabled")
        else:
            logger.warning("Sparse vector generator not available, using dense-only search")
            from app.retrieval import get_retriever
            retriever = get_retriever(k=default_k, use_hybrid_search=False)
    
    if default_rerank_top_k is None:
        default_rerank_top_k = settings.RERANKER_TOP_K
    
    return RAGService(
        retriever=retriever,
        llm_client=llm_client,
        default_k=default_k,
        default_rerank_top_k=default_rerank_top_k
    )

