"""
Document re-ranker for improving retrieval results.
Supports re-ranking strategy using BM25.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from app.core.logging import get_logger
from app.db.models import SearchResult

logger = get_logger(__name__)


@dataclass
class RerankResult:
    """Re-ranked result with original and new scores."""
    
    id: str
    text: str
    original_score: float
    rerank_score: float
    metadata: Dict[str, Any]
    final_score: float
    
    @classmethod
    def from_search_result(
        cls,
        result: SearchResult,
        rerank_score: float,
        score_combination: str = "weighted"
    ) -> "RerankResult":
        """
        Create RerankResult from SearchResult.
        
        Args:
            result: Original SearchResult
            rerank_score: Score from reranker
            score_combination: How to combine scores ('weighted', 'rerank_only', 'average')
        
        Returns:
            RerankResult instance
        """
        if score_combination == "rerank_only":
            final_score = rerank_score
        elif score_combination == "average":
            final_score = (result.score + rerank_score) / 2
        else:  # weighted
            final_score = 0.7 * rerank_score + 0.3 * result.score
        
        return cls(
            id=result.id,
            text=result.text,
            original_score=result.score,
            rerank_score=rerank_score,
            metadata=result.metadata,
            final_score=final_score
        )


class BaseReranker:
    """Base class for re-rankers."""
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Re-rank search results.
        
        Args:
            query: Original query string
            results: List of SearchResult objects to re-rank
            top_k: Number of top results to return (None for all)
        
        Returns:
            List of RerankResult objects sorted by final_score
        """
        raise NotImplementedError


class BM25Reranker(BaseReranker):
    """BM25-based re-ranker using rank-bm25."""
    
    def __init__(self, corpus: Optional[List[str]] = None):
        """
        Initialize BM25 re-ranker.
        
        Args:
            corpus: Optional corpus of documents for BM25 initialization
        """
        self.corpus = corpus or []
        self._bm25 = None
        
        if self.corpus:
            self._initialize_bm25()
    
    def _initialize_bm25(self):
        """Initialize BM25 model with corpus."""
        try:
            from rank_bm25 import BM25Okapi
            from nltk.tokenize import word_tokenize
            import nltk
            
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            tokenized_corpus = [word_tokenize(doc.lower()) for doc in self.corpus]
            self._bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"BM25 re-ranker initialized with {len(self.corpus)} documents")
        except ImportError:
            logger.warning("rank-bm25 or nltk not installed. Install with: pip install rank-bm25 nltk")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize BM25: {e}")
            raise
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Re-rank results using BM25.
        
        Args:
            query: Query string
            results: List of SearchResult objects
            top_k: Number of top results to return
        
        Returns:
            List of RerankResult objects
        """
        if not results:
            return []
        
        if self._bm25 is None:
            corpus = [result.text for result in results]
            self.corpus = corpus
            self._initialize_bm25()
        
        try:
            from nltk.tokenize import word_tokenize
            
            tokenized_query = word_tokenize(query.lower())
            scores = self._bm25.get_scores(tokenized_query)
            
            reranked = []
            for result, score in zip(results, scores):
                rerank_result = RerankResult.from_search_result(result, float(score))
                reranked.append(rerank_result)
            
            reranked.sort(key=lambda x: x.final_score, reverse=True)
            
            if top_k:
                reranked = reranked[:top_k]
            
            logger.debug(f"Re-ranked {len(results)} results, returning top {len(reranked)}")
            return reranked
        except Exception as e:
            logger.error(f"BM25 re-ranking failed: {e}")
            return [
                RerankResult.from_search_result(result, result.score, "rerank_only")
                for result in results
            ]

class NoReranker(BaseReranker):
    """No-op re-ranker that returns results as-is."""
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Return results without re-ranking.
        
        Args:
            query: Query string (unused)
            results: List of SearchResult objects
            top_k: Number of top results to return
        
        Returns:
            List of RerankResult objects
        """
        reranked = [
            RerankResult.from_search_result(result, result.score, "rerank_only")
            for result in results
        ]
        
        if top_k:
            reranked = reranked[:top_k]
        
        return reranked


def get_reranker(
    method: str = "none",
    corpus: Optional[List[str]] = None
) -> BaseReranker:
    """
    Get a re-ranker instance.
    
    Args:
        method: Re-ranking method ('none', 'bm25')
        corpus: Corpus for BM25 initialization (optional)
    
    Returns:
        BaseReranker instance
    """
    if method == "bm25":
        return BM25Reranker(corpus=corpus)
    else:
        return NoReranker()

