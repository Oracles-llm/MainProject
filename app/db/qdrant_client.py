"""
Qdrant client for vector database operations.
Handles connection, collection management, and vector operations.
"""

from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    SparseVectorParams,
    VectorsConfig
)
from qdrant_client.http import models
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class QdrantDB:
    """Qdrant database client wrapper for vector operations.
    
    Supports multiple modes:
    - Local in-memory: QDRANT_MODE=memory or QDRANT_LOCAL_PATH=:memory:
    - Local persistent: QDRANT_MODE=local with QDRANT_LOCAL_PATH set
    - Server mode: QDRANT_MODE=server (default)
    - Cloud mode: QDRANT_MODE=cloud with QDRANT_URL set
    """
    
    def __init__(
        self,
        mode: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        local_path: Optional[str] = None
    ):
        """
        Initialize Qdrant client.
        
        Args:
            mode: Connection mode - 'memory', 'local', 'server', or 'cloud'
            host: Qdrant host (for server mode, defaults to settings.QDRANT_HOST)
            port: Qdrant port (for server mode, defaults to settings.QDRANT_PORT)
            collection_name: Collection name (defaults to settings.QDRANT_COLLECTION_NAME)
            api_key: Optional API key for cloud Qdrant
            url: Optional Qdrant Cloud URL (for cloud mode)
            local_path: Path for local persistent storage (for local mode)
        """
        self.mode = mode or settings.QDRANT_MODE.lower()
        self.host = host or settings.QDRANT_HOST
        self.port = port or settings.QDRANT_PORT
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.api_key = api_key or settings.QDRANT_API_KEY
        self.url = url or settings.QDRANT_URL
        self.local_path = local_path or settings.QDRANT_LOCAL_PATH
        
        self._client = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Qdrant based on mode."""
        try:
            if self.mode == "memory":
                self._client = QdrantClient(":memory:")
                logger.info("Connected to Qdrant in-memory mode")
            elif self.mode == "local":
                if self.local_path:
                    self._client = QdrantClient(path=self.local_path)
                    logger.info(f"Connected to Qdrant local persistent mode: {self.local_path}")
                else:
                    self._client = QdrantClient(":memory:")
                    logger.warning("QDRANT_LOCAL_PATH not set, using in-memory mode")
            elif self.mode == "cloud":
                if not self.url:
                    raise ValueError("QDRANT_URL is required for cloud mode")
                self._client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key
                )
                logger.info(f"Connected to Qdrant Cloud: {self.url}")
            else:
                self._client = QdrantClient(
                    host=self.host,
                    port=self.port
                )
                logger.info(f"Connected to Qdrant server at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    @property
    def client(self) -> QdrantClient:
        """Get Qdrant client instance."""
        return self._client
    
    def create_collection(
        self,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
        distance: Distance = Distance.COSINE,
        enable_sparse_vectors: bool = True
    ) -> bool:
        """
        Create a new collection in Qdrant.
        
        Args:
            collection_name: Name of the collection
            vector_size: Size of the vectors (defaults to settings.EMBEDDING_DIMENSION)
            distance: Distance metric (COSINE, EUCLID, DOT)
        
        Returns:
            True if collection was created, False if it already exists
        """
        collection_name = collection_name or self.collection_name
        vector_size = vector_size or settings.EMBEDDING_DIMENSION
        
        try:
            collections = self._client.get_collections().collections
            existing_collections = [col.name for col in collections]
            
            if collection_name in existing_collections:
                logger.info(f"Collection '{collection_name}' already exists")
                return False
            
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                ),
                sparse_vectors_config={
                    "bm25": SparseVectorParams()
                } if enable_sparse_vectors else None
            )
            
            logger.info(f"Collection '{collection_name}' created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            raise
    
    def collection_exists(self, collection_name: Optional[str] = None) -> bool:
        """
        Check if collection exists.
        
        Args:
            collection_name: Name of the collection
        
        Returns:
            True if collection exists, False otherwise
        """
        collection_name = collection_name or self.collection_name
        
        try:
            collections = self._client.get_collections().collections
            return collection_name in [col.name for col in collections]
        except Exception as e:
            logger.error(f"Failed to check collection existence: {e}")
            return False
    
    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_name: Name of the collection to delete
        
        Returns:
            True if deleted successfully
        """
        collection_name = collection_name or self.collection_name
        
        try:
            self._client.delete_collection(collection_name)
            logger.info(f"Collection '{collection_name}' deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            raise
    
    def upsert_points(
        self,
        points: List[PointStruct],
        collection_name: Optional[str] = None
    ) -> bool:
        """
        Insert or update points in the collection.
        
        Args:
            points: List of PointStruct objects to upsert
            collection_name: Name of the collection
        
        Returns:
            True if successful
        """
        collection_name = collection_name or self.collection_name
        
        try:
            self._client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.debug(f"Upserted {len(points)} points to '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert points: {e}")
            raise
    
    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        collection_name: Optional[str] = None,
        score_threshold: Optional[float] = None,
        filter: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector to search for
            limit: Number of results to return
            collection_name: Name of the collection
            score_threshold: Minimum similarity score
            filter: Optional filter conditions
        
        Returns:
            List of search results with payload and scores
        """
        collection_name = collection_name or self.collection_name
        
        try:
            response = self._client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filter
            )
            
            search_results = []
            for point in response.points:
                search_results.append({
                    "id": str(point.id),
                    "score": point.score if hasattr(point, 'score') else 0.0,
                    "payload": point.payload if hasattr(point, 'payload') else {}
                })
            
            logger.debug(f"Found {len(search_results)} results")
            return search_results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def hybrid_search(
        self,
        query_vector: List[float],
        query_sparse_vector: Optional[Dict[int, float]] = None,
        limit: int = 10,
        collection_name: Optional[str] = None,
        score_threshold: Optional[float] = None,
        filter: Optional[Filter] = None,
        prefetch_limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining dense (semantic) and sparse (BM25) vectors.
        
        Args:
            query_vector: Dense query vector (semantic embedding)
            query_sparse_vector: Sparse query vector (BM25 keywords) as {token_id: score}
            limit: Number of results to return
            collection_name: Name of the collection
            score_threshold: Minimum similarity score
            filter: Optional filter conditions
            prefetch_limit: Number of candidates to fetch from each sub-query
        
        Returns:
            List of search results with payload and scores
        """
        collection_name = collection_name or self.collection_name
        
        try:
            from qdrant_client.models import Prefetch, SparseVector, FusionQuery, Fusion
            
            if query_sparse_vector:
                indices = list(query_sparse_vector.keys())
                values = list(query_sparse_vector.values())
                sparse_vector_obj = SparseVector(
                    indices=indices,
                    values=values
                )
                
                prefetch_queries = [
                    Prefetch(
                        query=query_vector,
                        limit=prefetch_limit
                    ),
                    Prefetch(
                        query=sparse_vector_obj,
                        using="bm25",
                        limit=prefetch_limit
                    )
                ]
                
                fusion_query = FusionQuery(fusion=Fusion.RRF)
                
                if score_threshold is not None:
                    logger.debug(f"Hybrid search with score_threshold={score_threshold}")
                
                response = self._client.query_points(
                    collection_name=collection_name,
                    prefetch=prefetch_queries,
                    query=fusion_query,
                    limit=limit,
                    score_threshold=score_threshold,
                    query_filter=filter
                )
            else:
                response = self._client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=limit,
                    score_threshold=score_threshold,
                    query_filter=filter
                )
            
            search_results = []
            for point in response.points:
                score = point.score if hasattr(point, 'score') else 0.0
                # Apply score threshold filtering (Qdrant may not filter RRF scores correctly)
                if score_threshold is not None and score < score_threshold:
                    continue
                search_results.append({
                    "id": str(point.id),
                    "score": score,
                    "payload": point.payload if hasattr(point, 'payload') else {}
                })
            
            if score_threshold is not None:
                logger.debug(f"Found {len(search_results)} results from hybrid search (filtered by threshold {score_threshold})")
            else:
                logger.debug(f"Found {len(search_results)} results from hybrid search")
            return search_results
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise
    
    def get_point(self, point_id: str, collection_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific point by ID.
        
        Args:
            point_id: ID of the point to retrieve
            collection_name: Name of the collection
        
        Returns:
            Point data with vector and payload, or None if not found
        """
        collection_name = collection_name or self.collection_name
        
        try:
            result = self._client.retrieve(
                collection_name=collection_name,
                ids=[point_id]
            )
            
            if result:
                point = result[0]
                return {
                    "id": point.id,
                    "vector": point.vector,
                    "payload": point.payload
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get point {point_id}: {e}")
            return None
    
    def delete_points(
        self,
        point_ids: List[str],
        collection_name: Optional[str] = None
    ) -> bool:
        """
        Delete points by IDs.
        
        Args:
            point_ids: List of point IDs to delete
            collection_name: Name of the collection
        
        Returns:
            True if successful
        """
        collection_name = collection_name or self.collection_name
        
        try:
            self._client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=point_ids
                )
            )
            logger.info(f"Deleted {len(point_ids)} points from '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete points: {e}")
            raise
    
    def get_collection_info(self, collection_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get collection information.
        
        Args:
            collection_name: Name of the collection
        
        Returns:
            Collection information dictionary
        """
        collection_name = collection_name or self.collection_name
        
        try:
            info = self._client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance,
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None
    
    def ensure_collection(
        self,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
        distance: Distance = Distance.COSINE,
        enable_sparse_vectors: bool = True
    ):
        """
        Ensure collection exists, create if it doesn't.
        
        Args:
            collection_name: Name of the collection
            vector_size: Size of the vectors
            distance: Distance metric
            enable_sparse_vectors: Whether to enable sparse vectors for hybrid search
        """
        collection_name = collection_name or self.collection_name
        
        if not self.collection_exists(collection_name):
            self.create_collection(collection_name, vector_size, distance, enable_sparse_vectors)
        else:
            logger.debug(f"Collection '{collection_name}' already exists")
    
    def close(self):
        """Close the Qdrant client connection."""
        if self._client:
            try:
                self._client.close()
                logger.debug("Qdrant client connection closed")
            except Exception as e:
                logger.warning(f"Error closing Qdrant client: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


_qdrant_db_instance = None

def get_qdrant_db() -> QdrantDB:
    """Get or create the global QdrantDB instance (lazy initialization)."""
    global _qdrant_db_instance
    if _qdrant_db_instance is None:
        _qdrant_db_instance = QdrantDB()
    return _qdrant_db_instance

# qdrant_db = get_qdrant_db() 