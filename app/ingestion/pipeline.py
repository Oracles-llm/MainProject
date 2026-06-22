"""End-to-end data ingestion pipeline for RAG.

This pipeline:

1. Loads .txt files from a folder using LangChain loaders.
2. Chunks documents using LangChain text splitters.
3. Generates embeddings via LangChain-compatible Gemini embeddings.
4. Writes vectors + payloads into Qdrant using LangChain's Qdrant integration.

All heavy lifting for loading, chunking, embedding, and vector store
integration is delegated to LangChain components as requested.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
from uuid import uuid4

from app.core.config import settings
from app.core.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_INGESTION_GLOB
from app.core.logging import get_logger
from app.db.qdrant_client import QdrantDB, get_qdrant_db
from app.db.models import VectorPoint
from app.embeddings.embedder import get_langchain_embeddings
from app.ingestion.loaders import load_text_documents
from app.ingestion.chunker import chunk_documents, DEFAULT_CHUNKING_METHOD, ChunkingMethod
from app.retrieval import get_sparse_vector_generator

try:
	from langchain_core.embeddings import Embeddings as LCEmbeddings
	from langchain_core.documents import Document
except ImportError as e:  # pragma: no cover - import-time failure
	raise ImportError(
		"LangChain core interfaces are not available. Install langchain-core."
	) from e


logger = get_logger(__name__)


@dataclass
class IngestionConfig:
	"""Configuration for the ingestion pipeline."""

	collection_name: str = settings.QDRANT_COLLECTION_NAME
	file_glob: str = DEFAULT_INGESTION_GLOB
	chunk_size: int = DEFAULT_CHUNK_SIZE
	chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
	chunking_method: str = DEFAULT_CHUNKING_METHOD
	enable_sparse_vectors: bool = True
	recreate_collection: bool = False


class IngestionPipeline:
	"""High-level ingestion pipeline built on LangChain components."""

	def __init__(
		self,
		config: Optional[IngestionConfig] = None,
		embeddings: Optional[LCEmbeddings] = None,
	) -> None:
		self.config = config or IngestionConfig()
		self.embeddings: LCEmbeddings = embeddings or get_langchain_embeddings()
		self._sparse_gen = get_sparse_vector_generator(model_name="Qdrant/bm25")

		if self._sparse_gen is None:
			self.config.enable_sparse_vectors = False
			logger.warning(
				"fastembed is unavailable. Ingestion will continue with dense-only vectors."
			)

	def _ensure_collection(self, qdrant: QdrantDB) -> None:
		"""Ensure the target Qdrant collection exists with correct settings."""

		logger.info(
			"Ensuring Qdrant collection '%s' exists (dim=%s, sparse=%s)",
			self.config.collection_name,
			settings.EMBEDDING_DIMENSION,
			self.config.enable_sparse_vectors,
		)

		qdrant.ensure_collection(
			collection_name=self.config.collection_name,
			vector_size=settings.EMBEDDING_DIMENSION,
			enable_sparse_vectors=self.config.enable_sparse_vectors,
			recreate_on_dimension_mismatch=self.config.recreate_collection,
		)

	def _build_payload(self, doc: Document, chunk_index: int) -> Dict[str, Any]:
		"""Create payload dictionary for a single chunked document."""

		metadata = dict(doc.metadata or {})
		metadata.setdefault("source", metadata.get("source", metadata.get("file_path")))
		metadata["chunk_index"] = chunk_index
		return {"text": doc.page_content, **metadata}

	def run(self, folder_path: str) -> int:
		"""Run the full ingestion pipeline for a folder of .txt files.

		Args:
			folder_path: Path to folder containing text files.

		Returns:
			Number of chunks ingested into Qdrant.
		"""

		base_path = Path(folder_path).expanduser().resolve()
		logger.info("Starting ingestion pipeline for folder: %s", base_path)

		# Reuse the global singleton — local Qdrant (SQLite-backed) holds an
		# exclusive file lock, so only one QdrantClient may be open on the same
		# path at a time.  Creating a second QdrantDB() here would conflict with
		# the client already held by VectorRetriever / RAGService.
		qdrant = get_qdrant_db()

		# Reset or create the collection before loading documents so
		# --recreate-collection also fixes stale schemas in empty folders.
		self._ensure_collection(qdrant)

		# 1. Load documents
		docs = load_text_documents(str(base_path), glob=self.config.file_glob)
		if not docs:
			logger.warning("No documents found to ingest in %s", base_path)
			return 0

		# 2. Chunk documents
		chunks = chunk_documents(
			documents=docs,
			chunk_size=self.config.chunk_size,
			chunk_overlap=self.config.chunk_overlap,
			method=self.config.chunking_method,
		)
		if not chunks:
			logger.warning("No chunks produced from documents in %s", base_path)
			return 0

		# 3. Generate dense and sparse vectors
		logger.info("Generating dense embeddings for %d chunks using LangChain", len(chunks))
		texts = [c.page_content for c in chunks]
		dense_vectors = self.embeddings.embed_documents(texts)

		if len(dense_vectors) != len(chunks):
			raise ValueError(
				f"Mismatch: {len(dense_vectors)} embeddings for {len(chunks)} chunks"
			)

		sparse_vectors = [None] * len(chunks)
		if self._sparse_gen and self.config.enable_sparse_vectors:
			logger.info("Generating sparse BM25 embeddings for %d chunks", len(chunks))
			sparse_vectors = self._sparse_gen.embed_documents(texts)

			if len(sparse_vectors) != len(chunks):
				raise ValueError(
					f"Mismatch: {len(sparse_vectors)} sparse embeddings for {len(chunks)} chunks"
				)
		else:
			logger.info("Skipping sparse BM25 embeddings; dense-only ingestion is active")

		# 4. Upsert points
		logger.info(
			"Upserting %d dense + sparse points into Qdrant collection '%s'",
			len(chunks),
			self.config.collection_name,
		)

		points = []
		for idx, (doc, dense, sparse) in enumerate(zip(chunks, dense_vectors, sparse_vectors)):
			payload = self._build_payload(doc, chunk_index=idx)
			point = VectorPoint(
				id=str(uuid4()),
				vector=dense,
				payload=payload,
				sparse_vectors={"bm25": sparse} if sparse else None,
			)
			points.append(point.to_point_struct())

		qdrant.upsert_points(points, collection_name=self.config.collection_name)

		logger.info("Ingestion pipeline completed successfully with %d chunks", len(chunks))
		return len(chunks)


__all__ = ["IngestionConfig", "IngestionPipeline"]
