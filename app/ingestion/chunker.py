"""Text chunking utilities for the RAG ingestion pipeline.

This module uses LangChain text splitters to chunk documents
into manageable pieces for embedding and storage.
"""

from typing import List

from app.core.logging import get_logger
from app.core.constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

try:
	from langchain_text_splitters import RecursiveCharacterTextSplitter
	from langchain_core.documents import Document
except ImportError as e:  # pragma: no cover - import-time failure
	raise ImportError(
		"LangChain text splitters are not available. Install langchain-text-splitters and langchain-core."
	) from e


logger = get_logger(__name__)


def get_text_splitter(
	chunk_size: int = DEFAULT_CHUNK_SIZE,
	chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> "RecursiveCharacterTextSplitter":
	"""Create a RecursiveCharacterTextSplitter with sensible defaults.

	Args:
		chunk_size: Target size of each text chunk.
		chunk_overlap: Overlap between consecutive chunks.
	"""

	logger.info(
		f"Creating RecursiveCharacterTextSplitter with chunk_size={chunk_size}, "
		f"chunk_overlap={chunk_overlap}"
	)

	return RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
		length_function=len,
		separators=["\n\n", "\n", ". ", " ", ""],
	)


def chunk_documents(
	documents: List["Document"],
	chunk_size: int = DEFAULT_CHUNK_SIZE,
	chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List["Document"]:
	"""Split LangChain documents into smaller chunks.

	Args:
		documents: List of documents to split.
		chunk_size: Target size of each chunk.
		chunk_overlap: Overlap between consecutive chunks.

	Returns:
		List of chunked Document objects.
	"""

	if not documents:
		logger.warning("No documents provided for chunking")
		return []

	splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
	chunks = splitter.split_documents(documents)
	logger.info(f"Chunked {len(documents)} documents into {len(chunks)} chunks")
	return chunks


__all__ = ["get_text_splitter", "chunk_documents"]
