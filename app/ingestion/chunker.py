"""Text chunking utilities for the RAG ingestion pipeline.

This module uses LangChain text splitters to chunk documents
into manageable pieces for embedding and storage.

Supported chunking methods (selected via the ``method`` parameter):

- ``recursive``  – RecursiveCharacterTextSplitter (default)
- ``character``  – CharacterTextSplitter
- ``token``      – TokenTextSplitter
- ``markdown``   – MarkdownTextSplitter
- ``nltk``       – NLTKTextSplitter
- ``spacy``      – SpacyTextSplitter
"""

from typing import List, Literal

from app.core.logging import get_logger
from app.core.constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

try:
	from langchain_text_splitters import (
		CharacterTextSplitter,
		MarkdownTextSplitter,
		RecursiveCharacterTextSplitter,
		TokenTextSplitter,
	)
	from langchain_core.documents import Document
except ImportError as e:  # pragma: no cover - import-time failure
	raise ImportError(
		"LangChain text splitters are not available. Install langchain-text-splitters and langchain-core."
	) from e


logger = get_logger(__name__)


# The canonical list of chunking methods exposed to the frontend.
CHUNKING_METHODS: List[str] = [
	"recursive",
	"character",
	"token",
	"markdown",
	"nltk",
	"spacy",
]

DEFAULT_CHUNKING_METHOD = "recursive"

ChunkingMethod = Literal[
	"recursive", "character", "token", "markdown", "nltk", "spacy"
]


def get_text_splitter(
	chunk_size: int = DEFAULT_CHUNK_SIZE,
	chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
	method: ChunkingMethod = DEFAULT_CHUNKING_METHOD,
):
	"""Create a LangChain text splitter for the requested *method*.

	Args:
		chunk_size: Target size of each text chunk.
		chunk_overlap: Overlap between consecutive chunks.
		method: One of the supported chunking method keys.

	Returns:
		A LangChain ``TextSplitter`` instance.
	"""

	logger.info(
		"Creating text splitter: method=%s, chunk_size=%d, chunk_overlap=%d",
		method, chunk_size, chunk_overlap,
	)

	if method == "character":
		return CharacterTextSplitter(
			chunk_size=chunk_size,
			chunk_overlap=chunk_overlap,
			separator="\n\n",
		)

	if method == "token":
		return TokenTextSplitter(
			chunk_size=chunk_size,
			chunk_overlap=chunk_overlap,
		)

	if method == "markdown":
		return MarkdownTextSplitter(
			chunk_size=chunk_size,
			chunk_overlap=chunk_overlap,
		)

	if method == "nltk":
		try:
			from langchain_text_splitters import NLTKTextSplitter
		except ImportError:
			logger.warning("NLTKTextSplitter unavailable, falling back to recursive")
			return _recursive_splitter(chunk_size, chunk_overlap)
		return NLTKTextSplitter(
			chunk_size=chunk_size,
			chunk_overlap=chunk_overlap,
		)

	if method == "spacy":
		try:
			from langchain_text_splitters import SpacyTextSplitter
		except ImportError:
			logger.warning("SpacyTextSplitter unavailable, falling back to recursive")
			return _recursive_splitter(chunk_size, chunk_overlap)
		return SpacyTextSplitter(
			chunk_size=chunk_size,
			chunk_overlap=chunk_overlap,
		)

	# Default: recursive
	return _recursive_splitter(chunk_size, chunk_overlap)


def _recursive_splitter(chunk_size: int, chunk_overlap: int):
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
	method: ChunkingMethod = DEFAULT_CHUNKING_METHOD,
) -> List["Document"]:
	"""Split LangChain documents into smaller chunks.

	Args:
		documents: List of documents to split.
		chunk_size: Target size of each chunk.
		chunk_overlap: Overlap between consecutive chunks.
		method: Chunking method to use.

	Returns:
		List of chunked Document objects.
	"""

	if not documents:
		logger.warning("No documents provided for chunking")
		return []

	splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, method=method)
	chunks = splitter.split_documents(documents)
	logger.info(f"Chunked {len(documents)} documents into {len(chunks)} chunks (method={method})")
	return chunks


__all__ = ["CHUNKING_METHODS", "DEFAULT_CHUNKING_METHOD", "ChunkingMethod", "get_text_splitter", "chunk_documents"]
