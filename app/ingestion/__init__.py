"""Ingestion package exposing the LangChain-based pipeline."""

from .loaders import load_text_documents
from .chunker import get_text_splitter, chunk_documents
from .pipeline import IngestionConfig, IngestionPipeline

__all__ = [
	"load_text_documents",
	"get_text_splitter",
	"chunk_documents",
	"IngestionConfig",
	"IngestionPipeline",
]
