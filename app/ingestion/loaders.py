"""Document loaders for the RAG ingestion pipeline.

This module uses LangChain loaders to read text documents from disk.
"""

from pathlib import Path
from typing import List

from app.core.logging import get_logger
from app.core.constants import DEFAULT_INGESTION_GLOB

try:
	from langchain_community.document_loaders import DirectoryLoader, TextLoader
	from langchain_core.documents import Document
except ImportError as e:  # pragma: no cover - import-time failure
	raise ImportError(
		"LangChain loaders are not available. Install langchain-community and langchain-core."
	) from e


logger = get_logger(__name__)


def load_text_documents(folder_path: str, glob: str = DEFAULT_INGESTION_GLOB) -> List["Document"]:
	"""Load .txt documents from a folder using LangChain's DirectoryLoader.

	Args:
		folder_path: Path to the folder that contains text files.
		glob: Glob pattern for files (defaults to **/*.txt).

	Returns:
		List of LangChain Document objects.
	"""

	base_path = Path(folder_path).expanduser().resolve()
	if not base_path.exists() or not base_path.is_dir():
		raise ValueError(f"Folder does not exist or is not a directory: {base_path}")

	logger.info(f"Loading documents from {base_path} with pattern '{glob}'")

	loader = DirectoryLoader(
		str(base_path),
		glob=glob,
		loader_cls=TextLoader,
		show_progress=True,
		loader_kwargs={"encoding": "utf-8"},
	)

	docs: List[Document] = loader.load()
	logger.info(f"Loaded {len(docs)} documents from {base_path}")
	return docs


__all__ = ["load_text_documents"]
