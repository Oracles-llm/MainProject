"""CLI script to ingest a folder of documents into Qdrant.

Usage example:

	python scripts/ingest_data.py --folder data/texts

This script delegates loading, chunking, embedding, and storage to
LangChain-based components defined in app.ingestion.
"""

import argparse
import sys
from pathlib import Path

# Ensure app package is importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging import get_logger
from app.ingestion import IngestionPipeline, IngestionConfig


logger = get_logger(__name__)


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Ingest a folder of documents into Qdrant using LangChain."
	)
	parser.add_argument(
		"--folder",
		"-f",
		required=True,
		help="Path to folder containing source files (e.g. .md)",
	)
	parser.add_argument(
		"--chunk-size",
		type=int,
		default=800,
		help="Chunk size for text splitting (default: 800)",
	)
	parser.add_argument(
		"--chunk-overlap",
		type=int,
		default=200,
		help="Chunk overlap for text splitting (default: 200)",
	)
	parser.add_argument(
		"--glob",
		default=None,
		help=(
			"Glob pattern for files inside the folder (e.g. '**/*.md'). "
			"If omitted, the pipeline's default (currently '**/*.md') is used."
		),
	)
	parser.add_argument(
		"--recreate-collection",
		action="store_true",
		help="Delete and recreate the target Qdrant collection if the embedding dimension changed.",
	)

	args = parser.parse_args()

	folder = Path(args.folder).expanduser().resolve()
	if not folder.exists() or not folder.is_dir():
		print(f"Folder does not exist or is not a directory: {folder}")
		return 1

	config = IngestionConfig(
		chunk_size=args.chunk_size,
		chunk_overlap=args.chunk_overlap,
		recreate_collection=args.recreate_collection,
	)
	if args.glob is not None:
		config.file_glob = args.glob
	pipeline = IngestionPipeline(config=config)

	logger.info("Running ingestion pipeline for folder: %s", folder)
	try:
		num_chunks = pipeline.run(str(folder))
	except Exception as e:  # pragma: no cover - CLI error path
		logger.error("Ingestion failed: %s", e, exc_info=True)
		print(f"Ingestion failed: {e}")
		return 1

	print(f"Ingestion completed successfully. Chunks ingested: {num_chunks}")
	return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
	raise SystemExit(main())
