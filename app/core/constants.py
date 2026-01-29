"""Core constants used across the application.

These are intentionally minimal and focused on RAG ingestion defaults.
"""

# Ingestion defaults
# Default to text files; this can be overridden via IngestionConfig.
DEFAULT_INGESTION_GLOB = "**/*.txt"
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 60
