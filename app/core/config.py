"""
Application configuration and settings.
Loads environment variables and initializes core components like logging.
"""

import os
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class Settings:
    """Application settings loaded from environment variables."""
    
    APP_NAME: str = os.getenv("APP_NAME", "LLM RAG Platform")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: str = os.getenv("LOG_DIR", "logs")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE", None)
    
    QDRANT_MODE: str = os.getenv("QDRANT_MODE", "server")
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "documents")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY", None)
    QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL", None)
    QDRANT_LOCAL_PATH: Optional[str] = os.getenv("QDRANT_LOCAL_PATH", None)
    
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY", None)
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    LLM_BASE_URL: Optional[str] = os.getenv("LLM_BASE_URL", None)


settings = Settings()


def initialize_logging():
    """Initialize the application logger using settings."""
    from app.core.logging import setup_logger
    
    logger = setup_logger(
        name="Oracales llm",
        log_level=settings.LOG_LEVEL,
        log_file=settings.LOG_FILE,
        log_dir=settings.LOG_DIR
    )
    
    logger.info(f"Logger initialized with level: {settings.LOG_LEVEL}")
    return logger


logger = initialize_logging()

