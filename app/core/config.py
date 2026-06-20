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
    
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "fastembed")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    EMBEDDING_MODEL_PATH: Optional[str] = os.getenv("EMBEDDING_MODEL_PATH", None)
    EMBEDDING_N_CTX: int = int(os.getenv("EMBEDDING_N_CTX", "8192"))
    EMBEDDING_N_THREADS: Optional[int] = int(os.getenv("EMBEDDING_N_THREADS")) if os.getenv("EMBEDDING_N_THREADS") else None
    EMBEDDING_N_GPU_LAYERS: int = int(os.getenv("EMBEDDING_N_GPU_LAYERS", "0"))
    EMBEDDING_VERBOSE: bool = os.getenv("EMBEDDING_VERBOSE", "False").lower() == "true"
    EMBEDDING_CACHE_DIR: Optional[str] = os.getenv("EMBEDDING_CACHE_DIR", None)
    EMBEDDING_USE_CUDA: bool = os.getenv("EMBEDDING_USE_CUDA", "False").lower() == "true"
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", None)
    
    LLM_MODEL_PATH: Optional[str] = os.getenv("LLM_MODEL_PATH", None)
    LLM_N_CTX: int = int(os.getenv("LLM_N_CTX", "2048"))
    LLM_N_THREADS: Optional[int] = int(os.getenv("LLM_N_THREADS")) if os.getenv("LLM_N_THREADS") else None
    LLM_N_GPU_LAYERS: int = int(os.getenv("LLM_N_GPU_LAYERS", "0"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", "0.9"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "512"))
    LLM_VERBOSE: bool = os.getenv("LLM_VERBOSE", "False").lower() == "true"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "llama_cpp")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY", None)
    LLM_BASE_URL: Optional[str] = os.getenv("LLM_BASE_URL", None)

    DISABLE_RAG: bool = os.getenv("DISABLE_RAG", "False").lower() == "true"
    
    HYBRID_SEARCH_SCORE_THRESHOLD: Optional[float] = float(os.getenv("HYBRID_SEARCH_SCORE_THRESHOLD", "0.55"))
    RERANKER_TOP_K: int = int(os.getenv("RERANKER_TOP_K", "5"))


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

