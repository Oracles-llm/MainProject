"""
FastAPI application with lifespan management for model loading.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import get_logger
from app.api.routes import router

logger = get_logger(__name__)

rag_service = None
llm_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Loads the model once on startup and cleans up on shutdown.
    """
    global rag_service
    global llm_client
    
    logger.info("=" * 60)
    logger.info("Starting RAG Chat API Application")
    logger.info("=" * 60)
    
    try:
        logger.info("Initializing RAG service...")
        logger.info(f"LLM Provider: {settings.LLM_PROVIDER}")
        logger.info(f"LLM Model Path: {settings.LLM_MODEL_PATH}")
        
        if settings.LLM_PROVIDER.lower() == "llama_cpp":
            if not settings.LLM_MODEL_PATH:
                raise ValueError(
                    "LLM_MODEL_PATH must be set in environment variables "
                    "when using llama_cpp provider"
                )
            logger.info(f"Loading model from: {settings.LLM_MODEL_PATH}")
            logger.info("This may take a few moments...")
        
        from app.llm import get_llm_client
        llm_client = get_llm_client()
        
        logger.info("Model loaded successfully!")
        logger.info(f"Context window: {settings.LLM_N_CTX}")
        logger.info(f"Temperature: {settings.LLM_TEMPERATURE}")
        logger.info(f"Max tokens: {settings.LLM_MAX_TOKENS}")
        
        if settings.DISABLE_RAG:
            logger.info("RAG disabled - routing queries directly to the LLM")
        else:
            logger.info("Initializing sparse vector generator for hybrid search...")
            from app.retrieval import get_sparse_vector_generator, get_retriever
            from app.services.rag_service import RAGService, RerankStrategy

            sparse_gen = get_sparse_vector_generator(model_name="Qdrant/bm25")
            if sparse_gen:
                logger.info("Sparse vector generator initialized successfully - hybrid search enabled")
            else:
                logger.warning("Sparse vector generator not initialized - using dense-only search")

            retriever = get_retriever(
                k=8,
                use_hybrid_search=True if sparse_gen else False,
                sparse_vector_generator=sparse_gen
            )

            rag_service = RAGService(
                retriever=retriever,
                llm_client=llm_client,
                default_k=8,
                default_rerank_top_k=None,
                default_use_reranking=True,
                default_rerank_strategy=RerankStrategy.BM25
            )

            logger.info("RAG service initialized successfully!")
        logger.info("=" * 60)
        logger.info("Application ready to accept requests")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}", exc_info=True)
        raise
    
    yield
    
    logger.info("Shutting down application...")
    if llm_client:
        try:
            if hasattr(llm_client, 'llm') and hasattr(llm_client.llm, 'client'):
                if hasattr(llm_client.llm.client, 'close'):
                    llm_client.llm.client.close()
                    logger.info("LLM client closed")
        except Exception as e:
            logger.warning(f"Error closing LLM client: {e}")
    
    logger.info("Application shutdown complete")


app = FastAPI(
    title=settings.APP_NAME,
    description="RAG (Retrieval-Augmented Generation) Chat API with self-hosted LLM",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Chat API",
        "version": "1.0.0",
        "status": "running"
    }

