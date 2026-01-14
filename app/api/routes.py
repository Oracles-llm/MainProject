"""
FastAPI routes for the RAG chat API.
"""

from fastapi import APIRouter, HTTPException
from app.api.schemas import ChatRequest, ChatResponse
from app.services.rag_service import RAGService
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


def get_rag_service() -> RAGService:
    """Dependency to get RAG service instance."""
    import app.api.main as api_main
    if api_main.rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service not initialized. Please wait for the application to start."
        )
    return api_main.rag_service


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint with RAG (Retrieval-Augmented Generation).
    
    Processes user queries by:
    1. Retrieving relevant documents from the vector database
    2. Generating a response using the LLM with context
    
    Args:
        request: Chat request with user query
    
    Returns:
        ChatResponse with generated response
    """
    try:
        rag_service = get_rag_service()
        logger.info(f"Received chat request: query='{request.query[:50]}...'")
        
        response = rag_service.query(query=request.query)
        
        return ChatResponse(response=response.answer)
    
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "RAG Chat API"
    }

