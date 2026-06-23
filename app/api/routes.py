"""
FastAPI routes for the RAG chat API.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import TYPE_CHECKING
from app.api.schemas import ChatRequest, ChatResponse
from app.core.logging import get_logger
from app.core.config import settings
from app.llm import LLMClient
from app.llm.prompts import build_chat_prompt_string

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])

if TYPE_CHECKING:
    from app.services.rag_service import RAGService


def get_rag_service() -> "RAGService":
    """Dependency to get RAG service instance."""
    import app.api.main as api_main
    if api_main.rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service not initialized. Please wait for the application to start."
        )
    return api_main.rag_service


def get_llm_client() -> LLMClient:
    """Dependency to get LLM client instance."""
    import app.api.main as api_main
    if api_main.llm_client is None:
        raise HTTPException(
            status_code=503,
            detail="LLM client not initialized. Please wait for the application to start."
        )
    return api_main.llm_client


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
        logger.info(f"Received chat request: query='{request.query[:50]}...'")

        if settings.DISABLE_RAG:
            llm_client = get_llm_client()
            response = llm_client.chat(user_query=request.query)
            return ChatResponse(response=response)

        rag_service = get_rag_service()
        response = rag_service.query(query=request.query)
        return ChatResponse(response=response.answer)
    
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


@router.post("/chat/stream")
async def stream_chat(request: ChatRequest) -> StreamingResponse:
    """
    Stream chat responses token by token.

    Keeps the existing /chat endpoint available for non-streaming clients while
    allowing the UI to render model output as soon as tokens are produced.
    """
    try:
        logger.info(f"Received streaming chat request: query='{request.query[:50]}...'")

        if settings.DISABLE_RAG:
            llm_client = get_llm_client()
            prompt = build_chat_prompt_string(user_query=request.query)
            token_stream = llm_client.stream(prompt)
        else:
            rag_service = get_rag_service()
            token_stream = rag_service.stream(query=request.query)

        return StreamingResponse(
            token_stream,
            media_type="text/plain; charset=utf-8",
            headers={"Cache-Control": "no-cache"}
        )

    except Exception as e:
        logger.error(f"Error starting streaming chat request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "RAG Chat API",
        "rag_enabled": not settings.DISABLE_RAG
    }

