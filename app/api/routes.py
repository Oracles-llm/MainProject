"""
FastAPI routes for the RAG chat API.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import TYPE_CHECKING
import json
from app.api.schemas import ChatRequest, ChatResponse
from app.core.logging import get_logger
from app.core.config import settings
from app.llm import LLMClient
from app.llm.prompts import build_chat_prompt_string

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])

if TYPE_CHECKING:
    from app.services.rag_service import RAGService


def stream_event(event_type: str, **payload) -> str:
    """Serialize one streaming response event as an NDJSON line."""
    return json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n"


def token_events(token_stream):
    """Wrap a plain token iterator in the UI streaming event protocol."""
    for token in token_stream:
        if token:
            yield stream_event("token", content=token)
    yield stream_event("done")


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
        logger.info(f"Received chat request: mode='{request.mode}', query='{request.query[:50]}...'")

        if settings.DISABLE_RAG:
            llm_client = get_llm_client()
            response = llm_client.chat(user_query=request.query)
            return ChatResponse(response=response)

        rag_service = get_rag_service()
        if request.mode == "thinking":
            response = rag_service.thinking_query(query=request.query)
        else:
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
        logger.info(
            f"Received streaming chat request: mode='{request.mode}', query='{request.query[:50]}...'"
        )

        if settings.DISABLE_RAG:
            llm_client = get_llm_client()
            prompt = build_chat_prompt_string(user_query=request.query)
            token_stream = token_events(llm_client.stream(prompt))
        else:
            rag_service = get_rag_service()
            if request.mode == "thinking":
                token_stream = rag_service.thinking_stream_events(query=request.query)
            else:
                token_stream = token_events(rag_service.stream(query=request.query))

        return StreamingResponse(
            token_stream,
            media_type="application/x-ndjson; charset=utf-8",
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

