"""
LLM client using LangChain with llama.cpp backend.
Supports self-hosted models with configurable parameters.
"""

from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from langchain_community.llms import LlamaCpp
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import CallbackManager
from langchain_core.outputs import LLMResult

from app.core.config import settings
from app.core.logging import get_logger
from app.llm.prompts import (
    build_rag_prompt_string,
    build_chat_prompt_string,
    format_context_documents
)

logger = get_logger(__name__)


class LLMClient:
    """LLM client wrapper using LangChain with llama.cpp backend."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: Optional[int] = None,
        n_threads: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        verbose: Optional[bool] = None,
        callback_manager: Optional[CallbackManager] = None
    ):
        """
        Initialize the LLM client.
        
        Args:
            model_path: Path to the llama.cpp model file (.gguf or .bin)
            n_ctx: Context window size
            n_threads: Number of CPU threads to use
            n_gpu_layers: Number of GPU layers (0 for CPU-only)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            verbose: Enable verbose logging
            callback_manager: Optional callback manager for streaming
        """
        self.model_path = model_path or settings.LLM_MODEL_PATH
        if not self.model_path:
            raise ValueError("LLM_MODEL_PATH must be provided either as parameter or environment variable")
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.n_ctx = n_ctx if n_ctx is not None else settings.LLM_N_CTX
        self.n_threads = n_threads if n_threads is not None else settings.LLM_N_THREADS
        self.n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else settings.LLM_N_GPU_LAYERS
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.top_p = top_p if top_p is not None else settings.LLM_TOP_P
        self.max_tokens = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS
        self.verbose = verbose if verbose is not None else settings.LLM_VERBOSE
        
        try:
            self.llm = LlamaCpp(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                verbose=self.verbose,
                callback_manager=callback_manager
            )
            logger.info(f"LLM client initialized with model: {self.model_path}")
            logger.info(f"Context: {self.n_ctx}, Threads: {self.n_threads}, GPU Layers: {self.n_gpu_layers}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt text
            stop: List of stop sequences
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text
        """
        try:
            logger.debug(f"Generating response with prompt length: {len(prompt)}")
            response = self.llm.invoke(prompt, stop=stop, **kwargs)
            logger.debug(f"Generated response length: {len(response)}")
            return response
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise
    
    def chat(
        self,
        user_query: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        context: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a chat response with optional history and context.
        
        Args:
            user_query: User's query/question
            chat_history: List of (role, message) tuples for chat history
            context: List of relevant document texts for RAG
            system_prompt: Custom system prompt
        
        Returns:
            Generated response
        """
        if context:
            context_str = format_context_documents(context)
            prompt_str = build_rag_prompt_string(
                user_query=user_query,
                context=context_str,
                chat_history=chat_history,
                system_prompt=system_prompt
            )
        else:
            prompt_str = build_chat_prompt_string(
                user_query=user_query,
                chat_history=chat_history,
                system_prompt=system_prompt
            )
        
        return self.generate(prompt_str)
    
    def rag(
        self,
        query: str,
        context_documents: List[str],
        chat_history: Optional[List[Tuple[str, str]]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a RAG response using context documents.
        
        Args:
            query: User's query
            context_documents: List of relevant document texts
            chat_history: Optional chat history
            system_prompt: Custom system prompt
        
        Returns:
            Generated response based on context
        """
        return self.chat(
            user_query=query,
            chat_history=chat_history,
            context=context_documents,
            system_prompt=system_prompt
        )
    
    def stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Stream generated tokens.
        
        Args:
            prompt: Input prompt text
            stop: List of stop sequences
            **kwargs: Additional generation parameters
        
        Yields:
            Generated token strings
        """
        try:
            for token in self.llm.stream(prompt, stop=stop, **kwargs):
                yield token
        except Exception as e:
            logger.error(f"Failed to stream response: {e}")
            raise
    
    @property
    def model(self) -> BaseLanguageModel:
        """Get the underlying LangChain LLM model."""
        return self.llm


def get_llm_client() -> LLMClient:
    """
    Get a default LLM client instance using settings.
    
    Returns:
        Configured LLMClient instance
    """
    return LLMClient()

