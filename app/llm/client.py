"""
LLM client with multi-provider support.
Supports both self-hosted llama.cpp and provider models (Gemini).
"""

from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from enum import Enum
import re

from langchain_community.llms import LlamaCpp
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import CallbackManager

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    ChatGoogleGenerativeAI = None

from app.core.config import settings
from app.core.logging import get_logger
from app.llm.prompts import (
    build_rag_prompt_string,
    build_chat_prompt_string,
    format_context_documents,
    NO_CONTEXT_ANSWER
)

logger = get_logger(__name__)


RAG_STOP_SEQUENCES = [
    "\n\nUser:",
    "\nUser:",
    "User:",
    "\n\nAssistant:",
    "\nAssistant:",
    "\n\nSystem:",
    "\nSystem:",
    "System:",
]


def _ensure_complete_sentence(text: str) -> str:
    """Trim trailing incomplete sentence so the answer ends cleanly."""
    if not text:
        return text
    # If the text already ends with sentence-ending punctuation, it's fine
    if text.rstrip()[-1] in '.!?"':
        return text.rstrip()
    # Find the last sentence-ending punctuation
    last_period = text.rfind('.')
    last_excl = text.rfind('!')
    last_ques = text.rfind('?')
    last_end = max(last_period, last_excl, last_ques)
    if last_end > 0:
        return text[:last_end + 1].strip()
    # No sentence-ending punctuation at all — return as-is with a period
    return text.rstrip().rstrip(',;:') + '.'


def _deduplicate_sentences(text: str) -> str:
    """Remove duplicate sentences while preserving order.

    SLMs often fall into repetition loops, producing the same sentence
    two or three times.  This helper splits on sentence-ending punctuation,
    keeps only the first occurrence of each sentence, and rejoins them.
    """
    if not text:
        return text
    # Split into sentences (keep the delimiter attached)
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    seen: set = set()
    unique: list = []
    for sentence in raw_sentences:
        normalised = sentence.strip().lower()
        if normalised and normalised not in seen:
            seen.add(normalised)
            unique.append(sentence.strip())
    return ' '.join(unique)


def clean_rag_answer(answer: str) -> str:
    """Remove common local-model overgeneration after the first direct answer."""
    cleaned = answer.strip()
    if not cleaned:
        return cleaned

    if cleaned == NO_CONTEXT_ANSWER:
        return cleaned

    for marker in RAG_STOP_SEQUENCES:
        marker_index = cleaned.find(marker.strip())
        if marker_index > 0:
            cleaned = cleaned[:marker_index].strip()

    if NO_CONTEXT_ANSWER in cleaned and cleaned != NO_CONTEXT_ANSWER:
        cleaned = cleaned.replace(NO_CONTEXT_ANSWER, "").strip()

    # Join non-empty lines (allow multi-line answers from the model)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        cleaned = ' '.join(lines)

    cleaned = re.sub(r"^\s*\d+[\.\)]\s*", "", cleaned).strip()

    # Remove duplicate sentences (SLM repetition loops)
    cleaned = _deduplicate_sentences(cleaned)

    # Ensure the answer ends on a complete sentence
    cleaned = _ensure_complete_sentence(cleaned)
    return cleaned


class LLMProvider(str, Enum):
    """LLM provider options."""
    LLAMA_CPP = "llama_cpp"
    GEMINI = "gemini"


class LLMClient:
    """
    Unified LLM client wrapper supporting multiple providers.
    Supports: llama.cpp (self-hosted) and Gemini (provider API).
    """
    
    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        # LlamaCpp parameters
        model_path: Optional[str] = None,
        n_ctx: Optional[int] = None,
        n_threads: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        # Common parameters
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        verbose: Optional[bool] = None,
        callback_manager: Optional[CallbackManager] = None,
        # Gemini parameters
        model_name: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the LLM client.
        
        Args:
            provider: LLM provider (defaults to LLM_PROVIDER env var)
            model_path: Path to llama.cpp model file (for llama_cpp provider)
            n_ctx: Context window size (for llama_cpp)
            n_threads: Number of CPU threads (for llama_cpp)
            n_gpu_layers: Number of GPU layers (for llama_cpp)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            verbose: Enable verbose logging
            callback_manager: Optional callback manager
            model_name: Model name for Gemini (e.g., "gemini-1.5-flash")
            api_key: API key for Gemini (defaults to GEMINI_API_KEY env var)
        """
        # Determine provider
        if provider is None:
            provider_str = settings.LLM_PROVIDER.lower()
            try:
                provider = LLMProvider(provider_str)
            except ValueError:
                logger.warning(f"Unknown provider '{provider_str}', defaulting to llama_cpp")
                provider = LLMProvider.LLAMA_CPP
        
        self.provider = provider
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.top_p = top_p if top_p is not None else settings.LLM_TOP_P
        self.max_tokens = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS
        self.verbose = verbose if verbose is not None else settings.LLM_VERBOSE
        
        # Initialize provider-specific LLM
        if provider == LLMProvider.LLAMA_CPP:
            self._init_llama_cpp(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                callback_manager=callback_manager
            )
        elif provider == LLMProvider.GEMINI:
            self._init_gemini(
                model_name=model_name,
                api_key=api_key
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _init_llama_cpp(
        self,
        model_path: Optional[str],
        n_ctx: Optional[int],
        n_threads: Optional[int],
        n_gpu_layers: Optional[int],
        callback_manager: Optional[CallbackManager]
    ):
        """Initialize LlamaCpp provider."""
        self.model_path = model_path or settings.LLM_MODEL_PATH
        if not self.model_path:
            raise ValueError("LLM_MODEL_PATH must be provided for llama_cpp provider")
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.n_ctx = n_ctx if n_ctx is not None else settings.LLM_N_CTX
        self.n_threads = n_threads if n_threads is not None else settings.LLM_N_THREADS
        self.n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else settings.LLM_N_GPU_LAYERS
        
        try:
            self.llm = LlamaCpp(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                repeat_penalty=1.3,
                verbose=self.verbose,
                callback_manager=callback_manager
            )
            logger.info(f"LLM client initialized with LlamaCpp: {self.model_path}")
            logger.info(f"Context: {self.n_ctx}, Threads: {self.n_threads}, GPU Layers: {self.n_gpu_layers}")
        except Exception as e:
            logger.error(f"Failed to initialize LlamaCpp: {e}")
            raise
    
    def _init_gemini(
        self,
        model_name: Optional[str],
        api_key: Optional[str]
    ):
        """Initialize Gemini provider."""
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "langchain-google-genai is not installed. "
                "Install with: pip install langchain-google-genai"
            )
        
        self.model_name = model_name or settings.LLM_MODEL
        self.api_key = api_key or settings.GEMINI_API_KEY
        
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY must be provided for gemini provider. "
                "Set it as environment variable or parameter."
            )
        
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=self.temperature,
                top_p=self.top_p,
                max_output_tokens=self.max_tokens,
                convert_system_message_to_human=True
            )
            logger.info(f"LLM client initialized with Gemini: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
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
            
            if self.provider == LLMProvider.GEMINI:
                response = self.llm.invoke(prompt, stop=stop, **kwargs)
            else:
                response = self.llm.invoke(prompt, stop=stop, **kwargs)
            
            result = response.content if hasattr(response, 'content') else str(response)
            logger.debug(f"Generated response length: {len(result)}")
            return result
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
            response = self.generate(
                prompt_str,
                stop=RAG_STOP_SEQUENCES,
                temperature=0.1,
                top_p=0.7,
                max_tokens=256,
            )
            return clean_rag_answer(response)
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
            for chunk in self.llm.stream(prompt, stop=stop, **kwargs):
                if hasattr(chunk, 'content'):
                    yield chunk.content
                else:
                    yield str(chunk)
        except Exception as e:
            logger.error(f"Failed to stream response: {e}")
            raise
    
    @property
    def model(self) -> BaseLanguageModel:
        """Get the underlying LangChain LLM model."""
        return self.llm


def get_llm_client(provider: Optional[LLMProvider] = None) -> LLMClient:
    """
    Get a default LLM client instance using settings.
    
    Args:
        provider: Optional provider override (defaults to LLM_PROVIDER setting)
    
    Returns:
        Configured LLMClient instance
    """
    return LLMClient(provider=provider)
