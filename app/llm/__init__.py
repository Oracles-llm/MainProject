"""
LLM client for text generation with multi-provider support.
"""

from app.llm.client import LLMClient, get_llm_client, LLMProvider
from app.llm.prompts import (
    get_system_prompt,
    get_rag_system_prompt,
    get_chat_system_prompt,
    create_rag_prompt_template,
    create_chat_prompt_template,
    create_context_only_prompt_template,
    format_chat_history,
    format_chat_history_as_string,
    format_context_documents,
    build_rag_prompt_string,
    build_chat_prompt_string
)

__all__ = [
    "LLMClient",
    "get_llm_client",
    "LLMProvider",
    "get_system_prompt",
    "get_rag_system_prompt",
    "get_chat_system_prompt",
    "create_rag_prompt_template",
    "create_chat_prompt_template",
    "create_context_only_prompt_template",
    "format_chat_history",
    "format_chat_history_as_string",
    "format_context_documents",
    "build_rag_prompt_string",
    "build_chat_prompt_string"
]
