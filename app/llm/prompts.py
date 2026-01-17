"""
Prompt templates for LLM interactions.
Supports chat history, user queries, and context from relevant documents.
"""

from typing import List, Tuple, Optional
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)


def get_system_prompt() -> str:
    """Get the default system prompt."""
    return """You are a helpful AI assistant. Use the provided context documents to answer the user's question accurately and concisely.

IMPORTANT INSTRUCTIONS:
- Answer ONLY the specific question asked by the user
- Do NOT generate additional questions or answers to questions not asked
- Do NOT create hypothetical Q&A pairs
- Keep your response focused and direct
- If the context doesn't contain enough information to answer the question, say so honestly
- Stop after answering the user's question - do not continue with additional content"""


def create_rag_prompt_template(system_prompt: Optional[str] = None) -> ChatPromptTemplate:
    """
    Create a RAG prompt template with chat history, user query, and context.
    
    Args:
        system_prompt: Custom system prompt (defaults to get_system_prompt())
    
    Returns:
        ChatPromptTemplate configured for RAG
    """
    if system_prompt is None:
        system_prompt = get_system_prompt()
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template(
            "Context documents:\n{context}\n\n"
            "Based on the context above, please answer the following question:\n{user_query}"
        )
    ])
    
    return prompt


def create_chat_prompt_template(system_prompt: Optional[str] = None) -> ChatPromptTemplate:
    """
    Create a chat prompt template with chat history and user query (no context).
    
    Args:
        system_prompt: Custom system prompt
    
    Returns:
        ChatPromptTemplate configured for general chat
    """
    if system_prompt is None:
        system_prompt = "You are a helpful AI assistant."
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{user_query}")
    ])
    
    return prompt


def create_context_only_prompt_template(system_prompt: Optional[str] = None) -> ChatPromptTemplate:
    """
    Create a prompt template with context but no chat history.
    
    Args:
        system_prompt: Custom system prompt (defaults to get_system_prompt())
    
    Returns:
        ChatPromptTemplate with context support
    """
    if system_prompt is None:
        system_prompt = get_system_prompt()
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template(
            "Context documents:\n{context}\n\n"
            "Based on the context above, please answer the following question:\n{user_query}"
        )
    ])
    
    return prompt


def format_chat_history(history: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Format chat history for LangChain prompt templates.
    
    Args:
        history: List of (role, message) tuples where role is 'human' or 'ai'
    
    Returns:
        Formatted chat history compatible with MessagesPlaceholder
    """
    formatted = []
    for role, message in history:
        formatted.append((role, message))
    return formatted


def format_chat_history_as_string(history: List[Tuple[str, str]]) -> str:
    """
    Format chat history as a string for LLM prompts.
    
    Args:
        history: List of (role, message) tuples where role is 'human' or 'ai'
    
    Returns:
        Formatted chat history as a string
    """
    if not history:
        return ""
    
    formatted_parts = []
    for role, message in history:
        if role.lower() == "human" or role.lower() == "user":
            formatted_parts.append(f"User: {message}")
        elif role.lower() == "ai" or role.lower() == "assistant":
            formatted_parts.append(f"Assistant: {message}")
        else:
            formatted_parts.append(f"{role.capitalize()}: {message}")
    
    return "\n".join(formatted_parts) + "\n"


def format_context_documents(documents: List[str]) -> str:
    """
    Format a list of document texts into a context string.
    
    Args:
        documents: List of document text strings
    
    Returns:
        Formatted context string
    """
    if not documents:
        return "No relevant documents found."
    
    context_parts = []
    for i, doc in enumerate(documents, 1):
        context_parts.append(f"[Document {i}]\n{doc}\n")
    
    return "\n".join(context_parts)


def build_rag_prompt_string(
    user_query: str,
    context: Optional[str] = None,
    chat_history: Optional[List[Tuple[str, str]]] = None,
    system_prompt: Optional[str] = None
) -> str:
    """
    Build a complete RAG prompt string for LlamaCpp.
    
    Args:
        user_query: User's query/question
        context: Formatted context string from documents
        chat_history: List of (role, message) tuples
        system_prompt: System prompt (defaults to get_system_prompt())
    
    Returns:
        Complete prompt string
    """
    if system_prompt is None:
        system_prompt = get_system_prompt()
    
    parts = [f"System: {system_prompt}\n"]
    
    if chat_history:
        history_str = format_chat_history_as_string(chat_history)
        parts.append(history_str)
    
    if context:
        parts.append(f"Context documents:\n{context}\n")
    
    parts.append(f"User: {user_query}\n")
    parts.append("Assistant: ")
    
    return "\n".join(parts)


def build_chat_prompt_string(
    user_query: str,
    chat_history: Optional[List[Tuple[str, str]]] = None,
    system_prompt: Optional[str] = None
) -> str:
    """
    Build a complete chat prompt string for LlamaCpp.
    
    Args:
        user_query: User's query/question
        chat_history: List of (role, message) tuples
        system_prompt: System prompt
    
    Returns:
        Complete prompt string
    """
    if system_prompt is None:
        system_prompt = "You are a helpful AI assistant."
    
    parts = [f"System: {system_prompt}\n"]
    
    if chat_history:
        history_str = format_chat_history_as_string(chat_history)
        parts.append(history_str)
    
    parts.append(f"User: {user_query}\n")
    parts.append("Assistant:")
    
    return "\n".join(parts)

