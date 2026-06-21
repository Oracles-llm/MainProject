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


NO_CONTEXT_ANSWER = "I dont have information to give answer"


def get_system_prompt() -> str:
    """
    Backwards-compatible default system prompt.

    This project is primarily RAG-first; callers that want a non-RAG prompt should
    use get_chat_system_prompt().
    """
    return get_rag_system_prompt()


def get_rag_system_prompt() -> str:
    """
    System prompt for RAG mode (context documents are available).

    Goal: maximize grounded, accurate answers with minimal hallucination.
    """
    return f"""You are a strict context-only question answering assistant.
The context documents are the ONLY source you may use.

RULES:
1) Answer only from the context documents. Do not use outside knowledge.
2) If the context does not clearly contain the answer, reply exactly:
{NO_CONTEXT_ANSWER}
3) Do not explain missing information. Do not ask follow-up questions.
4) Keep answers concise: two to three sentences maximum. Always finish on a complete sentence.
5) Do not use numbering or bullet points.
6) Do not add citations, preambles, summaries, examples, or extra details unless the user explicitly asks and the context supports them.
7) Use chat history only to understand the user's current question, not as a source of facts.

Before answering, silently check whether every fact in your answer is directly supported by the context."""


def get_chat_system_prompt() -> str:
    """
    System prompt for non-RAG mode (no retrieval / no context documents).

    Goal: be helpful while being explicit about uncertainty and missing project-specific data.
    """
    return """You are a helpful, careful assistant.
No retrieval system is available, and you may not have access to any private/project-specific documents.

RULES (NO-RAG MODE):
1) Answer using general knowledge and the information the user provided in the conversation.
2) If the user asks about project-specific details (code, configs, internal docs) that you cannot see, say so plainly and ask for the exact missing artifacts (file content, error logs, inputs/outputs).
3) If you are uncertain, state your uncertainty and provide the most likely explanation plus a verification step.
4) Be concise and actionable. Prefer concrete steps, commands, and checks.

OUTPUT STYLE:
- Give a direct answer first.
- Then provide short, ordered steps or bullets if needed.
- Avoid filler and avoid inventing details."""


def create_rag_prompt_template(system_prompt: Optional[str] = None) -> ChatPromptTemplate:
    """
    Create a RAG prompt template with chat history, user query, and context.
    
    Args:
        system_prompt: Custom system prompt (defaults to get_system_prompt())
    
    Returns:
        ChatPromptTemplate configured for RAG
    """
    if system_prompt is None:
        system_prompt = get_rag_system_prompt()
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template(
            "Context documents:\n{context}\n\n"
            "Question:\n{user_query}\n\n"
            "Answer using only the context. If the answer is not in the context, reply exactly:\n"
            + NO_CONTEXT_ANSWER
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
        system_prompt = get_chat_system_prompt()
    
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
        system_prompt = get_rag_system_prompt()
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template(
            "Context documents:\n{context}\n\n"
            "Question:\n{user_query}\n\n"
            "Answer using only the context. If the answer is not in the context, reply exactly:\n"
            + NO_CONTEXT_ANSWER
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
        return NO_CONTEXT_ANSWER
    
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
        system_prompt = get_rag_system_prompt()
    
    parts = [f"System: {system_prompt}\n"]
    
    if chat_history:
        history_str = format_chat_history_as_string(chat_history)
        parts.append(history_str)
    
    if context:
        parts.append(f"Context documents:\n{context}\n")
    
    parts.append(f"User: {user_query}\n")
    parts.append("Assistant: Answer concisely in two to three complete sentences.\n")
    
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
        system_prompt = get_chat_system_prompt()
    
    parts = [f"System: {system_prompt}\n"]
    
    if chat_history:
        history_str = format_chat_history_as_string(chat_history)
        parts.append(history_str)
    
    parts.append(f"User: {user_query}\n")
    parts.append("Assistant:")
    
    return "\n".join(parts)

