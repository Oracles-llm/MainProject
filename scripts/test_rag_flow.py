"""
Test script for the complete RAG flow using RAGService.
Tests retrieval, reranking, and generation in one unified flow.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services import get_rag_service, RerankStrategy
from app.core.config import settings
from app.core.logging import get_logger
from app.db.qdrant_client import QdrantDB

logger = get_logger(__name__)


def test_rag_flow(
    user_query: str,
    k: int = 10,
    rerank_top_k: int = 5,
    use_reranking: bool = False,
    rerank_strategy: RerankStrategy = RerankStrategy.NONE
):
    """
    Test the complete RAG flow using RAGService.
    
    Args:
        user_query: User's query
        k: Number of documents to retrieve initially
        rerank_top_k: Number of top documents after reranking
        use_reranking: Whether to use reranking
        rerank_strategy: Reranking strategy (NONE or BM25)
    """
    print("\n" + "=" * 80)
    print(f"RAG FLOW TEST - Query: '{user_query}'")
    print("=" * 80 + "\n")
    
    # Ensure Qdrant uses local mode for this script
    settings.QDRANT_MODE = "local"
    settings.QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", "./data/qdrant")
    
    # Step 1: Initialize RAG Service
    print("STEP 1: INITIALIZING RAG SERVICE")
    print("-" * 80)
    try:
        rag_service = get_rag_service(default_k=k, default_rerank_top_k=rerank_top_k)
        print("✓ RAG service initialized")
        print(f"  Default k: {k}, Default rerank_top_k: {rerank_top_k}")
        print()
    except Exception as e:
        print(f"✗ Failed to initialize RAG service: {e}")
        import traceback
        traceback.print_exc()
        return
    print()
    
    # Step 2: Execute RAG Query
    print("STEP 2: EXECUTING RAG QUERY")
    print("-" * 80)
    print(f"Query: {user_query}")
    print(f"Retrieve top {k} documents")
    if use_reranking:
        print(f"Rerank to top {rerank_top_k} documents using {rerank_strategy.value}")
    else:
        print("Reranking: Disabled")
    print()
    
    try:
        response = rag_service.query(
            query=user_query,
            k=k,
            rerank_top_k=rerank_top_k,
            use_reranking=use_reranking,
            rerank_strategy=rerank_strategy
        )
        print("✓ RAG query completed successfully")
        print()
    except Exception as e:
        print(f"✗ Failed to execute RAG query: {e}")
        import traceback
        traceback.print_exc()
        return
    print()
    
    # Step 3: Display Retrieved Documents
    print("STEP 3: RETRIEVED DOCUMENTS")
    print("-" * 80)
    retrieved_docs = response.retrieved_documents
    print(f"Retrieved {len(retrieved_docs)} documents:")
    print()
    
    if retrieved_docs:
        for i, doc in enumerate(retrieved_docs[:5], 1):
            print(f"[{i}] Score: {doc.score:.4f}")
            print(f"    ID: {doc.id}")
            print(f"    Text: {doc.text[:150]}...")
            if doc.metadata:
                print(f"    Metadata: {doc.metadata}")
            print()
        
        if len(retrieved_docs) > 5:
            print(f"... and {len(retrieved_docs) - 5} more documents\n")
    else:
        print("No documents retrieved.\n")
    print()
    
    # Step 4: Display Reranked Documents (if used)
    if use_reranking and response.reranked_documents:
        print("STEP 4: RERANKED DOCUMENTS")
        print("-" * 80)
        reranked_docs = response.reranked_documents
        print(f"Reranked to {len(reranked_docs)} documents:")
        print()
        
        for i, doc in enumerate(reranked_docs, 1):
            print(f"[{i}] Final Score: {doc.final_score:.4f}")
            print(f"    Original Score: {doc.original_score:.4f}")
            print(f"    Rerank Score: {doc.rerank_score:.4f}")
            print(f"    ID: {doc.id}")
            print(f"    Text: {doc.text[:150]}...")
            print()
        print()
    
    # Step 5: Display Generated Answer
    print("STEP 5: GENERATED ANSWER")
    print("-" * 80)
    answer = response.answer
    print(f"Answer ({len(answer)} characters):")
    print()
    print("=" * 80)
    print(answer)
    print("=" * 80)
    print()
    
    # Step 6: Display Metadata
    print("STEP 6: METADATA")
    print("-" * 80)
    metadata = response.metadata
    print("Query Metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    print()
    
    # Step 7: Display Used Documents
    if response.used_documents:
        print("STEP 7: DOCUMENTS USED FOR GENERATION")
        print("-" * 80)
        used_docs = response.used_documents
        print(f"Used {len(used_docs)} documents for generation:")
        print()
        
        for i, doc in enumerate(used_docs, 1):
            print(f"[{i}] Score: {doc.score:.4f}")
            print(f"    ID: {doc.id}")
            print(f"    Text: {doc.text[:100]}...")
            print()
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Query: {user_query}")
    print(f"Retrieved: {len(response.retrieved_documents)} documents")
    print(f"Used for generation: {len(response.used_documents)} documents")
    print(f"Answer length: {len(response.answer)} characters")
    print(f"Reranking: {'Yes' if use_reranking else 'No'}")
    if use_reranking:
        print(f"Rerank strategy: {rerank_strategy.value}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    # Example queries
    test_queries = [
        "What is Python?",
        "How does FastAPI work?",
        "What is a vector database?",
        "Explain RAG (Retrieval-Augmented Generation)"
    ]
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Usage: python test_rag_flow.py [query] [--rerank] [--k N] [--rerank-top-k N]")
            print()
            print("Arguments:")
            print("  query              User query (default: first example query)")
            print("  --rerank           Enable reranking (default: disabled)")
            print("  --rerank-strategy  Reranking strategy: 'none' or 'bm25' (default: 'none')")
            print("  --k N              Number of documents to retrieve (default: 10)")
            print("  --rerank-top-k N   Number of top documents after reranking (default: 5)")
            print()
            print("Examples:")
            print("  python test_rag_flow.py 'What is Python?'")
            print("  python test_rag_flow.py 'What is RAG?' --rerank")
            print("  python test_rag_flow.py 'Explain FastAPI' --rerank --rerank-strategy bm25")
            print("  python test_rag_flow.py 'What is a vector DB?' --k 15 --rerank-top-k 7")
            sys.exit(0)
        
        # Parse arguments
        query = None
        use_reranking = False
        rerank_strategy = RerankStrategy.NONE
        k = 10
        rerank_top_k = 5
        
        i = 1
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg == "--rerank":
                use_reranking = True
            elif arg == "--rerank-strategy" and i + 1 < len(sys.argv):
                strategy_str = sys.argv[i + 1].lower()
                if strategy_str == "bm25":
                    rerank_strategy = RerankStrategy.BM25
                elif strategy_str == "none":
                    rerank_strategy = RerankStrategy.NONE
                i += 1
            elif arg == "--k" and i + 1 < len(sys.argv):
                k = int(sys.argv[i + 1])
                i += 1
            elif arg == "--rerank-top-k" and i + 1 < len(sys.argv):
                rerank_top_k = int(sys.argv[i + 1])
                i += 1
            elif not arg.startswith("--"):
                query = arg
            i += 1
        
        if query is None:
            query = test_queries[0]
            print(f"\nNo query provided. Using example query: '{query}'\n")
        
        test_rag_flow(
            user_query=query,
            k=k,
            rerank_top_k=rerank_top_k,
            use_reranking=use_reranking,
            rerank_strategy=rerank_strategy
        )
    else:
        # Use default query
        print("\nNo arguments provided. Using example query.")
        print("Usage: python test_rag_flow.py 'your query here' [--rerank]")
        print("For help: python test_rag_flow.py --help\n")
        test_rag_flow(test_queries[0])
