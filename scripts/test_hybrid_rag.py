"""
Test script for RAG with hybrid search (dense + sparse vectors).
Tests both dense-only and hybrid search to compare results.
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
from app.retrieval import get_retriever, get_sparse_vector_generator

logger = get_logger(__name__)


def compare_search_modes(query: str, k: int = 5):
    """Compare dense-only vs hybrid search results."""
    print("=" * 80)
    print("SEARCH MODE COMPARISON")
    print("=" * 80)
    print(f"Query: {query}")
    print(f"Retrieving top {k} documents")
    print()
    
    settings.QDRANT_MODE = "local"
    settings.QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", "./data/qdrant")
    
    with QdrantDB(mode="local", local_path=settings.QDRANT_LOCAL_PATH) as qdrant:
        from app.embeddings import get_embedder
        
        embedder = get_embedder()
        query_embedding = embedder.embed_query(query)
        
        print("1. DENSE-ONLY SEARCH (Semantic)")
        print("-" * 80)
        dense_results = []
        try:
            dense_results = qdrant.search(
                query_vector=query_embedding,
                limit=k,
                collection_name=settings.QDRANT_COLLECTION_NAME
            )
            print(f"✓ Retrieved {len(dense_results)} documents:")
            dense_ids = []
            for i, result in enumerate(dense_results, 1):
                dense_ids.append(result['id'])
                print(f"  [{i}] Score: {result['score']:.4f} | ID: {result['id'][:8]}...")
                print(f"      Text: {result['payload'].get('text', '')[:100]}...")
                print()
        except Exception as e:
            print(f"❌ Error: {e}")
            print()
        
        print("2. HYBRID SEARCH (BM25 + Semantic)")
        print("-" * 80)
        hybrid_results = []
        try:
            sparse_gen = get_sparse_vector_generator(model_name="Qdrant/bm25")
            if sparse_gen:
                query_sparse = sparse_gen.generate_query_sparse_vector(query)
                print(f"✓ Generated sparse vector with {len(query_sparse)} tokens")
                hybrid_results = qdrant.hybrid_search(
                    query_vector=query_embedding,
                    query_sparse_vector=query_sparse,
                    limit=k,
                    collection_name=settings.QDRANT_COLLECTION_NAME
                )
                print(f"✓ Retrieved {len(hybrid_results)} documents:")
                hybrid_ids = []
                for i, result in enumerate(hybrid_results, 1):
                    hybrid_ids.append(result['id'])
                    print(f"  [{i}] Score: {result['score']:.4f} | ID: {result['id'][:8]}...")
                    print(f"      Text: {result['payload'].get('text', '')[:100]}...")
                    print()
                
                # Verification
                print("3. VERIFICATION")
                print("-" * 80)
                if dense_ids and hybrid_ids:
                    if dense_ids != hybrid_ids:
                        print("✓ VERIFIED: Results differ - Hybrid search is working!")
                        print(f"  - Dense-only top ID: {dense_ids[0][:8]}...")
                        print(f"  - Hybrid search top ID: {hybrid_ids[0][:8]}...")
                        if len(set(dense_ids) & set(hybrid_ids)) > 0:
                            common = len(set(dense_ids) & set(hybrid_ids))
                            print(f"  - Common documents: {common}/{k}")
                    else:
                        print("⚠ Results are identical (might be expected for this query)")
                        print("  - Try a more specific query to see differences")
                else:
                    print("⚠ Could not compare - missing results")
            else:
                print("❌ Sparse vector generator not available")
                print("  - Hybrid search cannot work without sparse vectors")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 80)
    print()


def test_hybrid_rag(
    user_query: str,
    k: int = 5,
    rerank_top_k: int = 3,
    use_reranking: bool = False,
    compare_modes: bool = True
):
    """
    Test RAG with hybrid search.
    
    Args:
        user_query: User's query
        k: Number of documents to retrieve
        rerank_top_k: Number of top documents after reranking
        use_reranking: Whether to use reranking
        compare_modes: Whether to compare dense vs hybrid search
    """
    print("\n" + "=" * 80)
    print(f"HYBRID RAG TEST - Query: '{user_query}'")
    print("=" * 80)
    print()
    
    settings.QDRANT_MODE = "local"
    settings.QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", "./data/qdrant")
    
    if compare_modes:
        compare_search_modes(user_query, k)
    
    # Step 1: Initialize RAG Service with Hybrid Search
    print("STEP 1: INITIALIZING RAG SERVICE WITH HYBRID SEARCH")
    print("-" * 80)
    try:
        rag_service = get_rag_service(
            default_k=k,
            default_rerank_top_k=rerank_top_k,
            use_hybrid_search=True
        )
        print("✓ RAG service initialized with hybrid search enabled")
        print(f"  Default k: {k}, Default rerank_top_k: {rerank_top_k}")
        print()
    except Exception as e:
        print(f"✗ Failed to initialize RAG service: {e}")
        import traceback
        traceback.print_exc()
        return
    print()
    
    # Step 2: Execute RAG Query
    print("STEP 2: EXECUTING HYBRID RAG QUERY")
    print("-" * 80)
    print(f"Query: {user_query}")
    print(f"Retrieve top {k} documents using hybrid search (BM25 + semantic)")
    if use_reranking:
        print(f"Rerank to top {rerank_top_k} documents using {RerankStrategy.BM25.value}")
    else:
        print("Reranking: Disabled")
    print()
    
    try:
        response = rag_service.query(
            query=user_query,
            k=k,
            rerank_top_k=rerank_top_k,
            use_reranking=use_reranking,
            rerank_strategy=RerankStrategy.BM25 if use_reranking else RerankStrategy.NONE
        )
        print("✓ Hybrid RAG query completed successfully")
        print()
    except Exception as e:
        print(f"✗ Failed to execute RAG query: {e}")
        import traceback
        traceback.print_exc()
        return
    print()
    
    # Step 3: Display Retrieved Documents
    print("STEP 3: RETRIEVED DOCUMENTS (Hybrid Search)")
    print("-" * 80)
    retrieved_docs = response.retrieved_documents
    print(f"Retrieved {len(retrieved_docs)} documents using hybrid search:")
    print()
    
    if retrieved_docs:
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"[{i}] Score: {doc.score:.4f}")
            print(f"    ID: {doc.id}")
            print(f"    Text: {doc.text[:150]}...")
            if doc.metadata:
                metadata_str = ", ".join([f"{k}={v}" for k, v in doc.metadata.items() if k != "text"])
                if metadata_str:
                    print(f"    Metadata: {metadata_str}")
            print()
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
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Query: {user_query}")
    print(f"Search Mode: Hybrid (BM25 + Semantic)")
    print(f"Retrieved: {len(response.retrieved_documents)} documents")
    print(f"Used for generation: {len(response.used_documents)} documents")
    print(f"Answer length: {len(response.answer)} characters")
    print(f"Reranking: {'Yes' if use_reranking else 'No'}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RAG with hybrid search")
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="User query (default: example query)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=3,
        help="Number of top documents after reranking (default: 3)"
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable reranking"
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip comparison of dense vs hybrid search"
    )
    
    args = parser.parse_args()
    
    # Example queries
    test_queries = [
        "What is Python?",
        "How does FastAPI work?",
        "What is a vector database?",
        "Explain RAG (Retrieval-Augmented Generation)",
        "What is Qdrant?",
        "How does LangChain work?"
    ]
    
    query = args.query or test_queries[0]
    
    if not args.query:
        print(f"\nNo query provided. Using example query: '{query}'\n")
        print("Usage: python test_hybrid_rag.py 'your query here' [options]")
        print("For help: python test_hybrid_rag.py --help\n")
    
    test_hybrid_rag(
        user_query=query,
        k=args.k,
        rerank_top_k=args.rerank_top_k,
        use_reranking=args.rerank,
        compare_modes=not args.no_compare
    )

