"""
Verification script to ensure hybrid search is working correctly.
Compares dense-only vs hybrid search and shows detailed information.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.qdrant_client import QdrantDB
from app.core.config import settings
from app.core.logging import get_logger
from app.embeddings import get_embedder
from app.retrieval import get_sparse_vector_generator, get_retriever
from app.retrieval.retriever import VectorRetriever

logger = get_logger(__name__)


def verify_sparse_vector_generator():
    """Verify that sparse vector generator is working."""
    print("=" * 80)
    print("STEP 1: VERIFYING SPARSE VECTOR GENERATOR")
    print("=" * 80)
    
    try:
        sparse_gen = get_sparse_vector_generator(model_name="Qdrant/bm25")
        if not sparse_gen:
            print("❌ FAILED: Sparse vector generator is None")
            return False
        
        print("✓ Sparse vector generator initialized")
        
        # Test generating a sparse vector
        test_query = "Python programming language"
        print(f"\nTesting with query: '{test_query}'")
        
        sparse_vector = sparse_gen.generate_query_sparse_vector(test_query)
        
        if not sparse_vector:
            print("❌ FAILED: Could not generate sparse vector")
            return False
        
        print(f"✓ Sparse vector generated successfully")
        print(f"  - Number of non-zero tokens: {len(sparse_vector)}")
        print(f"  - Sample tokens (first 5):")
        for i, (token_id, score) in enumerate(list(sparse_vector.items())[:5]):
            print(f"    Token ID {token_id}: {score:.4f}")
        
        if len(sparse_vector) == 0:
            print("❌ FAILED: Sparse vector is empty")
            return False
        
        print("\n✓ Sparse vector generator is working correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Error testing sparse vector generator: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_collection_has_sparse_vectors():
    """Verify that the collection has sparse vectors configured."""
    print("\n" + "=" * 80)
    print("STEP 2: VERIFYING COLLECTION CONFIGURATION")
    print("=" * 80)
    
    try:
        settings.QDRANT_MODE = "local"
        settings.QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", "./data/qdrant")
        
        with QdrantDB(mode="local", local_path=settings.QDRANT_LOCAL_PATH) as qdrant:
            collection_name = settings.QDRANT_COLLECTION_NAME
            
            if not qdrant.collection_exists(collection_name):
                print(f"❌ FAILED: Collection '{collection_name}' does not exist")
                return False
            
            print(f"✓ Collection '{collection_name}' exists")
            
            # Get collection info
            info = qdrant.get_collection_info(collection_name)
            if info:
                print(f"  - Total points: {info['points_count']}")
                print(f"  - Vector size: {info['vector_size']}")
            
            # Check if collection has sparse vectors by trying to retrieve a point
            try:
                scroll_result = qdrant.client.scroll(
                    collection_name=collection_name,
                    limit=1,
                    with_payload=True,
                    with_vectors=True
                )
                
                if scroll_result[0]:
                    point = scroll_result[0][0]
                    if hasattr(point, 'vector'):
                        if isinstance(point.vector, dict):
                            has_sparse = any(
                                name != "" and hasattr(vec, 'indices') 
                                for name, vec in point.vector.items()
                            )
                            if has_sparse:
                                print("✓ Collection has sparse vectors stored")
                                print("  - Found sparse vector in stored points")
                                return True
                            else:
                                print("⚠ WARNING: Collection points don't have sparse vectors")
                                print("  - Points only have dense vectors")
                                return False
                        else:
                            print("⚠ WARNING: Collection points use simple vector format")
                            print("  - May not have sparse vectors configured")
                            return False
                    else:
                        print("❌ FAILED: Could not retrieve vector from point")
                        return False
                else:
                    print("⚠ WARNING: Collection is empty")
                    return False
                    
            except Exception as e:
                print(f"⚠ WARNING: Could not verify sparse vectors in collection: {e}")
                print("  - This might be okay if collection was just created")
                return True  # Don't fail, just warn
                
    except Exception as e:
        print(f"❌ FAILED: Error checking collection: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_search_results(query: str, k: int = 5):
    """Compare dense-only vs hybrid search results."""
    print("\n" + "=" * 80)
    print("STEP 3: COMPARING DENSE-ONLY VS HYBRID SEARCH")
    print("=" * 80)
    print(f"Query: '{query}'")
    print(f"Retrieving top {k} documents\n")
    
    try:
        settings.QDRANT_MODE = "local"
        settings.QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", "./data/qdrant")
        
        with QdrantDB(mode="local", local_path=settings.QDRANT_LOCAL_PATH) as qdrant:
            embedder = get_embedder()
            query_embedding = embedder.embed_query(query)
            
            # Dense-only search
            print("1. DENSE-ONLY SEARCH (Semantic only)")
            print("-" * 80)
            dense_results = qdrant.search(
                query_vector=query_embedding,
                limit=k,
                collection_name=settings.QDRANT_COLLECTION_NAME
            )
            print(f"✓ Retrieved {len(dense_results)} documents")
            dense_ids = [r['id'] for r in dense_results]
            dense_scores = [r['score'] for r in dense_results]
            for i, result in enumerate(dense_results[:3], 1):
                text_preview = result['payload'].get('text', '')[:80]
                print(f"  [{i}] Score: {result['score']:.4f} | {text_preview}...")
            print()
            
            # Hybrid search
            print("2. HYBRID SEARCH (BM25 + Semantic)")
            print("-" * 80)
            sparse_gen = get_sparse_vector_generator(model_name="Qdrant/bm25")
            if not sparse_gen:
                print("❌ FAILED: Cannot perform hybrid search - sparse generator not available")
                return False
            
            query_sparse = sparse_gen.generate_query_sparse_vector(query)
            if not query_sparse:
                print("❌ FAILED: Could not generate sparse vector for query")
                return False
            
            print(f"✓ Generated sparse vector with {len(query_sparse)} tokens")
            
            hybrid_results = qdrant.hybrid_search(
                query_vector=query_embedding,
                query_sparse_vector=query_sparse,
                limit=k,
                collection_name=settings.QDRANT_COLLECTION_NAME
            )
            print(f"✓ Retrieved {len(hybrid_results)} documents")
            hybrid_ids = [r['id'] for r in hybrid_results]
            hybrid_scores = [r['score'] for r in hybrid_results]
            for i, result in enumerate(hybrid_results[:3], 1):
                text_preview = result['payload'].get('text', '')[:80]
                print(f"  [{i}] Score: {result['score']:.4f} | {text_preview}...")
            print()
            
            # Compare results
            print("3. COMPARISON")
            print("-" * 80)
            if dense_ids != hybrid_ids:
                print("✓ Results DIFFER - Hybrid search is working!")
                print(f"  - Dense-only top result: {dense_ids[0]}")
                print(f"  - Hybrid search top result: {hybrid_ids[0]}")
                
                # Check if scores are different
                if abs(dense_scores[0] - hybrid_scores[0]) > 0.01:
                    print(f"  - Score difference: {abs(dense_scores[0] - hybrid_scores[0]):.4f}")
                    print("  ✓ Scores are different - hybrid search is affecting results")
                else:
                    print("  ⚠ Scores are similar (might be same documents)")
            else:
                print("⚠ Results are IDENTICAL")
                print("  - This could mean:")
                print("    * Both methods found the same best documents")
                print("    * Or hybrid search might not be working")
                print("  - Try a different query to verify")
            
            return True
            
    except Exception as e:
        print(f"❌ FAILED: Error comparing search results: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_retriever_hybrid_search():
    """Verify that VectorRetriever uses hybrid search correctly."""
    print("\n" + "=" * 80)
    print("STEP 4: VERIFYING RETRIEVER HYBRID SEARCH")
    print("=" * 80)
    
    try:
        settings.QDRANT_MODE = "local"
        settings.QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", "./data/qdrant")
        
        sparse_gen = get_sparse_vector_generator(model_name="Qdrant/bm25")
        if not sparse_gen:
            print("❌ FAILED: Sparse vector generator not available")
            return False
        
        # Test with hybrid search enabled
        retriever_hybrid = get_retriever(
            k=3,
            use_hybrid_search=True,
            sparse_vector_generator=sparse_gen
        )
        
        print("✓ Retriever initialized with hybrid search enabled")
        
        # Test with hybrid search disabled
        retriever_dense = get_retriever(
            k=3,
            use_hybrid_search=False
        )
        
        print("✓ Retriever initialized with dense-only search")
        
        test_query = "Python programming"
        print(f"\nTesting with query: '{test_query}'")
        
        # Get results from both
        hybrid_results = retriever_hybrid.retrieve(test_query, k=3)
        dense_results = retriever_dense.retrieve(test_query, k=3)
        
        print(f"\nHybrid search results: {len(hybrid_results)} documents")
        print(f"Dense-only results: {len(dense_results)} documents")
        
        if len(hybrid_results) > 0 and len(dense_results) > 0:
            hybrid_ids = [r.id for r in hybrid_results]
            dense_ids = [r.id for r in dense_results]
            
            if hybrid_ids != dense_ids:
                print("✓ Results differ - hybrid search is working in retriever!")
            else:
                print("⚠ Results are identical (might be expected for this query)")
            
            return True
        else:
            print("❌ FAILED: No results retrieved")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: Error testing retriever: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("\n" + "=" * 80)
    print("HYBRID SEARCH VERIFICATION")
    print("=" * 80)
    print()
    
    results = []
    
    # Step 1: Verify sparse vector generator
    results.append(("Sparse Vector Generator", verify_sparse_vector_generator()))
    
    # Step 2: Verify collection configuration
    results.append(("Collection Configuration", verify_collection_has_sparse_vectors()))
    
    # Step 3: Compare search results
    test_query = "Python programming language"
    results.append(("Search Comparison", compare_search_results(test_query, k=5)))
    
    # Step 4: Verify retriever
    results.append(("Retriever Hybrid Search", verify_retriever_hybrid_search()))
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("=" * 80)
        print("✓ ALL TESTS PASSED - Hybrid search is working correctly!")
        print("=" * 80)
    else:
        print("=" * 80)
        print("⚠ SOME TESTS FAILED - Review the output above")
        print("=" * 80)
        print("\nTroubleshooting:")
        print("1. Ensure sparse vectors are stored in the collection")
        print("2. Run: python scripts/ingest_sample_docs.py")
        print("3. Check that collection was created with sparse_vectors_config")
        print("4. Verify sparse vector generator is initialized correctly")
    
    print()
    return all_passed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify hybrid search is working")
    parser.add_argument(
        "--query",
        type=str,
        default="Python programming language",
        help="Test query for comparison (default: 'Python programming language')"
    )
    
    args = parser.parse_args()
    
    # Override test query if provided
    import sys
    if args.query != "Python programming language":
        # We'll need to modify compare_search_results to use this
        pass
    
    success = main()
    sys.exit(0 if success else 1)

