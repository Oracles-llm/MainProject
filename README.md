## TESTING THE RAG FLOW

# Basic test (no reranking)
python scripts/test_rag_flow.py "What is Python?"

# With reranking
python scripts/test_rag_flow.py "What is RAG?" --rerank

# With BM25 reranking
python scripts/test_rag_flow.py "Explain FastAPI" --rerank --rerank-strategy bm25

# Custom retrieval parameters
python scripts/test_rag_flow.py "What is a vector DB?" --k 15 --rerank-top-k 7 --rerank

# Show help
python scripts/test_rag_flow.py --help

==========================================

# View all points (default limit: 100)
python scripts/view_vector_db.py

# View first 5 points
python scripts/view_vector_db.py --limit 5

# View a specific point by ID
python scripts/view_vector_db.py --point-id <point-id>

# Use a different collection
python scripts/view_vector_db.py --collection my_collection

# Use a different Qdrant path
python scripts/view_vector_db.py --local-path ./custom/path


===============================

# Basic hybrid RAG test
python scripts/test_hybrid_rag.py "What is Python?"

# With custom parameters
python scripts/test_hybrid_rag.py "What is RAG?" --k 5 --rerank-top-k 3

# With reranking enabled
python scripts/test_hybrid_rag.py "Explain FastAPI" --rerank

# Skip comparison mode
python scripts/test_hybrid_rag.py "What is Qdrant?" --no-compare