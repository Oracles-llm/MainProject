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