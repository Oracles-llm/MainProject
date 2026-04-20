## Running The Backend

Start only the API server:

```bash
python main.py
```

## Running The Desktop App

`MainProject` now includes a bundled Java desktop client that replaces the separate `llmUi` repo workflow. This starts the FastAPI backend, waits for it to become healthy, compiles the Java UI with `javac`, and opens the desktop chat window:

```bash
python main.py --desktop-ui
```

Useful variants:

```bash
python main.py --desktop-ui --disable-rag
python main.py --desktop-ui --host 127.0.0.1 --port 8010
```

Requirements:

- Python environment for `MainProject`
- Java/JDK available on `PATH` (`java` and `javac`)

After this integration, you can ignore `llmUi` for normal app usage.

## Local Embedding Model

`MainProject` can run with a local GGUF embedding model through `llama-cpp-python`.

Recommended configuration for `Qwen3-Embedding-0.6B`:

```env
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=Qwen3-Embedding-0.6B
EMBEDDING_MODEL_PATH=./models/Qwen3-Embedding-0.6B.gguf
EMBEDDING_DIMENSION=1024
EMBEDDING_N_CTX=8192
EMBEDDING_N_GPU_LAYERS=0
EMBEDDING_VERBOSE=false
```

Important:

- Put the GGUF file under `MainProject/models/`
- If your collection was created with the old embedding size, re-create or re-ingest it after switching to `1024`
- `Qwen3-Embedding-0.6B` replaces the online Gemini embedding dependency for retrieval and ingestion

If you already have a `768`-dim Qdrant collection from the old embedding model, rebuild it with:

```bash
python scripts/ingest_data.py --folder data --recreate-collection
```

Or if you use the bundled sample docs:

```bash
python scripts/ingest_sample_docs.py
```

## Testing The RAG Flow

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
