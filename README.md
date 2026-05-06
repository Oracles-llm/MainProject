## Running The Backend

Start only the API server:

```bash
python main.py
```

## Running The AI Chat Companion

The React/TanStack UI from `ai-chat-companion` is now copied into this project at `MainProject/ai-chat-companion`.

Start the backend and UI together:

```bash
python main.py --ai-chat-companion
```

This starts FastAPI, starts the Vite UI server, waits for both to be ready, and opens the UI in your default browser.

Useful variants:

```bash
python main.py --ai-chat-companion --disable-rag
python main.py --ai-chat-companion --host 127.0.0.1 --port 8010
python main.py --ai-chat-companion --ui-port 5174
```

You can also run the backend and UI separately during development:

```bash
python main.py
cd ai-chat-companion
npm install
npm run dev
```

The UI sends chat requests to `http://127.0.0.1:8000/api/v1/chat` by default. To point it at a different backend:

```bash
$env:VITE_API_BASE_URL="http://127.0.0.1:8010"
npm run dev
```

Chat history is saved locally in the user's browser/Electron storage under the `oracles.chat.history.v1` key, so previous chats remain on the same device.

For a final Windows installer, package this UI inside Electron or Tauri and have that desktop shell start the packaged Python backend as a child process. That gives users a normal `.exe` without requiring them to manually open a web browser.

Requirements:

- Python environment for `MainProject`
- Node.js/npm available on `PATH`

After this integration, you can ignore `llmUi` for normal app usage.

## Local Dense Embeddings

`MainProject` can run with local dense embeddings through `fastembed`, which avoids the failing Qwen3 GGUF embedding load path.

Recommended configuration:

```env
EMBEDDING_PROVIDER=fastembed
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIMENSION=384
EMBEDDING_USE_CUDA=false
```

Important:

- `fastembed` downloads and caches the dense model on first use
- If your collection was created with a different embedding size, re-create or re-ingest it after switching providers
- If you want to try ONNX/CUDA later, set `EMBEDDING_USE_CUDA=true`

If you already have a collection from the old embedding model, rebuild it with a folder that actually contains your source `.txt` files:

```bash
python scripts/ingest_data.py --folder path/to/your-txt-docs --recreate-collection
```

If you only need to reset the collection schema and do not have any docs yet, the same command will now recreate the collection even when the folder is empty.

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
