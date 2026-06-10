## Running From Source

Run all commands from the `MainProject` folder:

## Running The Backend

Start only the API server:

```powershell
python main.py
```

The backend runs on `http://127.0.0.1:8000` by default.

## Running The AI Chat Companion

The React/TanStack UI from `ai-chat-companion` is now copied into this project at `MainProject/ai-chat-companion`.

Start the desktop application, not the browser:

```powershell
python main.py --ai-chat-companion
```

This opens the Electron desktop app in its own application window. Electron starts the packaged/static React UI and starts the FastAPI backend in the background.

Start the browser-based web UI instead:

```powershell
python main.py --web-chat-companion
```

Useful variants:

```powershell
python main.py --ai-chat-companion --disable-rag
python main.py --web-chat-companion --host 127.0.0.1 --port 8010
python main.py --web-chat-companion --ui-port 5174
```

You can also run the backend and UI separately during development:

```powershell
python main.py
cd ai-chat-companion
npm install
npm run dev
```

The UI sends chat requests to `http://127.0.0.1:8000/api/v1/chat` by default. To point it at a different backend:

```powershell
$env:VITE_API_BASE_URL="http://127.0.0.1:8010"
npm run dev
```

Chat history is saved locally in the user's browser/Electron storage under the `oracles.chat.history.v1` key, so previous chats remain on the same device.

## Building The Windows Installer

The project now uses Electron for the desktop window, PyInstaller for the Python backend, and electron-builder for the Windows installer.

Install build dependencies once:

```powershell
cd ai-chat-companion
npm install
cd ..
python -m pip install "pyinstaller>=6.0" rank-bm25
```

Build or rebuild the packaged backend after Python/backend changes:

```powershell
cd ai-chat-companion
npm run pack:backend
```

Build the Windows installer after UI-only changes:

```powershell
cd ai-chat-companion
npm run dist:win
```

Build everything after backend and UI changes:

```powershell
cd ai-chat-companion
npm run dist:win:full
```

Generated files:

- Installer: `ai-chat-companion/release/Oracles AI Chat Companion Setup 0.1.0.exe`
- Portable unpacked app: `ai-chat-companion/release/win-unpacked/Oracles AI Chat Companion.exe`

Run the already-built portable app directly:

```powershell
& "C:\Users\User\OneDrive\Desktop\Oracles llm\MainProject\ai-chat-companion\release\win-unpacked\Oracles AI Chat Companion.exe"
```

Run the already-built installer directly:

```powershell
& "C:\Users\User\OneDrive\Desktop\Oracles llm\MainProject\ai-chat-companion\release\Oracles AI Chat Companion Setup 0.1.0.exe"
```

These generated folders are ignored by git. The current backend package is large because the local `llama_cpp` installation includes CUDA runtime DLLs. A CPU-only `llama-cpp-python` environment will produce a smaller installer.

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

```powershell
python scripts/ingest_data.py --folder path/to/your-txt-docs --recreate-collection
```

If you only need to reset the collection schema and do not have any docs yet, the same command will now recreate the collection even when the folder is empty.

Or if you use the bundled sample docs:

```powershell
python scripts/ingest_sample_docs.py
```

## Testing The RAG Flow

Basic test without reranking:

```powershell
python scripts/test_rag_flow.py "What is Python?"
```

Test with reranking:

```powershell
python scripts/test_rag_flow.py "What is RAG?" --rerank
```

Test with BM25 reranking:

```powershell
python scripts/test_rag_flow.py "Explain FastAPI" --rerank --rerank-strategy bm25
```

Use custom retrieval parameters:

```powershell
python scripts/test_rag_flow.py "What is a vector DB?" --k 15 --rerank-top-k 7 --rerank
```

Show help:

```powershell
python scripts/test_rag_flow.py --help
```

## Viewing The Vector DB

View all points, with the default limit of 100:

```powershell
python scripts/view_vector_db.py
```

View the first 5 points:

```powershell
python scripts/view_vector_db.py --limit 5
```

View a specific point by ID:

```powershell
python scripts/view_vector_db.py --point-id <point-id>
```

Use a different collection:

```powershell
python scripts/view_vector_db.py --collection my_collection
```

Use a different Qdrant path:

```powershell
python scripts/view_vector_db.py --local-path ./custom/path
```

## Testing Hybrid RAG

Basic hybrid RAG test:

```powershell
python scripts/test_hybrid_rag.py "What is Python?"
```

Use custom parameters:

```powershell
python scripts/test_hybrid_rag.py "What is RAG?" --k 5 --rerank-top-k 3
```

Enable reranking:

```powershell
python scripts/test_hybrid_rag.py "Explain FastAPI" --rerank
```

Skip comparison mode:

```powershell
python scripts/test_hybrid_rag.py "What is Qdrant?" --no-compare
```
