"""
Script to ingest sample documents into Qdrant for testing.
Uses local persistent storage.
"""

import sys
from pathlib import Path
from datetime import datetime
import uuid

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.qdrant_client import QdrantDB
from app.db.models import Document, VectorPoint
from app.embeddings import get_embedder
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def get_sample_documents():
    """Get sample documents for testing."""
    return [
        {
            "text": "Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
            "source": "python_intro.txt",
            "metadata": {"category": "programming", "topic": "python_basics"}
        },
        {
            "text": "FastAPI is a modern, fast web framework for building APIs with Python based on standard Python type hints. It is built on top of Starlette and Pydantic, making it one of the fastest Python frameworks available. FastAPI provides automatic API documentation, data validation, and async support out of the box.",
            "source": "fastapi_intro.txt",
            "metadata": {"category": "programming", "topic": "web_frameworks"}
        },
        {
            "text": "Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently. They are essential for semantic search, recommendation systems, and machine learning applications. Popular vector databases include Qdrant, Pinecone, Weaviate, and Milvus.",
            "source": "vector_db.txt",
            "metadata": {"category": "databases", "topic": "vector_databases"}
        },
        {
            "text": "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with language generation. RAG systems first retrieve relevant documents from a knowledge base, then use those documents as context for generating accurate responses. This approach helps reduce hallucinations and improves answer quality.",
            "source": "rag_concept.txt",
            "metadata": {"category": "ai", "topic": "rag"}
        },
        {
            "text": "Qdrant is an open-source vector database written in Rust, designed for production use. It provides fast similarity search, filtering capabilities, and supports various distance metrics like cosine, Euclidean, and dot product. Qdrant can be deployed locally or in the cloud and offers Python, JavaScript, and other language clients.",
            "source": "qdrant_info.txt",
            "metadata": {"category": "databases", "topic": "qdrant"}
        },
        {
            "text": "LangChain is a framework for developing applications powered by language models. It provides abstractions for chains, agents, memory, and retrieval, making it easier to build complex LLM applications. LangChain supports integration with various vector stores, LLM providers, and tools.",
            "source": "langchain_info.txt",
            "metadata": {"category": "ai", "topic": "langchain"}
        },
        {
            "text": "Embeddings are numerical representations of text that capture semantic meaning. They convert text into dense vectors in a high-dimensional space where similar texts are close together. Embeddings are used for semantic search, clustering, classification, and as inputs to machine learning models.",
            "source": "embeddings_concept.txt",
            "metadata": {"category": "ai", "topic": "embeddings"}
        },
        {
            "text": "Llama.cpp is a C++ implementation of the LLaMA model inference. It enables running large language models efficiently on consumer hardware, including CPUs and GPUs. Llama.cpp supports quantization to reduce model size and memory requirements while maintaining reasonable performance.",
            "source": "llama_cpp.txt",
            "metadata": {"category": "ai", "topic": "llm_inference"}
        }
    ]


def ingest_documents():
    """Ingest sample documents into Qdrant."""
    try:
        # Initialize embedder
        logger.info("Initializing embedder...")
        embedder = get_embedder()
        
        # Initialize Qdrant with local path
        logger.info("Initializing Qdrant...")
        local_path = "./data/qdrant"
        qdrant = QdrantDB(
            mode="local",
            local_path=local_path
        )
        
        # Ensure collection exists
        collection_name = settings.QDRANT_COLLECTION_NAME
        logger.info(f"Ensuring collection '{collection_name}' exists...")
        qdrant.ensure_collection(
            collection_name=collection_name,
            vector_size=settings.EMBEDDING_DIMENSION
        )
        
        # Get sample documents
        sample_docs = get_sample_documents()
        logger.info(f"Processing {len(sample_docs)} documents...")
        
        # Prepare documents for embedding
        doc_texts = [doc["text"] for doc in sample_docs]
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embedding_result = embedder.embed_documents(doc_texts)
        
        if len(embedding_result.embeddings) != len(sample_docs):
            raise ValueError(f"Mismatch: {len(embedding_result.embeddings)} embeddings for {len(sample_docs)} documents")
        
        # Create vector points
        points = []
        for i, (doc_data, embedding) in enumerate(zip(sample_docs, embedding_result.embeddings)):
            doc_id = str(uuid.uuid4())
            
            payload = {
                "text": doc_data["text"],
                "source": doc_data["source"],
                "chunk_index": i,
                "created_at": datetime.now().isoformat(),
                **doc_data["metadata"]
            }
            
            point = VectorPoint(
                id=doc_id,
                vector=embedding,
                payload=payload
            )
            points.append(point.to_point_struct())
        
        # Upsert points
        logger.info(f"Inserting {len(points)} documents into Qdrant...")
        qdrant.upsert_points(points, collection_name=collection_name)
        
        logger.info(f"Successfully ingested {len(points)} documents into Qdrant!")
        logger.info(f"Collection: {collection_name}")
        logger.info(f"Storage path: {qdrant.local_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to ingest documents: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 80)
    print("SAMPLE DOCUMENT INGESTION")
    print("=" * 80)
    print()
    
    success = ingest_documents()
    
    if success:
        print("\n" + "=" * 80)
        print("INGESTION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("INGESTION FAILED!")
        print("=" * 80)
        sys.exit(1)

