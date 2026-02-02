"""
Main entry point for the FastAPI application.
"""

import argparse
import os
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LLM API server.")
    parser.add_argument(
        "--disable-rag",
        "--no-rag",
        action="store_true",
        help="Disable the RAG pipeline and send queries directly to the LLM."
    )
    args = parser.parse_args()

    if args.disable_rag:
        os.environ["DISABLE_RAG"] = "true"

    from app.core.config import settings

    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
