"""
Document preparation API endpoints.

Implements the three endpoints consumed by the frontend DocumentPreparationView:
  POST   /api/v1/documents/prepare          – upload files, start async ingestion job
  GET    /api/v1/documents/status/{job_id}  – poll per-document progress
  POST   /api/v1/documents/cleanup/{job_id} – delete temp files and remove job record

The ingestion pipeline is CPU-heavy (embedding), so it runs in a thread-pool
executor to keep the async event loop responsive.

All Qdrant access goes through the global get_qdrant_db() singleton so only one
local-mode client is ever open at a time.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Dict, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.core.logging import get_logger
from app.ingestion import IngestionConfig, IngestionPipeline

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------

# Job schema stored in _jobs:
#   {
#     "status": "processing" | "completed" | "failed",
#     "tmp_dir": "/tmp/abc123",
#     "documents": {
#       "<filename>": {
#         "status": "processing" | "completed" | "failed",
#         "chunks": 0,
#         "error": None | "..."
#       }
#     }
#   }

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = Lock()

# Single shared executor so we don't spawn unbounded threads.
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ingest")


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


def _process_single_file(
    filename: str,
    file_tmp_dir: str,
    job_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Ingest one file and update the job record when done.

    Runs inside the thread-pool executor — must NOT use asyncio primitives.
    """
    try:
        logger.info("Ingesting file '%s' (job=%s)", filename, job_id)

        config = IngestionConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # Use glob that matches any file extension so txt/md/pdf all work.
            file_glob="**/*",
        )
        pipeline = IngestionPipeline(config=config)
        num_chunks = pipeline.run(file_tmp_dir)

        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["documents"][filename]["status"] = "completed"
                _jobs[job_id]["documents"][filename]["chunks"] = num_chunks

        logger.info(
            "Ingested '%s' -> %d chunks (job=%s)", filename, num_chunks, job_id
        )

    except Exception as exc:
        logger.error(
            "Ingestion failed for '%s' (job=%s): %s", filename, job_id, exc, exc_info=True
        )
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["documents"][filename]["status"] = "failed"
                _jobs[job_id]["documents"][filename]["error"] = str(exc)


def _process_job(
    job_id: str,
    tmp_dir: str,
    file_entries: list[tuple[str, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Process all files for a job sequentially, then mark the job done.

    file_entries: list of (filename, per-file-tmp-dir) tuples.
    Runs inside the thread-pool executor.
    """
    try:
        for filename, file_tmp_dir in file_entries:
            # Check if job was removed (cleanup called mid-run)
            with _jobs_lock:
                if job_id not in _jobs:
                    logger.warning("Job %s was removed mid-run, aborting.", job_id)
                    return

            _process_single_file(filename, file_tmp_dir, job_id, chunk_size, chunk_overlap)

        # Determine overall status
        with _jobs_lock:
            if job_id not in _jobs:
                return
            doc_statuses = [
                d["status"] for d in _jobs[job_id]["documents"].values()
            ]
            if all(s == "completed" for s in doc_statuses):
                _jobs[job_id]["status"] = "completed"
            elif any(s == "completed" for s in doc_statuses):
                # Partial success — still mark completed so the UI finishes polling
                _jobs[job_id]["status"] = "completed"
            else:
                _jobs[job_id]["status"] = "failed"

    except Exception as exc:
        logger.error("Unexpected error in job %s: %s", job_id, exc, exc_info=True)
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["status"] = "failed"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/prepare")
async def prepare_documents(
    files: list[UploadFile] = File(...),
    chunk_size: int = Form(900),
    chunk_overlap: int = Form(120),
    parallel_workers: int = Form(4),  # accepted but not used (sequential for safety)
) -> JSONResponse:
    """
    Accept one or more uploaded files, persist them to a temp directory,
    and kick off an async ingestion job.

    Returns ``{"job_id": "<uuid>"}`` immediately so the client can start polling.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    job_id = str(uuid.uuid4())
    tmp_dir = tempfile.mkdtemp(prefix=f"ingest_{job_id}_")
    logger.info("Created temp dir for job %s: %s", job_id, tmp_dir)

    # Save each uploaded file into its own sub-directory so the pipeline's
    # DirectoryLoader finds exactly one file per IngestionPipeline.run() call.
    file_entries: list[tuple[str, str]] = []
    documents: Dict[str, Any] = {}

    try:
        for upload in files:
            filename = Path(upload.filename or "unknown").name
            file_sub_dir = Path(tmp_dir) / f"file_{uuid.uuid4().hex}"
            file_sub_dir.mkdir(parents=True, exist_ok=True)
            dest = file_sub_dir / filename

            content = await upload.read()
            dest.write_bytes(content)

            file_entries.append((filename, str(file_sub_dir)))
            documents[filename] = {"status": "processing", "chunks": 0, "error": None}

    except Exception as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.error("Failed to save uploaded files for job %s: %s", job_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to save files: {exc}")

    # Register job before spawning thread so /status can respond immediately.
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "processing",
            "tmp_dir": tmp_dir,
            "documents": documents,
        }

    # Run ingestion in the background thread pool — non-blocking.
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        _executor,
        _process_job,
        job_id,
        tmp_dir,
        file_entries,
        chunk_size,
        chunk_overlap,
    )

    logger.info(
        "Job %s started: %d file(s), chunk_size=%d, chunk_overlap=%d",
        job_id, len(files), chunk_size, chunk_overlap,
    )
    return JSONResponse(content={"job_id": job_id})


@router.get("/status/{job_id}")
async def get_job_status(job_id: str) -> JSONResponse:
    """
    Return the current processing status of a job.

    Response shape expected by the frontend::

        {
            "status": "processing" | "completed" | "failed",
            "documents": {
                "<filename>": {
                    "status": "processing" | "completed" | "failed",
                    "chunks": 12,
                    "error": null | "..."
                }
            }
        }
    """
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return JSONResponse(content={
        "status": job["status"],
        "documents": {
            name: {
                "status": doc["status"],
                "chunks": doc["chunks"],
                "error": doc.get("error"),
            }
            for name, doc in job["documents"].items()
        },
    })


@router.post("/cleanup/{job_id}")
async def cleanup_job(job_id: str) -> JSONResponse:
    """
    Remove the temporary files created for a job and delete its record.

    Safe to call whether the job succeeded, failed, or is still processing
    (though calling it mid-processing will cause in-flight threads to abort
    gracefully on their next lock check).
    """
    with _jobs_lock:
        job = _jobs.pop(job_id, None)

    if job is None:
        # Idempotent — already cleaned up or never existed.
        return JSONResponse(content={"detail": "Job not found or already cleaned up."})

    tmp_dir = job.get("tmp_dir")
    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("Cleaned up temp dir for job %s: %s", job_id, tmp_dir)

    return JSONResponse(content={"detail": "Cleanup complete."})
