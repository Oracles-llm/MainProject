"""
Main entry point for the FastAPI application and desktop launcher.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent
JAVA_MAIN_CLASS = "com.oracles.desktop.OraclesDesktop"


def compile_desktop_ui() -> Path:
    """Compile the Java desktop UI into the local build directory."""
    source_dir = PROJECT_ROOT / "desktop-ui" / "src"
    build_dir = PROJECT_ROOT / "desktop-ui" / "build" / "classes"
    java_files = sorted(source_dir.rglob("*.java"))

    if not java_files:
        raise FileNotFoundError(f"No Java UI sources found in {source_dir}")

    build_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["javac", "-d", str(build_dir), *[str(path) for path in java_files]],
        cwd=PROJECT_ROOT,
        check=True,
    )
    return build_dir


def wait_for_backend(api_base_url: str, timeout_seconds: int = 180) -> None:
    """Wait until the backend health check responds successfully."""
    deadline = time.time() + timeout_seconds
    health_url = f"{api_base_url}/api/v1/health"
    last_error = "backend did not become ready"

    while time.time() < deadline:
        try:
            with urlopen(health_url, timeout=5) as response:
                if 200 <= response.status < 300:
                    return
                last_error = f"health check returned HTTP {response.status}"
        except URLError as error:
            last_error = str(error)
        except Exception as error:  # pragma: no cover - defensive startup path
            last_error = str(error)
        time.sleep(1)

    raise RuntimeError(f"Timed out waiting for backend at {health_url}: {last_error}")


def start_backend(host: str, port: int, disable_rag: bool) -> subprocess.Popen[str]:
    """Start the FastAPI backend as a subprocess for desktop mode."""
    from app.core.config import settings

    env = os.environ.copy()
    if disable_rag:
        env["DISABLE_RAG"] = "true"

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.api.main:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        settings.LOG_LEVEL.lower(),
    ]

    return subprocess.Popen(command, cwd=PROJECT_ROOT, env=env)


def stop_backend(process: subprocess.Popen[str] | None) -> None:
    """Stop the backend subprocess if it is still running."""
    if process is None or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def launch_desktop_mode(host: str, port: int, disable_rag: bool) -> None:
    """Compile and launch the bundled Java desktop UI with the API server."""
    api_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    api_base_url = f"http://{api_host}:{port}"

    build_dir = compile_desktop_ui()
    backend_process = None

    try:
        backend_process = start_backend(host=host, port=port, disable_rag=disable_rag)
        wait_for_backend(api_base_url)
        subprocess.run(
            ["java", "-cp", str(build_dir), JAVA_MAIN_CLASS, api_base_url],
            cwd=PROJECT_ROOT,
            check=True,
        )
    finally:
        stop_backend(backend_process)


def build_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser."""
    parser = argparse.ArgumentParser(description="Run the LLM API server.")
    parser.add_argument(
        "--disable-rag",
        "--no-rag",
        action="store_true",
        help="Disable the RAG pipeline and send queries directly to the LLM.",
    )
    parser.add_argument(
        "--desktop-ui",
        action="store_true",
        help="Launch the bundled Java desktop UI and the API backend together.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="API host binding. Defaults to 0.0.0.0.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API port. Defaults to 8000.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.disable_rag:
        os.environ["DISABLE_RAG"] = "true"

    from app.core.config import settings

    if args.desktop_ui:
        launch_desktop_mode(host=args.host, port=args.port, disable_rag=args.disable_rag)
    else:
        uvicorn.run(
            "app.api.main:app",
            host=args.host,
            port=args.port,
            reload=settings.DEBUG,
            log_level=settings.LOG_LEVEL.lower(),
        )
