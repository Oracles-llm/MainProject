"""
Main entry point for the FastAPI application and React UI launcher.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent
AI_CHAT_COMPANION_DIR = PROJECT_ROOT / "ai-chat-companion"


def find_available_port(host: str, preferred_port: int) -> int:
    """Find an available TCP port, starting with the preferred port."""
    bind_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host

    for port in range(preferred_port, preferred_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((bind_host, port))
            except OSError:
                continue
            return port

    raise RuntimeError(f"No available UI port found from {preferred_port} to {preferred_port + 99}")


def wait_for_url(url: str, timeout_seconds: int = 180) -> None:
    """Wait until a URL responds successfully."""
    deadline = time.time() + timeout_seconds
    last_error = "URL did not become ready"

    while time.time() < deadline:
        try:
            with urlopen(url, timeout=5) as response:
                if 200 <= response.status < 300:
                    return
                last_error = f"request returned HTTP {response.status}"
        except URLError as error:
            last_error = str(error)
        except Exception as error:  # pragma: no cover - defensive startup path
            last_error = str(error)
        time.sleep(1)

    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def wait_for_backend(api_base_url: str, timeout_seconds: int = 180) -> None:
    """Wait until the backend health check responds successfully."""
    wait_for_url(f"{api_base_url}/api/v1/health", timeout_seconds=timeout_seconds)


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


def stop_process(process: subprocess.Popen[str] | None) -> None:
    """Stop a subprocess if it is still running."""
    if process is None or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def ensure_ai_chat_companion_dependencies() -> None:
    """Install UI dependencies on first run if node_modules is missing."""
    if not AI_CHAT_COMPANION_DIR.is_dir():
        raise FileNotFoundError(f"AI chat UI directory not found: {AI_CHAT_COMPANION_DIR}")

    if (AI_CHAT_COMPANION_DIR / "node_modules").is_dir():
        return

    npm_executable = get_npm_executable()
    print("Installing ai-chat-companion dependencies with npm install...")
    subprocess.run([npm_executable, "install"], cwd=AI_CHAT_COMPANION_DIR, check=True)


def get_npm_executable() -> str:
    """Return the npm executable path for the current platform."""
    executable = shutil.which("npm.cmd" if os.name == "nt" else "npm") or shutil.which("npm")
    if executable is None:
        raise RuntimeError("npm was not found on PATH. Install Node.js before launching the UI.")
    return executable


def start_ai_chat_companion(
    api_base_url: str,
    ui_host: str,
    ui_port: int,
) -> subprocess.Popen[str]:
    """Start the React UI dev server."""
    npm_executable = get_npm_executable()
    env = os.environ.copy()
    env["VITE_API_BASE_URL"] = api_base_url

    command = [
        npm_executable,
        "run",
        "dev",
        "--",
        "--host",
        ui_host,
        "--port",
        str(ui_port),
        "--strictPort",
    ]

    return subprocess.Popen(command, cwd=AI_CHAT_COMPANION_DIR, env=env)


def launch_ai_chat_companion(
    host: str,
    port: int,
    disable_rag: bool,
    ui_host: str,
    ui_port: int,
) -> None:
    """Launch the FastAPI backend and React UI together."""
    api_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    api_base_url = f"http://{api_host}:{port}"
    resolved_ui_port = find_available_port(ui_host, ui_port)
    browser_ui_host = "127.0.0.1" if ui_host in {"0.0.0.0", "::"} else ui_host
    ui_base_url = f"http://{browser_ui_host}:{resolved_ui_port}"

    backend_process = None
    ui_process = None

    try:
        ensure_ai_chat_companion_dependencies()
        backend_process = start_backend(host=host, port=port, disable_rag=disable_rag)
        wait_for_backend(api_base_url)
        ui_process = start_ai_chat_companion(
            api_base_url=api_base_url,
            ui_host=ui_host,
            ui_port=resolved_ui_port,
        )
        wait_for_url(ui_base_url, timeout_seconds=60)
        print(f"Opening AI Chat Companion at {ui_base_url}")
        webbrowser.open(ui_base_url)
        ui_process.wait()
    except KeyboardInterrupt:
        print("Stopping AI Chat Companion...")
    finally:
        stop_process(ui_process)
        stop_process(backend_process)


def launch_ai_chat_companion_desktop(disable_rag: bool) -> None:
    """Launch the Electron desktop UI, which starts the backend itself."""
    ensure_ai_chat_companion_dependencies()
    npm_executable = get_npm_executable()
    env = os.environ.copy()

    if disable_rag:
        env["DISABLE_RAG"] = "true"

    subprocess.run([npm_executable, "run", "desktop"], cwd=AI_CHAT_COMPANION_DIR, env=env, check=True)


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
        "--ai-chat-companion",
        action="store_true",
        help="Launch the Electron AI chat desktop app and the API backend together.",
    )
    parser.add_argument(
        "--web-chat-companion",
        action="store_true",
        help="Launch the React AI chat UI in the default browser with the API backend.",
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
    parser.add_argument(
        "--ui-host",
        default="127.0.0.1",
        help="AI chat UI host binding. Defaults to 127.0.0.1.",
    )
    parser.add_argument(
        "--ui-port",
        type=int,
        default=5173,
        help="AI chat UI port. Defaults to 5173, or the next available port.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.disable_rag:
        os.environ["DISABLE_RAG"] = "true"

    from app.core.config import logger, settings

    if args.ai_chat_companion:
        launch_ai_chat_companion_desktop(disable_rag=args.disable_rag)
    elif args.web_chat_companion:
        launch_ai_chat_companion(
            host=args.host,
            port=args.port,
            disable_rag=args.disable_rag,
            ui_host=args.ui_host,
            ui_port=args.ui_port,
        )
    else:
        logger.info(
            "Launching API server on %s:%s with debug=%s",
            args.host,
            args.port,
            settings.DEBUG,
        )
        if settings.DEBUG:
            uvicorn.run(
                "app.api.main:app",
                host=args.host,
                port=args.port,
                reload=True,
                log_level=settings.LOG_LEVEL.lower(),
            )
        else:
            logger.info("Importing FastAPI app before Uvicorn startup")
            from app.api.main import app as fastapi_app

            logger.info("FastAPI app imported successfully")
            uvicorn.run(
                fastapi_app,
                host=args.host,
                port=args.port,
                reload=False,
                log_level=settings.LOG_LEVEL.lower(),
            )
