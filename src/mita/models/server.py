"""Ollama server lifecycle management — start, stop, health check."""

from __future__ import annotations

import atexit
import shutil
import subprocess
import time

from rich.console import Console

from mita.models.ollama_client import OllamaClient

# Singleton tracking: did *we* start the server?
_managed_process: subprocess.Popen[bytes] | None = None


def find_ollama_binary() -> str | None:
    """Locate the ollama binary on the system."""
    return shutil.which("ollama")


def is_server_running(host: str = "http://localhost:11434", timeout: int = 5) -> bool:
    """Check if an Ollama server is reachable."""
    client = OllamaClient(host=host, timeout=timeout)
    return client.is_running()


def start_server(
    host: str = "http://localhost:11434",
    timeout: int = 30,
    console: Console | None = None,
) -> bool:
    """Start the Ollama server in the background if not already running.

    Returns True if the server is running (either already was or we started it).
    Returns False if we failed to start it.
    """
    global _managed_process  # noqa: PLW0603

    # Already running? Nothing to do.
    if is_server_running(host, timeout=5):
        return True

    binary = find_ollama_binary()
    if binary is None:
        if console:
            console.print(
                "[red]Ollama binary not found.[/red]\n"
                "Install Ollama from [bold]https://ollama.com[/bold]"
            )
        return False

    if console:
        console.print("[dim]Starting Ollama server...[/dim]")

    # Parse host/port for OLLAMA_HOST env var
    env_host = host.replace("http://", "").replace("https://", "")

    try:
        _managed_process = subprocess.Popen(
            [binary, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={"OLLAMA_HOST": env_host, "PATH": _get_path()},
        )
    except OSError as e:
        if console:
            console.print(f"[red]Failed to start Ollama: {e}[/red]")
        return False

    # Register cleanup so we stop it on exit
    atexit.register(stop_server)

    # Wait for the server to become healthy
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _managed_process.poll() is not None:
            # Process exited unexpectedly
            if console:
                console.print("[red]Ollama server exited unexpectedly.[/red]")
            _managed_process = None
            return False
        if is_server_running(host, timeout=2):
            if console:
                console.print("[green]Ollama server started.[/green]")
            return True
        time.sleep(0.5)

    if console:
        console.print("[red]Ollama server did not become ready in time.[/red]")
    stop_server()
    return False


def stop_server(console: Console | None = None) -> bool:
    """Stop the Ollama server if we started it.

    Returns True if we stopped it, False if there was nothing to stop.
    """
    global _managed_process  # noqa: PLW0603

    if _managed_process is None:
        return False

    if _managed_process.poll() is None:
        # Still running — terminate gracefully
        _managed_process.terminate()
        try:
            _managed_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _managed_process.kill()
            _managed_process.wait(timeout=5)

    if console:
        console.print("[dim]Ollama server stopped.[/dim]")

    _managed_process = None
    return True


def is_managed() -> bool:
    """Return True if the current Ollama server was started by Mita."""
    return _managed_process is not None and _managed_process.poll() is None


def ensure_server(
    host: str = "http://localhost:11434",
    auto_manage: bool = True,
    console: Console | None = None,
) -> bool:
    """Ensure Ollama is available — start it if auto_manage is enabled.

    This is the main entry point for the agent loop and CLI commands.
    """
    if is_server_running(host, timeout=5):
        return True

    if not auto_manage:
        if console:
            console.print(
                "[red]Cannot connect to Ollama.[/red]\n"
                "Start it manually with [bold]ollama serve[/bold] "
                "or set [bold]ollama.auto_manage = true[/bold] in your config."
            )
        return False

    return start_server(host=host, console=console)


def _get_path() -> str:
    """Get PATH from the current environment."""
    import os

    return os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
