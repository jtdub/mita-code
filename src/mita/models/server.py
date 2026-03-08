"""Ollama server lifecycle management — start, stop, health check.

Runs Ollama as a detached daemon tracked via a PID file so it persists
across short-lived CLI commands like ``mita models list``.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from rich.console import Console

from mita.models.ollama_client import OllamaClient


def _get_pid_dir() -> Path:
    """Return the directory for the PID file (lazy to avoid module-level Path.home())."""
    return Path.home() / ".config" / "mita"


def _get_pid_file() -> Path:
    """Return the PID file path (lazy to avoid module-level Path.home())."""
    return _get_pid_dir() / "ollama.pid"


def find_ollama_binary() -> str | None:
    """Locate the ollama binary on the system."""
    return shutil.which("ollama")


def is_server_running(host: str = "http://localhost:11434", timeout: int = 5) -> bool:
    """Check if an Ollama server is reachable."""
    client = OllamaClient(host=host, timeout=timeout)
    return client.is_running()


def _read_pid() -> int | None:
    """Read the PID from the PID file, return None if missing or stale."""
    try:
        pid = int(_get_pid_file().read_text().strip())
        # Check if process is alive
        os.kill(pid, 0)
        return pid
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError, OSError):
        _remove_pid_file()
        return None


def _write_pid(pid: int) -> None:
    """Write a PID to the PID file."""
    _get_pid_dir().mkdir(parents=True, exist_ok=True)
    _get_pid_file().write_text(str(pid))


def _remove_pid_file() -> None:
    """Remove the PID file if it exists."""
    try:
        _get_pid_file().unlink()
    except FileNotFoundError:
        pass


def start_server(
    host: str = "http://localhost:11434",
    timeout: int = 30,
    console: Console | None = None,
) -> bool:
    """Start the Ollama server as a detached daemon.

    The server persists after Mita exits. Its PID is tracked in
    ``~/.config/mita/ollama.pid`` so ``stop_server`` can find it later.

    Returns True if the server is running (either already was or we started it).
    """
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

    env = os.environ.copy()
    env_host = host.replace("http://", "").replace("https://", "")
    env["OLLAMA_HOST"] = env_host

    try:
        proc = subprocess.Popen(
            [binary, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
    except OSError as e:
        if console:
            console.print(f"[red]Failed to start Ollama: {e}[/red]")
        return False

    _write_pid(proc.pid)

    # Wait for the server to become healthy
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            if console:
                console.print("[red]Ollama server exited unexpectedly.[/red]")
            _remove_pid_file()
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
    """Stop the Ollama server if it was started by Mita (tracked via PID file).

    Returns True if we stopped it, False if there was nothing to stop.
    """
    pid = _read_pid()
    if pid is None:
        return False

    try:
        os.kill(pid, signal.SIGTERM)
        # Wait for graceful shutdown
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
                time.sleep(0.25)
            except ProcessLookupError:
                break
        else:
            # Still alive after 10s — force kill
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    except ProcessLookupError:
        pass  # Already dead

    _remove_pid_file()

    if console:
        console.print("[dim]Ollama server stopped.[/dim]")

    return True


def is_managed() -> bool:
    """Return True if a Mita-started Ollama server is tracked and alive."""
    return _read_pid() is not None


def ensure_server(
    host: str = "http://localhost:11434",
    auto_manage: bool = True,
    console: Console | None = None,
) -> bool:
    """Ensure Ollama is available — start it if auto_manage is enabled."""
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


def ensure_model(
    model_name: str,
    host: str = "http://localhost:11434",
    timeout: int = 120,
    console: Console | None = None,
) -> bool:
    """Ensure a model is installed, pulling it automatically if missing."""
    client = OllamaClient(host=host, timeout=timeout)

    # Check if model is already installed
    try:
        installed = client.list_models()
        for m in installed:
            # Match both exact name and name without tag
            if m.name == model_name or m.name == f"{model_name}:latest":
                return True
            # Also match if user specified without tag
            if m.name.split(":")[0] == model_name.split(":")[0] and (
                ":" not in model_name or m.name == model_name
            ):
                return True
    except (ConnectionError, OSError):
        if console:
            console.print("[red]Cannot connect to Ollama to check models.[/red]")
        return False

    # Model not found — pull it
    if console:
        console.print(f"[yellow]Model '{model_name}' is not installed.[/yellow]")
        console.print(f"[dim]Pulling {model_name}...[/dim]")

    try:
        from rich.progress import (
            BarColumn,
            DownloadColumn,
            Progress,
            TextColumn,
            TransferSpeedColumn,
        )

        if console:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Downloading", total=None)
                for update in client.pull(model_name, stream=True):
                    status = update.get("status", "")
                    completed = update.get("completed", 0)
                    total = update.get("total", 0)
                    if total > 0:
                        progress.update(
                            task, completed=completed, total=total, description=status
                        )
                    else:
                        progress.update(task, description=status)
                progress.update(task, description="Complete")
            console.print(f"[green]Model '{model_name}' pulled successfully.[/green]")
        else:
            for _update in client.pull(model_name, stream=False):
                pass
        return True
    except Exception as e:
        if console:
            console.print(f"[red]Failed to pull '{model_name}': {e}[/red]")
        return False
