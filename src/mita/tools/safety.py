"""Destructive action detection, banned command checking, and path safety."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from mita.config.schema import ToolSettings
from mita.tools.schema import ToolCall, ToolDefinition

# Paths that should never be read or written by tools
SENSITIVE_PATH_PATTERNS: list[str] = [
    ".ssh",
    ".gnupg",
    ".aws/credentials",
    ".env",
    ".netrc",
]

# Patterns that indicate destructive shell commands
DESTRUCTIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+(-[a-zA-Z]*f|-[a-zA-Z]*r|--force|--recursive)", re.IGNORECASE),
    re.compile(
        r"\bgit\s+(push\s+(--force|-f\b)|reset\s+--hard|clean\s+(-[a-zA-Z]*f|--force))",
        re.IGNORECASE,
    ),
    re.compile(r"\bchmod\s+(-R|--recursive)\s+0?7", re.IGNORECASE),
    re.compile(r"\bchown\s+(-R|--recursive)\b", re.IGNORECASE),
    re.compile(r"\b(truncate|shred)\b", re.IGNORECASE),
    re.compile(r">\s*/dev/\w+", re.IGNORECASE),
    re.compile(r"\bxargs\s+rm\b", re.IGNORECASE),
]

# Git subcommands that are destructive and need confirmation
DESTRUCTIVE_GIT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^push\s+--force", re.IGNORECASE),
    re.compile(r"^push\s+-f\b", re.IGNORECASE),
    re.compile(r"^reset\s+--hard", re.IGNORECASE),
    re.compile(r"^clean\s+-[a-zA-Z]*f", re.IGNORECASE),
    re.compile(r"^checkout\s+--\s", re.IGNORECASE),
    re.compile(r"^branch\s+-[dD]\b", re.IGNORECASE),
]


def _normalize_command(command: str) -> str:
    """Normalize a command for comparison by tokenizing and rejoining."""
    try:
        tokens = shlex.split(command.strip())
    except ValueError:
        # If shlex can't parse it, fall back to whitespace normalization
        tokens = command.strip().split()
    return " ".join(tokens)


def is_command_banned(command: str, banned_commands: list[str]) -> bool:
    """Check if a shell command matches any banned command pattern.

    Uses both substring matching on the raw command and token-normalized
    matching for robustness against whitespace/quoting variations.
    """
    raw = command.strip()
    normalized = _normalize_command(raw)
    for banned in banned_commands:
        banned_norm = _normalize_command(banned)
        if banned in raw or banned_norm in normalized:
            return True
    return False


def is_command_destructive(command: str) -> bool:
    """Check if a shell command looks destructive."""
    return any(pattern.search(command) for pattern in DESTRUCTIVE_PATTERNS)


def is_git_command_destructive(subcommand: str) -> bool:
    """Check if a git subcommand is destructive."""
    stripped = subcommand.strip()
    return any(pattern.search(stripped) for pattern in DESTRUCTIVE_GIT_PATTERNS)


def needs_confirmation(
    tool_call: ToolCall,
    tool_def: ToolDefinition,
    settings: ToolSettings,
) -> bool:
    """Determine if a tool call needs user confirmation before execution.

    Returns True if:
    - The tool is marked destructive AND confirm_destructive is on
      AND the tool is not in auto_approve.
    - Or the tool is 'shell' and the command looks destructive.
    - Or the tool is 'git' and the subcommand is destructive.
    """
    if not settings.confirm_destructive:
        return False

    if tool_call.name in settings.auto_approve:
        return False

    if tool_def.destructive:
        return True

    # Extra check for shell commands that look destructive
    if tool_call.name == "shell":
        command = tool_call.arguments.get("command", "")
        if is_command_destructive(command):
            return True

    # Check git subcommands for destructive operations
    if tool_call.name == "git":
        subcommand = tool_call.arguments.get("subcommand", "")
        if is_git_command_destructive(subcommand):
            return True

    return False


# ── Path boundary checking ───────────────────────────────────────

# Override for testing — set to a Path to bypass project root detection
_workspace_root_override: Path | None = None


def get_workspace_root() -> Path:
    """Return the workspace root directory for path boundary checks.

    Uses the project root (directory containing .git or .mita),
    falling back to the current working directory.
    """
    if _workspace_root_override is not None:
        return _workspace_root_override

    from mita.config.defaults import _find_project_root

    root = _find_project_root(Path.cwd())
    return root if root is not None else Path.cwd()


def _is_sensitive_path(resolved: Path) -> bool:
    """Check if a resolved path matches known sensitive locations."""
    path_str = str(resolved)
    home = str(Path.home())
    for pattern in SENSITIVE_PATH_PATTERNS:
        sensitive = f"{home}/{pattern}"
        if path_str == sensitive or path_str.startswith(f"{sensitive}/"):
            return True
    return False


def validate_path_for_read(resolved: Path) -> str | None:
    """Validate a resolved path is safe to read.

    Returns an error message if blocked, or None if allowed.
    Blocks access to known sensitive paths (SSH keys, credentials).
    """
    if _is_sensitive_path(resolved):
        return f"Access denied: {resolved} is a sensitive file"
    return None


def validate_path_for_write(resolved: Path) -> str | None:
    """Validate a resolved path is safe to write.

    Returns an error message if blocked, or None if allowed.
    Writes are restricted to the workspace root and its subdirectories.
    """
    if _is_sensitive_path(resolved):
        return f"Access denied: {resolved} is a sensitive file"

    workspace = get_workspace_root().resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError:
        return (
            f"Access denied: {resolved} is outside the workspace ({workspace}). "
            "File writes are restricted to the project directory."
        )
    return None


def validate_search_base(resolved: Path) -> str | None:
    """Validate a resolved directory is safe as a glob/grep search base.

    Returns an error message if blocked, or None if allowed.
    Search bases are restricted to the workspace root and its subdirectories.
    """
    workspace = get_workspace_root().resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError:
        return (
            f"Access denied: {resolved} is outside the workspace ({workspace}). "
            "Searches are restricted to the project directory."
        )
    return None
