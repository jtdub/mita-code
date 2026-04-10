"""Destructive action detection, banned command checking, and path safety."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from mita.config.schema import ToolSettings
from mita.tools.builtins.git import is_safe_git_command
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

# Shell command prefixes that should be skipped when extracting the actual command
COMMAND_PREFIXES: frozenset[str] = frozenset({"sudo", "env", "nohup", "nice", "time", "strace"})

# Split shell pipelines and chains — longer operators first to avoid partial matches
_SHELL_OPERATOR_RE = re.compile(r"&&|\|\||[|;&]")


def _tokenize(command: str) -> list[str]:
    """Tokenize a shell command, falling back to whitespace split on parse errors."""
    try:
        return shlex.split(command.strip())
    except ValueError:
        return command.strip().split()


def _normalize_command(command: str) -> str:
    """Normalize a command for comparison by tokenizing and rejoining."""
    return " ".join(_tokenize(command))


def _extract_command_names(command: str) -> set[str]:
    """Extract base command names from a shell command, handling pipes and chains."""
    segments = _SHELL_OPERATOR_RE.split(command)
    names: set[str] = set()
    for segment in segments:
        for token in _tokenize(segment):
            if "=" in token:
                continue
            if token in COMMAND_PREFIXES:
                continue
            names.add(token)
            break
    return names


def is_command_banned(command: str, banned_commands: list[str]) -> bool:
    """Check if a shell command matches any banned command pattern.

    Uses multiple strategies:
    1. Substring matching on the raw and normalized command (catches exact phrases)
    2. Parsed command name matching (catches the command regardless of flags)
    """
    raw = command.strip()
    normalized = _normalize_command(raw)
    cmd_names = _extract_command_names(raw)

    for banned in banned_commands:
        banned_norm = _normalize_command(banned)
        if banned in raw or banned_norm in normalized:
            return True
        banned_names = _extract_command_names(banned)
        if banned_names and banned_names.issubset(cmd_names):
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
    session_approved: set[str] | None = None,
) -> bool:
    """Determine if a tool call needs user confirmation before execution.

    Returns True if:
    - The tool is marked destructive AND confirm_destructive is on
      AND the tool is not in the effective auto-approve set or session-approved set.
    - Or the tool is 'shell' and the command looks destructive.
    - Or the tool is 'git' and the subcommand is destructive (not read-only).
    """
    if not settings.confirm_destructive:
        return False

    effective = settings.effective_auto_approve
    if session_approved:
        effective = effective | session_approved

    if tool_call.name in effective:
        return False

    if tool_def.destructive:
        return True

    if tool_call.name == "shell":
        command = tool_call.arguments.get("command", "")
        if is_command_destructive(command):
            return True

    if tool_call.name == "git":
        subcommand = tool_call.arguments.get("subcommand", "")
        if is_safe_git_command(subcommand):
            return False
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


def _check_within_workspace(resolved: Path, operation: str) -> str | None:
    """Check that a resolved path is within the workspace root.

    Returns an error message if the path is outside the workspace, or None if allowed.
    """
    workspace = get_workspace_root().resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError:
        return (
            f"Access denied: {resolved} is outside the workspace ({workspace}). "
            f"{operation} are restricted to the project directory."
        )
    return None


def validate_path_for_read(resolved: Path) -> str | None:
    """Validate a resolved path is safe to read.

    Returns an error message if blocked, or None if allowed.
    Blocks access to known sensitive paths and paths outside the workspace.
    """
    if _is_sensitive_path(resolved):
        return f"Access denied: {resolved} is a sensitive file"
    return _check_within_workspace(resolved, "File reads")


def validate_path_for_write(resolved: Path) -> str | None:
    """Validate a resolved path is safe to write.

    Returns an error message if blocked, or None if allowed.
    Writes are restricted to the workspace root and its subdirectories.
    """
    if _is_sensitive_path(resolved):
        return f"Access denied: {resolved} is a sensitive file"
    return _check_within_workspace(resolved, "File writes")


def validate_search_base(resolved: Path) -> str | None:
    """Validate a resolved directory is safe as a glob/grep search base.

    Returns an error message if blocked, or None if allowed.
    Search bases are restricted to the workspace root and its subdirectories.
    """
    return _check_within_workspace(resolved, "Searches")
