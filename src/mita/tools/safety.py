"""Destructive action detection and banned command checking."""

from __future__ import annotations

import re

from mita.config.schema import ToolSettings
from mita.tools.schema import ToolCall, ToolDefinition

# Patterns that indicate destructive shell commands
DESTRUCTIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+(-[a-zA-Z]*f|-[a-zA-Z]*r)", re.IGNORECASE),
    re.compile(r"\bgit\s+(push\s+--force|reset\s+--hard|clean\s+-[a-zA-Z]*f)", re.IGNORECASE),
    re.compile(r"\bchmod\s+-R\s+0?7", re.IGNORECASE),
    re.compile(r"\bchown\s+-R\b", re.IGNORECASE),
    re.compile(r"\b(truncate|shred)\b", re.IGNORECASE),
    re.compile(r">\s*/dev/\w+", re.IGNORECASE),
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


def is_command_banned(command: str, banned_commands: list[str]) -> bool:
    """Check if a shell command matches any banned command pattern."""
    normalized = command.strip()
    for banned in banned_commands:
        if banned in normalized:
            return True
    return False


def is_command_destructive(command: str) -> bool:
    """Check if a shell command looks destructive."""
    for pattern in DESTRUCTIVE_PATTERNS:
        if pattern.search(command):
            return True
    return False


def is_git_command_destructive(subcommand: str) -> bool:
    """Check if a git subcommand is destructive."""
    stripped = subcommand.strip()
    for pattern in DESTRUCTIVE_GIT_PATTERNS:
        if pattern.search(stripped):
            return True
    return False


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
