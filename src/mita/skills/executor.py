"""Render skill templates with argument substitution."""

from __future__ import annotations

import re

from mita.skills.loader import Skill


def render_skill(skill: Skill, user_input: str) -> str:
    """Render a skill template, substituting arguments from user input.

    Args:
        skill: The skill to render.
        user_input: The raw user input (e.g., "/commit -m 'fix bug'").

    Returns:
        The rendered prompt string.
    """
    args = _parse_args(user_input, skill)
    return _substitute(skill.template, args)


def _parse_args(user_input: str, skill: Skill) -> dict[str, str]:
    """Parse arguments from user input into a dict."""
    args: dict[str, str] = {}

    # Remove the skill name prefix (e.g., "/commit" -> rest of input)
    parts = user_input.strip().split(None, 1)
    remainder = parts[1] if len(parts) > 1 else ""

    # Map positional and named args
    skill_args = skill.frontmatter.args
    if not skill_args:
        # No declared args — pass entire remainder as {{input}}
        args["input"] = remainder
        return args

    # Simple approach: split remainder by spaces, assign positionally
    # Also support --name=value and --name value
    tokens = _tokenize(remainder)
    positional_idx = 0

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            # Named argument
            key = token.lstrip("-")
            if "=" in key:
                k, v = key.split("=", 1)
                args[k] = v
            elif i + 1 < len(tokens):
                args[key] = tokens[i + 1]
                i += 1
        elif positional_idx < len(skill_args):
            args[skill_args[positional_idx].name] = token
            positional_idx += 1
        i += 1

    # Fill defaults for missing args
    for arg in skill_args:
        if arg.name not in args and arg.default is not None:
            args[arg.name] = str(arg.default)

    return args


def _tokenize(text: str) -> list[str]:
    """Tokenize input, respecting quoted strings."""
    tokens: list[str] = []
    for match in re.finditer(r'"([^"]*)"' r"|'([^']*)'" r"|(\S+)", text):
        tokens.append(match.group(1) or match.group(2) or match.group(3) or "")
    return tokens


def _substitute(template: str, args: dict[str, str]) -> str:
    """Replace {{arg}} placeholders in the template.

    Raises:
        ValueError: If any placeholder has no matching argument.
    """
    # Find all placeholders first to validate
    placeholders = re.findall(r"\{\{(\w+)\}\}", template)
    missing = [p.strip() for p in placeholders if p.strip() not in args]
    if missing:
        raise ValueError(f"Missing required skill arguments: {', '.join(sorted(set(missing)))}")

    def replacer(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        return args[key]

    return re.sub(r"\{\{(\w+)\}\}", replacer, template)
