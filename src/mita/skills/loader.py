"""Discover and parse skill Markdown files from configured paths."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class SkillFrontmatter(BaseModel):
    """YAML frontmatter for a skill file."""

    name: str
    description: str = ""
    args: list[SkillArg] = Field(default_factory=list)
    trigger: str | None = None


class SkillArg(BaseModel):
    """A parameter accepted by a skill."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any | None = None


# Rebuild SkillFrontmatter to pick up SkillArg
SkillFrontmatter.model_rebuild()


class Skill(BaseModel):
    """A parsed skill: frontmatter + Markdown template."""

    frontmatter: SkillFrontmatter
    template: str
    source_path: Path


def discover_skills(skills_paths: list[str]) -> list[Skill]:
    """Discover and load all skills from the configured paths."""
    skills: list[Skill] = []
    seen_names: set[str] = set()

    for raw_path in skills_paths:
        path = Path(raw_path).expanduser()
        if not path.is_dir():
            continue
        for md_file in sorted(path.glob("*.md")):
            skill = parse_skill_file(md_file)
            if skill and skill.frontmatter.name not in seen_names:
                skills.append(skill)
                seen_names.add(skill.frontmatter.name)

    return skills


def parse_skill_file(path: Path) -> Skill | None:
    """Parse a single skill Markdown file."""
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None

    frontmatter, template = _split_frontmatter(content)
    if frontmatter is None:
        return None

    try:
        data = yaml.safe_load(frontmatter)
    except yaml.YAMLError:
        return None

    if not isinstance(data, dict) or "name" not in data:
        return None

    try:
        fm = SkillFrontmatter(**data)
    except Exception:  # noqa: BLE001
        return None

    return Skill(frontmatter=fm, template=template.strip(), source_path=path)


def find_skill(skills: list[Skill], name_or_trigger: str) -> Skill | None:
    """Find a skill by name or trigger pattern."""
    # Strip leading slash for matching
    query = name_or_trigger.lstrip("/").split()[0] if name_or_trigger else ""

    for skill in skills:
        if skill.frontmatter.name == query:
            return skill
        if skill.frontmatter.trigger:
            if re.search(skill.frontmatter.trigger, name_or_trigger):
                return skill

    return None


def _split_frontmatter(content: str) -> tuple[str | None, str]:
    """Split YAML frontmatter from Markdown body."""
    content = content.strip()
    if not content.startswith("---"):
        return None, content

    # Find the closing ---
    end = content.find("---", 3)
    if end == -1:
        return None, content

    frontmatter = content[3:end].strip()
    body = content[end + 3 :].strip()
    return frontmatter, body
