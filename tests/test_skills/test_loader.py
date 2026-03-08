"""Tests for mita.skills.loader."""

from __future__ import annotations

from pathlib import Path

from mita.skills.loader import (
    Skill,
    SkillArg,
    SkillFrontmatter,
    _split_frontmatter,
    discover_skills,
    find_skill,
    parse_skill_file,
)

VALID_SKILL = """\
---
name: test-skill
description: "A test skill"
args:
  - name: target
    type: string
    description: "The target"
    required: true
trigger: "^/test-skill$"
---

Do something with {{target}}.
"""

MINIMAL_SKILL = """\
---
name: minimal
---

Hello.
"""

NO_FRONTMATTER = "Just plain markdown."

BAD_YAML = """\
---
name: [invalid
---

Body.
"""


class TestSplitFrontmatter:
    def test_valid_frontmatter(self) -> None:
        fm, body = _split_frontmatter(VALID_SKILL)
        assert fm is not None
        assert "name: test-skill" in fm
        assert "Do something" in body

    def test_no_frontmatter(self) -> None:
        fm, body = _split_frontmatter(NO_FRONTMATTER)
        assert fm is None
        assert body == NO_FRONTMATTER

    def test_minimal(self) -> None:
        fm, body = _split_frontmatter(MINIMAL_SKILL)
        assert fm is not None
        assert "name: minimal" in fm
        assert body == "Hello."


class TestParseSkillFile:
    def test_valid(self, tmp_path: Path) -> None:
        p = tmp_path / "test.md"
        p.write_text(VALID_SKILL)
        skill = parse_skill_file(p)
        assert skill is not None
        assert skill.frontmatter.name == "test-skill"
        assert skill.frontmatter.description == "A test skill"
        assert len(skill.frontmatter.args) == 1
        assert skill.frontmatter.args[0].name == "target"
        assert skill.frontmatter.trigger == "^/test-skill$"
        assert "{{target}}" in skill.template

    def test_minimal(self, tmp_path: Path) -> None:
        p = tmp_path / "minimal.md"
        p.write_text(MINIMAL_SKILL)
        skill = parse_skill_file(p)
        assert skill is not None
        assert skill.frontmatter.name == "minimal"
        assert skill.frontmatter.args == []

    def test_no_frontmatter(self, tmp_path: Path) -> None:
        p = tmp_path / "plain.md"
        p.write_text(NO_FRONTMATTER)
        assert parse_skill_file(p) is None

    def test_bad_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.md"
        p.write_text(BAD_YAML)
        assert parse_skill_file(p) is None

    def test_missing_name(self, tmp_path: Path) -> None:
        p = tmp_path / "noname.md"
        p.write_text("---\ndescription: no name\n---\nBody.")
        assert parse_skill_file(p) is None

    def test_nonexistent_file(self) -> None:
        assert parse_skill_file(Path("/nonexistent/skill.md")) is None


class TestDiscoverSkills:
    def test_discovers_skills(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text(VALID_SKILL)
        (tmp_path / "b.md").write_text(MINIMAL_SKILL)
        skills = discover_skills([str(tmp_path)])
        names = {s.frontmatter.name for s in skills}
        assert names == {"test-skill", "minimal"}

    def test_deduplicates_by_name(self, tmp_path: Path) -> None:
        d1 = tmp_path / "dir1"
        d2 = tmp_path / "dir2"
        d1.mkdir()
        d2.mkdir()
        (d1 / "skill.md").write_text(VALID_SKILL)
        (d2 / "skill.md").write_text(VALID_SKILL)
        skills = discover_skills([str(d1), str(d2)])
        assert len(skills) == 1

    def test_skips_missing_dirs(self) -> None:
        skills = discover_skills(["/nonexistent/path"])
        assert skills == []

    def test_skips_non_md_files(self, tmp_path: Path) -> None:
        (tmp_path / "readme.txt").write_text("not a skill")
        skills = discover_skills([str(tmp_path)])
        assert skills == []


class TestFindSkill:
    def _make_skills(self) -> list[Skill]:
        return [
            Skill(
                frontmatter=SkillFrontmatter(
                    name="commit",
                    description="Commit skill",
                    trigger="^/commit$",
                ),
                template="Commit template",
                source_path=Path("commit.md"),
            ),
            Skill(
                frontmatter=SkillFrontmatter(
                    name="explain",
                    description="Explain skill",
                    args=[SkillArg(name="target", description="what to explain")],
                ),
                template="Explain {{target}}",
                source_path=Path("explain.md"),
            ),
        ]

    def test_find_by_name(self) -> None:
        skills = self._make_skills()
        result = find_skill(skills, "/commit")
        assert result is not None
        assert result.frontmatter.name == "commit"

    def test_find_by_trigger(self) -> None:
        skills = self._make_skills()
        result = find_skill(skills, "/commit")
        assert result is not None
        assert result.frontmatter.name == "commit"

    def test_find_with_args(self) -> None:
        skills = self._make_skills()
        result = find_skill(skills, "/explain src/main.py")
        assert result is not None
        assert result.frontmatter.name == "explain"

    def test_not_found(self) -> None:
        skills = self._make_skills()
        assert find_skill(skills, "/nonexistent") is None

    def test_empty_input(self) -> None:
        skills = self._make_skills()
        assert find_skill(skills, "") is None
