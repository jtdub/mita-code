"""Tests for mita.skills.manager."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from mita.skills.manager import create_skill, list_skills, show_paths, show_skill

VALID_SKILL = """\
---
name: demo
description: "A demo skill"
args: []
---

Demo prompt.
"""


def _write_skill(tmp_path: Path) -> Path:
    d = tmp_path / "skills"
    d.mkdir()
    (d / "demo.md").write_text(VALID_SKILL)
    return d


class TestListSkills:
    def test_lists_skills(self, tmp_path: Path, capsys: object) -> None:
        d = _write_skill(tmp_path)
        with patch("mita.skills.manager.load_config") as mock_cfg:
            mock_cfg.return_value.skills_paths = [str(d)]
            list_skills()

    def test_no_skills(self, tmp_path: Path) -> None:
        with patch("mita.skills.manager.load_config") as mock_cfg:
            mock_cfg.return_value.skills_paths = [str(tmp_path / "empty")]
            list_skills()


class TestShowSkill:
    def test_found(self, tmp_path: Path) -> None:
        d = _write_skill(tmp_path)
        with patch("mita.skills.manager.load_config") as mock_cfg:
            mock_cfg.return_value.skills_paths = [str(d)]
            show_skill("demo")

    def test_not_found(self, tmp_path: Path) -> None:
        d = _write_skill(tmp_path)
        with patch("mita.skills.manager.load_config") as mock_cfg:
            mock_cfg.return_value.skills_paths = [str(d)]
            show_skill("nonexistent")


class TestCreateSkill:
    def test_creates_file(self, tmp_path: Path) -> None:
        target = tmp_path / "skills"
        with patch("mita.skills.manager.load_config") as mock_cfg:
            mock_cfg.return_value.skills_paths = [str(target)]
            create_skill("new-skill")
        assert (target / "new-skill.md").exists()
        content = (target / "new-skill.md").read_text()
        assert "name: new-skill" in content

    def test_no_overwrite(self, tmp_path: Path) -> None:
        d = _write_skill(tmp_path)
        with patch("mita.skills.manager.load_config") as mock_cfg:
            mock_cfg.return_value.skills_paths = [str(d)]
            create_skill("demo")  # Already exists
        # File should still have original content
        assert "A demo skill" in (d / "demo.md").read_text()


class TestShowPaths:
    def test_shows_paths(self, tmp_path: Path) -> None:
        with patch("mita.skills.manager.load_config") as mock_cfg:
            mock_cfg.return_value.skills_paths = [str(tmp_path)]
            show_paths()
