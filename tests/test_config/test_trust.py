"""Tests for the project-directory trust store (audit finding C1)."""

from __future__ import annotations

from pathlib import Path

from mita.config.loader import load_config
from mita.config.trust import (
    add_trusted,
    filter_untrusted,
    is_trusted,
    security_relevant_keys,
)


class TestFilterUntrusted:
    def test_strips_hooks_and_plugins(self) -> None:
        data = {
            "model": {"default": "x"},
            "hooks": [{"event": "session_start", "command": "echo hi"}],
            "plugins": [{"name": "p", "command": "npx", "args": []}],
        }
        cleaned, stripped = filter_untrusted(data)
        assert "hooks" not in cleaned
        assert "plugins" not in cleaned
        assert cleaned["model"] == {"default": "x"}
        assert set(stripped) == {"hooks", "plugins"}

    def test_strips_gate_weakening_tool_keys_only(self) -> None:
        data = {
            "tools": {
                "permission_mode": "trust",
                "auto_approve": ["shell"],
                "confirm_destructive": False,
                "banned_commands": ["rm -rf /"],
            }
        }
        cleaned, stripped = filter_untrusted(data)
        assert cleaned["tools"] == {"banned_commands": ["rm -rf /"]}
        assert set(stripped) == {
            "tools.permission_mode",
            "tools.auto_approve",
            "tools.confirm_destructive",
        }

    def test_no_security_keys_is_unchanged(self) -> None:
        data = {"model": {"default": "x"}}
        cleaned, stripped = filter_untrusted(data)
        assert cleaned == data
        assert stripped == []

    def test_input_not_mutated(self) -> None:
        data = {"hooks": [{"event": "session_start", "command": "x"}]}
        filter_untrusted(data)
        assert "hooks" in data


class TestTrustStore:
    def test_untrusted_by_default(self, tmp_path: Path) -> None:
        assert is_trusted(tmp_path) is False

    def test_add_and_check(self, tmp_path: Path) -> None:
        add_trusted(tmp_path)
        assert is_trusted(tmp_path) is True

    def test_security_relevant_keys(self) -> None:
        data = {"hooks": [], "tools": {"permission_mode": "trust"}, "model": {}}
        keys = security_relevant_keys(data)
        assert "hooks" in keys
        assert "tools.permission_mode" in keys
        assert "model" not in keys


class TestLoadConfigTrust:
    def _write_project(self, root: Path, body: str) -> None:
        mita_dir = root / ".mita"
        mita_dir.mkdir(exist_ok=True)
        (mita_dir / "settings.toml").write_text(body)

    def test_untrusted_project_hooks_ignored(self, tmp_project: Path) -> None:
        self._write_project(
            tmp_project,
            '[[hooks]]\nevent = "session_start"\ncommand = "echo pwned"\n',
        )
        cfg = load_config(project_root=tmp_project)
        assert cfg.hooks == []

    def test_untrusted_permission_mode_ignored(self, tmp_project: Path) -> None:
        self._write_project(tmp_project, '[tools]\npermission_mode = "trust"\n')
        cfg = load_config(project_root=tmp_project)
        assert cfg.tools.permission_mode.value == "ask"

    def test_trusted_project_hooks_applied(self, tmp_project: Path) -> None:
        self._write_project(
            tmp_project,
            '[[hooks]]\nevent = "session_start"\ncommand = "echo ok"\n',
        )
        add_trusted(tmp_project)
        cfg = load_config(project_root=tmp_project)
        assert len(cfg.hooks) == 1
        assert cfg.hooks[0].command == "echo ok"

    def test_explicit_trust_override_applies_keys(self, tmp_project: Path) -> None:
        self._write_project(tmp_project, '[tools]\npermission_mode = "trust"\n')
        cfg = load_config(project_root=tmp_project, trust_project=True)
        assert cfg.tools.permission_mode.value == "trust"

    def test_non_security_keys_always_applied(self, tmp_project: Path) -> None:
        self._write_project(tmp_project, '[model]\ndefault = "custom:7b"\n')
        cfg = load_config(project_root=tmp_project)
        assert cfg.model.default == "custom:7b"
