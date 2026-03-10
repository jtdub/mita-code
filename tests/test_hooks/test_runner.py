"""Tests for hooks runner."""

from __future__ import annotations

import pytest

from mita.config.schema import HookDefinition
from mita.hooks.runner import VALID_EVENTS, _matches, _render_command, run_hooks


class TestValidEvents:
    def test_expected_events(self) -> None:
        assert "session_start" in VALID_EVENTS
        assert "session_end" in VALID_EVENTS
        assert "pre_tool_call" in VALID_EVENTS
        assert "post_tool_call" in VALID_EVENTS
        assert "on_file_write" in VALID_EVENTS


class TestMatches:
    def test_no_match_pattern(self) -> None:
        hook = HookDefinition(event="on_file_write", command="echo hi")
        assert _matches(hook, {"file_path": "/tmp/test.py"})

    def test_match_file_path(self) -> None:
        hook = HookDefinition(event="on_file_write", command="echo", match="*.py")
        assert _matches(hook, {"file_path": "src/main.py"})
        assert not _matches(hook, {"file_path": "src/main.js"})

    def test_match_tool_name(self) -> None:
        hook = HookDefinition(event="pre_tool_call", command="echo", match="file_*")
        assert _matches(hook, {"tool": "file_write"})
        assert _matches(hook, {"tool": "file_edit"})
        assert not _matches(hook, {"tool": "shell"})

    def test_no_context(self) -> None:
        hook = HookDefinition(event="session_start", command="echo", match="*.py")
        assert _matches(hook, None)

    def test_context_without_matching_keys(self) -> None:
        hook = HookDefinition(event="pre_tool_call", command="echo", match="*.py")
        # No file_path or tool in context
        assert not _matches(hook, {"something": "else"})


class TestRenderCommand:
    def test_no_context(self) -> None:
        assert _render_command("echo hello", None) == "echo hello"

    def test_substitute_file_path(self) -> None:
        result = _render_command("ruff check {file_path}", {"file_path": "src/main.py"})
        assert result == "ruff check src/main.py"

    def test_substitute_multiple(self) -> None:
        result = _render_command(
            "echo {tool} {file_path}", {"tool": "file_write", "file_path": "/tmp/f.py"}
        )
        assert result == "echo file_write /tmp/f.py"

    def test_missing_variable_left_alone(self) -> None:
        result = _render_command("echo {unknown}", {"file_path": "test.py"})
        assert result == "echo {unknown}"

    def test_shell_injection_escaped(self) -> None:
        result = _render_command("echo {file_path}", {"file_path": '"; rm -rf /'})
        assert "rm -rf" not in result or "'" in result
        # The malicious payload should be safely quoted
        assert result.startswith("echo ")

    def test_spaces_in_path_quoted(self) -> None:
        result = _render_command("echo {file_path}", {"file_path": "path with spaces/file.py"})
        assert "'path with spaces/file.py'" in result


class TestRunHooks:
    @pytest.mark.asyncio
    async def test_no_matching_hooks(self) -> None:
        hooks = [HookDefinition(event="session_start", command="echo hi")]
        results = await run_hooks("session_end", hooks)
        assert results == []

    @pytest.mark.asyncio
    async def test_run_simple_hook(self) -> None:
        hooks = [HookDefinition(event="session_start", command="echo hello")]
        results = await run_hooks("session_start", hooks)
        assert len(results) == 1
        assert results[0]["returncode"] == 0
        assert "hello" in results[0]["stdout"]

    @pytest.mark.asyncio
    async def test_hook_with_context(self) -> None:
        hooks = [HookDefinition(event="on_file_write", command="echo {file_path}")]
        results = await run_hooks("on_file_write", hooks, context={"file_path": "/tmp/test.py"})
        assert len(results) == 1
        assert "/tmp/test.py" in results[0]["stdout"]

    @pytest.mark.asyncio
    async def test_hook_with_match_filter(self) -> None:
        hooks = [
            HookDefinition(event="on_file_write", command="echo py", match="*.py"),
            HookDefinition(event="on_file_write", command="echo js", match="*.js"),
        ]
        results = await run_hooks("on_file_write", hooks, context={"file_path": "test.py"})
        assert len(results) == 1
        assert "py" in results[0]["stdout"]

    @pytest.mark.asyncio
    async def test_hook_failure_captured(self) -> None:
        hooks = [HookDefinition(event="session_start", command="false")]
        results = await run_hooks("session_start", hooks)
        assert len(results) == 1
        assert results[0]["returncode"] != 0

    @pytest.mark.asyncio
    async def test_hook_timeout(self) -> None:
        hooks = [HookDefinition(event="session_start", command="sleep 10")]
        results = await run_hooks("session_start", hooks, timeout=0.5)
        assert len(results) == 1
        assert results[0]["returncode"] == -1
        assert "timed out" in results[0]["stderr"]

    @pytest.mark.asyncio
    async def test_multiple_hooks_same_event(self) -> None:
        hooks = [
            HookDefinition(event="session_start", command="echo one"),
            HookDefinition(event="session_start", command="echo two"),
        ]
        results = await run_hooks("session_start", hooks)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_hook_invalid_command(self) -> None:
        hooks = [
            HookDefinition(event="session_start", command="/nonexistent/binary/that/does/not/exist")
        ]
        results = await run_hooks("session_start", hooks)
        assert len(results) == 1
        assert results[0]["returncode"] != 0

    @pytest.mark.asyncio
    async def test_event_included_in_result(self) -> None:
        hooks = [HookDefinition(event="session_end", command="echo done")]
        results = await run_hooks("session_end", hooks)
        assert results[0]["event"] == "session_end"

    @pytest.mark.asyncio
    async def test_hook_with_console_output(self) -> None:
        from rich.console import Console

        console = Console(file=open("/dev/null", "w"))
        hooks = [HookDefinition(event="session_start", command="echo test")]
        results = await run_hooks("session_start", hooks, console=console)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_hook_failure_with_console(self) -> None:
        from rich.console import Console

        console = Console(file=open("/dev/null", "w"))
        hooks = [HookDefinition(event="session_start", command="false")]
        results = await run_hooks("session_start", hooks, console=console)
        assert len(results) == 1
        assert results[0]["returncode"] != 0
