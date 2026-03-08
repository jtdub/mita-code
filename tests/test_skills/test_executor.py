"""Tests for mita.skills.executor."""

from __future__ import annotations

from pathlib import Path

from mita.skills.executor import _parse_args, _substitute, _tokenize, render_skill
from mita.skills.loader import Skill, SkillArg, SkillFrontmatter


def _make_skill(
    name: str = "test",
    args: list[SkillArg] | None = None,
    template: str = "Do {{input}}.",
) -> Skill:
    return Skill(
        frontmatter=SkillFrontmatter(
            name=name,
            description="test",
            args=args or [],
        ),
        template=template,
        source_path=Path("test.md"),
    )


class TestTokenize:
    def test_simple(self) -> None:
        assert _tokenize("foo bar baz") == ["foo", "bar", "baz"]

    def test_double_quotes(self) -> None:
        assert _tokenize('foo "bar baz"') == ["foo", "bar baz"]

    def test_single_quotes(self) -> None:
        assert _tokenize("foo 'bar baz'") == ["foo", "bar baz"]

    def test_empty(self) -> None:
        assert _tokenize("") == []

    def test_mixed(self) -> None:
        assert _tokenize("""hello "world test" 'foo bar'""") == [
            "hello",
            "world test",
            "foo bar",
        ]


class TestSubstitute:
    def test_basic(self) -> None:
        assert _substitute("Hello {{name}}!", {"name": "world"}) == "Hello world!"

    def test_missing_key(self) -> None:
        assert _substitute("Hello {{name}}!", {}) == "Hello {{name}}!"

    def test_multiple(self) -> None:
        result = _substitute("{{a}} and {{b}}", {"a": "X", "b": "Y"})
        assert result == "X and Y"


class TestParseArgs:
    def test_no_declared_args(self) -> None:
        skill = _make_skill()
        args = _parse_args("/test some input", skill)
        assert args == {"input": "some input"}

    def test_positional_arg(self) -> None:
        skill = _make_skill(
            args=[SkillArg(name="target", description="target")],
            template="Explain {{target}}.",
        )
        args = _parse_args("/test src/main.py", skill)
        assert args == {"target": "src/main.py"}

    def test_named_arg(self) -> None:
        skill = _make_skill(
            args=[SkillArg(name="target", description="target")],
            template="Explain {{target}}.",
        )
        args = _parse_args("/test --target=src/main.py", skill)
        assert args == {"target": "src/main.py"}

    def test_named_arg_space(self) -> None:
        skill = _make_skill(
            args=[SkillArg(name="target", description="target")],
            template="Explain {{target}}.",
        )
        args = _parse_args("/test --target src/main.py", skill)
        assert args == {"target": "src/main.py"}

    def test_default_value(self) -> None:
        skill = _make_skill(
            args=[
                SkillArg(
                    name="target",
                    description="target",
                    required=False,
                    default="staged",
                ),
            ],
            template="Review {{target}}.",
        )
        args = _parse_args("/test", skill)
        assert args == {"target": "staged"}

    def test_no_input(self) -> None:
        skill = _make_skill()
        args = _parse_args("/test", skill)
        assert args == {"input": ""}


class TestRenderSkill:
    def test_render_with_args(self) -> None:
        skill = _make_skill(
            args=[SkillArg(name="target", description="target")],
            template="Explain {{target}} in detail.",
        )
        result = render_skill(skill, "/test src/main.py")
        assert result == "Explain src/main.py in detail."

    def test_render_no_args(self) -> None:
        skill = _make_skill(template="Do {{input}}.")
        result = render_skill(skill, "/test hello world")
        assert result == "Do hello world."

    def test_render_default_args(self) -> None:
        skill = _make_skill(
            args=[
                SkillArg(name="target", required=False, default="staged"),
            ],
            template="Review {{target}}.",
        )
        result = render_skill(skill, "/test")
        assert result == "Review staged."
