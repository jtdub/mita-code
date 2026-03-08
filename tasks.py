"""Invoke task definitions for Mita Code development."""

from invoke import Context, task


@task
def test(c: Context, verbose: bool = False) -> None:
    """Run tests with pytest."""
    cmd = "poetry run pytest tests/"
    if verbose:
        cmd += " -v"
    c.run(cmd, pty=True)


@task
def lint(c: Context, fix: bool = False) -> None:
    """Run ruff linter."""
    cmd = "poetry run ruff check src/ tests/"
    if fix:
        cmd += " --fix"
    c.run(cmd, pty=True)


@task
def fmt(c: Context, check: bool = False) -> None:
    """Run ruff formatter."""
    cmd = "poetry run ruff format src/ tests/"
    if check:
        cmd += " --check"
    c.run(cmd, pty=True)


@task
def isort(c: Context, check: bool = False) -> None:
    """Run isort import sorter."""
    cmd = "poetry run isort src/ tests/"
    if check:
        cmd += " --check-only --diff"
    c.run(cmd, pty=True)


@task
def typecheck(c: Context) -> None:
    """Run mypy type checking."""
    c.run("poetry run mypy src/", pty=True)


@task
def pylint_check(c: Context) -> None:
    """Run pylint."""
    c.run("poetry run pylint src/mita/", pty=True)


@task(pre=[isort, fmt, lint, typecheck, test])
def check(c: Context) -> None:
    """Run all checks: isort, format, lint, typecheck, test."""


@task
def clean(c: Context) -> None:
    """Remove build artifacts and caches."""
    c.run("rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache")
    c.run("find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true")
