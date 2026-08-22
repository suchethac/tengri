"""Tests for the doc grammar keys guard."""

import pathlib
import subprocess
import sys

import pytest


def test_check_doc_grammar_keys_guard() -> None:
    """Run the guard on the live repository and ensure it passes.

    This test imports and runs tools/check_doc_grammar_keys.py::main()
    to verify that all grammar structural keys are properly documented
    and that documentation doesn't refer to non-existent keys.
    """
    repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
    guard_script = repo_root / "tools" / "check_doc_grammar_keys.py"

    assert guard_script.exists(), f"Guard script not found at {guard_script}"

    # Run the guard as a subprocess to avoid import order issues
    result = subprocess.run(
        [sys.executable, str(guard_script)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        pytest.fail(f"Grammar documentation guard failed:\n{result.stdout}\n{result.stderr}")

    # If we got here, the guard passed
    assert result.returncode == 0
    assert "✓" in result.stdout or "complete" in result.stdout.lower()
