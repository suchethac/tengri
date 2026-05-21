"""Tests for the _display helper TENGRI_QUIET semantics."""

from __future__ import annotations

import pytest

from tengri._display import _display


@pytest.mark.contract
def test_tengri_quiet_suppresses_only_when_exactly_one(capsys, monkeypatch):
    """`TENGRI_QUIET=1` silences; any other value (or unset) prints.

    Single test covers the full state space: default-on, explicit-off,
    explicit-on. The print-mechanics themselves (newline handling,
    multiline) are stdlib `print()` behavior, not ours to retest.
    """
    # Default: prints.
    monkeypatch.delenv("TENGRI_QUIET", raising=False)
    _display("hello")
    assert capsys.readouterr().out == "hello\n"

    # Explicit non-"1": prints.
    monkeypatch.setenv("TENGRI_QUIET", "0")
    _display("hello")
    assert capsys.readouterr().out == "hello\n"

    # Exactly "1": silenced.
    monkeypatch.setenv("TENGRI_QUIET", "1")
    _display("hello")
    assert capsys.readouterr().out == ""
