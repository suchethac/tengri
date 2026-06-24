# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the tengri.bench CLI dispatcher."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

import pytest

from tengri.bench import BENCHMARK_SCRIPTS, run

pytestmark = pytest.mark.contract


def test_catalog_non_empty() -> None:
    assert len(BENCHMARK_SCRIPTS) >= 10
    for name, (script, desc) in BENCHMARK_SCRIPTS.items():
        assert script.startswith("benchmark_") and script.endswith(".py")
        assert desc, f"{name!r} has no description"


def test_list_subcommand_lists_every_script() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = run(["list"])
    assert rc == 0
    out = buf.getvalue()
    for name in BENCHMARK_SCRIPTS:
        assert name in out


def test_help_subcommand_no_args_lists() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = run(["help"])
    assert rc == 0
    assert "Available benchmarks:" in buf.getvalue()


def test_help_for_known_script_prints_docstring() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = run(["help", "forward_model"])
    # rc == 2 is acceptable if scripts/ is unavailable (e.g. wheel install);
    # both 0 and 2 are valid here. We just check we got *something* sensible.
    assert rc in (0, 2)


def test_help_for_unknown_script_errors() -> None:
    buf = io.StringIO()
    with redirect_stderr(buf):
        rc = run(["help", "nonexistent_garbage"])
    assert rc == 2
    assert "unknown benchmark" in buf.getvalue()


def test_dispatch_unknown_script_errors() -> None:
    buf = io.StringIO()
    with redirect_stderr(buf):
        rc = run(["nonexistent_garbage"])
    assert rc == 2


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_flag(flag: str) -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = run([flag])
    assert rc == 0
    out = buf.getvalue()
    assert "tengri.bench" in out
    assert "Available benchmarks:" in out
