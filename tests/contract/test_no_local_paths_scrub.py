# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the local-path scrub behind ``tools/check_no_local_paths.py --fix``.

The guard (#1816) detects absolute home paths in committed files. Detection
alone did not hold: #1749 re-executed the 11 published notebooks one commit
later and shipped 26 fresh copies of
``<checkout>/src/tengri/forward/sed_model.py:7796``, because Python's warning
format prints the absolute source path and the captured output is committed.
Three more came from a diagnostic cell printing a resolved ``REPO_ROOT`` from
inside a git worktree.

That second class is why the scrub is applied to captured *output* rather than
to the warning format: a ``print`` of a resolved path is not a warning, and no
``formatwarning`` override would ever see it.

Most of this file is about the two ways a scrub can be worse than no scrub:
rewriting a path it cannot reconstruct, and reaching into cell source.

Note the assembly of ``_HOME`` below. This file is itself a committed file that
the guard scans, so it must not contain a literal absolute home path -- the
tests would pass while the guard failed on the tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Layout: tests/contract/<this_file> -> repo root is 2 levels up.
_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / "tools"))
sys.path.insert(0, str(_root / "scripts"))

from check_no_local_paths import scrub_text
from execute_notebooks import scrub_outputs

pytestmark = pytest.mark.contract

#: Split so this source file never contains the literal path it tests against.
#: ``_HOME_PATH`` needs an alphanumeric after the slash, and here that position
#: holds a quote, so the guard does not match this line.
_HOME = "/Users/" + "someone"
_CHECKOUT = f"{_HOME}/Projects/tengri"
_WORKTREE = f"{_CHECKOUT}/.claude/worktrees/docs-notebook-audit"

#: The exact leak #1749 shipped, 26 times across four notebooks.
WARNING_LEAK = (
    f"{_CHECKOUT}/src/tengri/forward/sed_model.py:7796: "
    "WildcardPartialFreeWarning: 'all_params' is FREE"
)
WARNING_CLEAN = (
    "src/tengri/forward/sed_model.py:7796: WildcardPartialFreeWarning: 'all_params' is FREE"
)


def test_rewrites_a_repo_absolute_path() -> None:
    """The warning leak becomes a repository-relative path."""
    assert scrub_text(WARNING_LEAK) == WARNING_CLEAN


def test_rewrites_a_worktree_path() -> None:
    """A worktree resolves to the same tree, so it scrubs to the same relative form."""
    assert scrub_text(f"SSP grids : {_WORKTREE}/data") == "SSP grids : data"
    assert scrub_text(f"Repo root : {_WORKTREE}") == "Repo root : ."


def test_bare_checkout_becomes_dot() -> None:
    assert scrub_text(f"Repo root : {_CHECKOUT}") == "Repo root : ."


@pytest.mark.parametrize(
    "text",
    [
        # No `tengri` segment: the repo cannot reconstruct these, so rewriting
        # would invent information. Report, never rewrite.
        f"{_HOME}/datasets/grid.h5",
        "/home/" + "someone/scratch/run17/out.npz",
        # Sibling directories that merely start with the repo name. `\b` after
        # `/tengri` matched the first and produced `.-data/grid.h5`.
        f"{_HOME}/tengri-data/grid.h5",
        f"{_HOME}/Projects/tengri_old/notes.md",
        # CI runner homes are identical for everyone and documented on purpose,
        # so the scrub reuses the scanner's allowlist instead of its own copy.
        "/home/runner/work/tengri/tengri",
        "/home/ubuntu/tengri/data",
    ],
)
def test_leaves_paths_it_cannot_reconstruct_alone(text: str) -> None:
    assert scrub_text(text) == text


def test_is_idempotent() -> None:
    """Running --fix twice must not keep eating the path."""
    once = scrub_text(WARNING_LEAK)
    assert scrub_text(once) == once


def test_anchors_on_the_checkout_not_the_package() -> None:
    """``src/tengri`` after the checkout must survive.

    The leading segment walk is lazy so the FIRST ``tengri`` -- the checkout --
    anchors the match. A greedy walk would anchor on ``src/tengri`` and swallow
    ``src/`` with it.
    """
    assert scrub_text(WARNING_LEAK).startswith("src/tengri/")


def _notebook(output: dict) -> SimpleNamespace:
    """A minimal nbformat-shaped notebook: cells are dicts with ``.get``."""
    return SimpleNamespace(
        cells=[{"cell_type": "code", "source": f"# {WARNING_LEAK}\n", "outputs": [output]}]
    )


def test_scrubs_stream_traceback_and_mime_bundle() -> None:
    """All three places captured console text lands."""
    nb = _notebook(
        {
            "output_type": "stream",
            "text": [WARNING_LEAK + "\n"],
            "traceback": [WARNING_LEAK],
            "data": {"text/plain": WARNING_LEAK},
        }
    )
    assert scrub_outputs(nb) == 3
    out = nb.cells[0]["outputs"][0]
    assert out["text"] == [WARNING_CLEAN + "\n"]
    assert out["traceback"] == [WARNING_CLEAN]
    assert out["data"]["text/plain"] == WARNING_CLEAN


def test_never_touches_cell_source() -> None:
    """The invariant that makes a bulk rewrite safe.

    Cell source comes from ``notebooks/<slug>.py``. A scrub that reaches it has
    stopped redacting a machine path and started editing code -- so even a
    source line containing the very string being scrubbed must survive byte for
    byte.
    """
    nb = _notebook({"output_type": "stream", "text": WARNING_LEAK})
    before = nb.cells[0]["source"]
    scrub_outputs(nb)
    assert nb.cells[0]["source"] == before
    assert WARNING_LEAK in nb.cells[0]["source"]


def test_reports_zero_when_there_is_nothing_to_scrub() -> None:
    nb = _notebook({"output_type": "stream", "text": "all good\n"})
    assert scrub_outputs(nb) == 0
