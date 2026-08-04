# SPDX-License-Identifier: BSD-3-Clause
"""Notebooks must resolve data and figure paths through one seam, not by hand.

A notebook that writes ``Path("../data")`` or ``Path("_figs")`` is correct only
while the working directory happens to be ``notebooks/``. Run it from the
repository root, from a scheduler, or from sphinx-gallery -- which ``chdir``s
into each script's directory before exec -- and the SSP lookup misses and falls
through to a 67 MB download, while figures scatter into a stray directory. That
is the #1486 failure class, and it fails *open*: nothing raises, the notebook
just does something else.

tengri already ships the general resolver. :func:`tengri.data_path` and
:func:`tengri.load_ssp` walk every ancestor of the working directory for a
``data/`` subdirectory, honor ``$TENGRI_DATA_DIR``, and fall back to the
package's own source root, so they answer the same regardless of where the
process started (#1431). ``notebooks/_setup.FIG_DIR`` anchors figure output the
same way.

These tests pin the *rule*, not the instances: any notebook re-introducing a
hand-rolled anchor fails here, including one written next year. Fixing the
twelve notebooks that carried the idiom without this guard would only have reset
the clock.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = ROOT / "notebooks"

#: Each entry is ``(compiled pattern, what to use instead)``. The message is the
#: whole value of the test: a reader who trips it needs the replacement, not a
#: restatement of the regex.
FORBIDDEN = (
    (
        # Anchored at the opening quote, so prose like "put it under data/" is
        # not flagged -- only a literal that is itself used as a path.
        re.compile(r"""["'](?:\.\./)?data/"""),
        'cwd-relative data path. Use tengri.load_ssp("<alias>", download=True), which '
        "honors $TENGRI_DATA_DIR, walks ancestors for data/, and fetches on miss. "
        "Catching this at the string, not at the call, is deliberate: "
        "05_adding_a_model.py passed one through a variable and slipped a narrower rule.",
    ),
    (
        re.compile(r"""FIG_DIR\s*=\s*Path\(\s*["']_figs["']\s*\)"""),
        "cwd-relative figure directory. Use `from _setup import FIG_DIR`, which is anchored.",
    ),
    (
        re.compile(r"""pyproject\.toml"""),
        "hand-rolled repository-root walk. Use `from _setup import REPO_ROOT` (anchored on "
        "that module's location), or tengri.data_path()/load_ssp() for data files. A "
        "marker-file walk additionally cannot work from an installed wheel, which has no "
        "pyproject.toml.",
    ),
    (
        re.compile(r"""find_spec\(\s*["']tengri["']"""),
        "walking outward from the installed package to find the repository. "
        "Use `from _setup import REPO_ROOT`.",
    ),
    (
        re.compile(r"""^\s*os\.chdir\(""", re.M),
        "changing the working directory to make a relative path resolve. It fixes one "
        "path by breaking every other one in the process, including the render pipeline's. "
        "Resolve the path instead.",
    ),
)


def _notebook_sources() -> list[Path]:
    """Every jupytext notebook source, excluding the shared helper modules."""
    return sorted(p for p in NOTEBOOK_DIR.glob("*.py") if not p.name.startswith("_"))


def test_the_notebook_directory_is_where_we_think_it_is():
    """Guard the guard: an empty glob would make every test below vacuous."""
    sources = _notebook_sources()
    assert len(sources) >= 10, f"expected the notebook spine, found {len(sources)} sources"


@pytest.mark.parametrize("py_path", _notebook_sources(), ids=lambda p: p.stem)
def test_notebook_does_not_hand_roll_a_path_anchor(py_path):
    """No notebook resolves data or figure paths relative to the working directory."""
    text = py_path.read_text(encoding="utf-8")
    offenses = [
        f"  line {n}: {line.strip()}\n    -> {advice}"
        for pattern, advice in FORBIDDEN
        for n, line in enumerate(text.splitlines(), start=1)
        if pattern.search(line)
    ]
    assert not offenses, f"{py_path.name} anchors paths by hand:\n" + "\n".join(offenses)


def test_shared_figure_directory_is_absolute():
    """``_setup.FIG_DIR`` must not depend on where the process started."""
    import sys

    sys.path.insert(0, str(NOTEBOOK_DIR))
    try:
        from _setup import FIG_DIR
    finally:
        sys.path.remove(str(NOTEBOOK_DIR))
    assert FIG_DIR.is_absolute(), f"FIG_DIR is cwd-relative: {FIG_DIR}"
    assert FIG_DIR.name == "_figs", f"unexpected figure directory: {FIG_DIR}"
