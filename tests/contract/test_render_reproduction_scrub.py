# SPDX-License-Identifier: BSD-3-Clause
"""Tests that ``scripts/render_reproduction_notebook.py`` cannot ship a machine path.

``tools/check_no_local_paths.py`` detects absolute home paths in committed
files; ``scripts/execute_notebooks.py`` has scrubbed its own writes since
#1816. The reproduction renderer did not, and the gap is not hypothetical: a
deprecation banner prints the absolute ``__file__`` of the module that raised
it, six such paths sat in the committed reproduction renders, and every render
produced from a worktree put them straight back. Re-rendering could not clear
them, because re-rendering is what creates them.

The banners are deliberately unfiltered -- each notebook's prose explains why
the deprecated model is the intended comparison target -- so the fix belongs at
the write, not at the warning.

Two invariants, the same two that govern the spine executor:

* Cell ``source`` is never touched. Source comes from the jupytext ``.py``, and
  a scrub that reached it would be editing code rather than redacting a path.
* The scrub runs before the file is written, so the stamped notebook and the
  ``docs/reproduction/`` copy taken from it are both clean. Checking the bytes
  on disk, rather than the scrub in isolation, is what makes this fail if the
  call is ever dropped from ``render_slug``.

Note the assembly of ``_HOME``: this file is itself scanned by the guard, so it
must not contain a literal absolute home path, or the tests would pass while
the guard failed on the tests.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Layout: tests/contract/<this_file> -> repo root is 2 levels up.
_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / "scripts"))
sys.path.insert(0, str(_root / "tools"))

import render_reproduction_notebook as rrn
from check_no_local_paths import _ALLOWED_PREFIXES, _HOME_PATH

pytestmark = pytest.mark.contract

#: Split so this source file never contains the literal path it tests against.
#: ``_HOME_PATH`` needs an alphanumeric after the slash; here that position
#: holds a quote, so the guard does not match this line.
_HOME = "/Users/" + "someone"

#: The exact shape of the six leaks this closes: a warning banner naming the
#: module that raised it, from a render run inside a git worktree.
WORKTREE_LEAK = (
    f"{_HOME}/Projects/tengri/.claude/worktrees/audit-example/src/tengri/components/agn/"
    "component.py:452: DeprecationWarning: AGN model 'skirtor_stalevski' is deprecated\n"
)

SLUG = "example"


def _home_path_hits(text: str) -> list[str]:
    """Every absolute home path the guard would reject in ``text``."""
    return [
        m.group(0)
        for m in _HOME_PATH.finditer(text)
        if not any(text.startswith(p, m.start()) for p in _ALLOWED_PREFIXES)
    ]


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A miniature repository holding one already-executed reproduction render.

    The notebook is pre-made rather than executed: this pins the *write* path
    of ``render_slug``, and running a real kernel would test nbconvert instead.
    """
    slug_dir = tmp_path / "reproduction" / SLUG
    slug_dir.mkdir(parents=True)

    (slug_dir / f"01_{SLUG}.py").write_text(
        "# %% [markdown]\n# A tiny reproduction.\n\n# %%\nprint('hello')\n",
        encoding="utf-8",
    )

    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "source": "print('hello')\n",
                "outputs": [
                    {"output_type": "stream", "name": "stderr", "text": WORKTREE_LEAK},
                    # A path inside the checkout being rendered, which the scrub
                    # makes repo-relative rather than redacting to "~/".
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": f"{tmp_path}/data/ssp_grid.h5 loaded\n",
                    },
                ],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (slug_dir / f"01_{SLUG}.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return tmp_path


@pytest.fixture
def rendered(tree: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """``render_slug`` run over :func:`tree` with jupytext/nbconvert stubbed out."""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    return rrn.render_slug(SLUG, root=tree)


def test_the_render_it_writes_carries_no_home_path(rendered: Path) -> None:
    """The contract: whatever the renderer writes, the guard must accept."""
    hits = _home_path_hits(rendered.read_text(encoding="utf-8"))
    assert not hits, f"the render script wrote {len(hits)} home path(s): {hits}"


def test_the_published_docs_copy_carries_no_home_path(tree: Path, rendered: Path) -> None:
    """``publish_slug`` copies bytes, so the twin is clean only if the source is."""
    docs_ipynb = rrn.publish_slug(SLUG, rendered, root=tree)
    hits = _home_path_hits(docs_ipynb.read_text(encoding="utf-8"))
    assert not hits, f"the published docs twin carries {len(hits)} home path(s): {hits}"


def test_a_worktree_render_reads_as_a_repo_relative_path(rendered: Path) -> None:
    """The leak becomes the path a main-checkout render would have printed."""
    nb = json.loads(rendered.read_text(encoding="utf-8"))
    text = nb["cells"][0]["outputs"][0]["text"]
    assert text.startswith("src/tengri/components/agn/component.py:452: DeprecationWarning")


def test_the_checkout_being_rendered_becomes_relative(rendered: Path) -> None:
    nb = json.loads(rendered.read_text(encoding="utf-8"))
    assert nb["cells"][0]["outputs"][1]["text"] == "data/ssp_grid.h5 loaded\n"


def test_cell_source_survives_byte_for_byte(rendered: Path) -> None:
    nb = json.loads(rendered.read_text(encoding="utf-8"))
    assert nb["cells"][0]["source"] == "print('hello')\n"


def test_the_render_is_still_stamped(rendered: Path) -> None:
    """The scrub must not cost the freshness stamp the guard reads."""
    nb = json.loads(rendered.read_text(encoding="utf-8"))
    assert set(nb["metadata"]["tengri_render"]) == {
        "source_sha256",
        "executed_at",
        "tengri_version",
    }


def test_the_stamp_does_not_depend_on_the_scrub(tree: Path) -> None:
    """Why scrubbing before the stamp is safe, pinned rather than asserted in prose.

    ``source_sha256`` hashes the jupytext ``.py``'s code cells, never the
    ``.ipynb`` bytes, so redacting an output cannot make a fresh render read as
    stale. If that ever changes, this fails and the ordering in ``render_slug``
    has to be revisited.
    """
    from _repro_render_shared import code_cell_source_sha256

    py_path = tree / "reproduction" / SLUG / f"01_{SLUG}.py"
    before = code_cell_source_sha256(py_path)

    nb = json.loads(
        (tree / "reproduction" / SLUG / f"01_{SLUG}.ipynb").read_text(encoding="utf-8")
    )
    rrn.strip_local_paths(nb, root=tree)

    assert code_cell_source_sha256(py_path) == before
