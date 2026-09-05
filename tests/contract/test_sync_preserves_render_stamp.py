# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the docs-sync transplant preserves a render stamp, never fabricates one.

``scripts/sync_reproduction_notebooks_for_docs.py`` refreshes a rendered
notebook's cell *source* from its jupytext ``.py`` without re-executing
anything, so a prose-only edit can reach ``docs/reproduction/`` without
paying for a real render (which the CI environment cannot do for most
comparisons). ``metadata["tengri_render"]`` -- the stamp
``scripts/render_reproduction_notebook.py`` writes -- must survive that
transplant exactly, and the sync must never manufacture one itself: doing so
would claim a fresh execution the sync never performed, and
``tools/check_repro_render_fresh.py`` would then report a stale render as
fresh.

``transplant_source`` only ever assigns ``cell.source`` on the existing
notebook object and writes it back -- it never touches notebook-level
``metadata`` -- so this pins that property directly rather than trusting it
by inspection.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output
from scripts.sync_reproduction_notebooks_for_docs import transplant_source

pytestmark = pytest.mark.contract

_PY_SOURCE_TEMPLATE = """\
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       format_name: percent
# ---

# %% [markdown]
# {prose}

# %%
print("hello")
"""


def _make_rendered_notebook(*, stamp: dict | None) -> nbformat.NotebookNode:
    """A minimal, already-rendered notebook: one markdown cell, one figure cell."""
    md = new_markdown_cell(source="Original prose.")
    code = new_code_cell(source='print("hello")')
    code.outputs = [
        new_output(
            "display_data",
            data={"image/png": "not-a-real-png-but-non-empty"},
        )
    ]
    nb = new_notebook(cells=[md, code])
    if stamp is not None:
        nb.metadata["tengri_render"] = stamp
    return nb


def test_sync_preserves_an_existing_stamp_exactly(tmp_path: Path) -> None:
    """A stamp already on the rendered ``.ipynb`` must survive a sync untouched."""
    stamp = {
        "source_sha256": "deadbeef" * 8,
        "executed_at": "2026-01-01T00:00:00+00:00",
        "tengri_version": "0.1.0",
    }
    ipynb_path = tmp_path / "01_mini.ipynb"
    nbformat.write(_make_rendered_notebook(stamp=stamp), str(ipynb_path))

    py_path = tmp_path / "01_mini.py"
    py_path.write_text(
        _PY_SOURCE_TEMPLATE.format(prose="Reworded prose -- a text-only edit."),
        encoding="utf-8",
    )

    transplant_source(py_path, ipynb_path)

    result = nbformat.read(str(ipynb_path), as_version=4)
    assert result.metadata.get("tengri_render") == stamp, (
        "a prose-only sync must not alter an existing render stamp"
    )
    assert "Reworded prose" in result.cells[0].source, (
        "the sync should still have refreshed the cell source"
    )


def test_sync_never_fabricates_a_stamp(tmp_path: Path) -> None:
    """A never-stamped rendered ``.ipynb`` must stay unstamped after a sync."""
    ipynb_path = tmp_path / "01_mini.ipynb"
    nbformat.write(_make_rendered_notebook(stamp=None), str(ipynb_path))

    py_path = tmp_path / "01_mini.py"
    py_path.write_text(
        _PY_SOURCE_TEMPLATE.format(prose="Some other reworded prose."),
        encoding="utf-8",
    )

    transplant_source(py_path, ipynb_path)

    result = nbformat.read(str(ipynb_path), as_version=4)
    assert "tengri_render" not in result.metadata, (
        "the sync must never invent a render stamp -- it never re-executes "
        "anything, so claiming one would be a false freshness signal"
    )
