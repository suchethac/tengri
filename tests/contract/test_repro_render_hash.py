# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the render-freshness hash is stable, code-sensitive, prose-blind.

``scripts/render_reproduction_notebook.py`` stamps a rendered notebook with
``code_cell_source_sha256(source .py)``; ``tools/check_repro_render_fresh.py``
recomputes the same hash from the checked-out ``.py`` to decide whether that
stamp is stale. This pins the three properties the whole scheme depends on:
identical source hashes identically, a code-cell edit changes the hash, and a
markdown-only edit does not.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tools._repro_render_shared import code_cell_source_sha256, parse_percent_cells

pytestmark = pytest.mark.contract

_SOURCE = """\
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       format_name: percent
# ---

# %% [markdown]
# # A reproduction comparison
#
# Some introductory prose.

# %%
import numpy as np

x = np.array([1, 2, 3])

# %% [markdown]
# A second markdown cell explaining the next figure.

# %%
y = x + 1
print(y)
"""


def test_same_source_gives_same_hash(tmp_path: Path) -> None:
    """Hashing the identical source twice must be deterministic."""
    p1 = tmp_path / "a.py"
    p2 = tmp_path / "b.py"
    p1.write_text(_SOURCE, encoding="utf-8")
    p2.write_text(_SOURCE, encoding="utf-8")

    assert code_cell_source_sha256(p1) == code_cell_source_sha256(p2)


def test_code_cell_edit_changes_hash(tmp_path: Path) -> None:
    """Editing a code cell's source must change the hash."""
    edited = _SOURCE.replace("y = x + 1", "y = x + 2")
    assert edited != _SOURCE  # guard against a no-op replace

    original = tmp_path / "original.py"
    changed = tmp_path / "changed.py"
    original.write_text(_SOURCE, encoding="utf-8")
    changed.write_text(edited, encoding="utf-8")

    assert code_cell_source_sha256(original) != code_cell_source_sha256(changed)


def test_markdown_only_edit_does_not_change_hash(tmp_path: Path) -> None:
    """Editing prose in a markdown cell must NOT change the hash.

    A rendered notebook is not stale just because a caveat was reworded;
    only a code-cell edit invalidates the stamp.
    """
    edited = _SOURCE.replace(
        "Some introductory prose.",
        "A substantially longer and differently worded introduction, "
        "rewritten for clarity and with an extra sentence.",
    )
    assert edited != _SOURCE  # guard against a no-op replace

    original = tmp_path / "original.py"
    changed = tmp_path / "changed.py"
    original.write_text(_SOURCE, encoding="utf-8")
    changed.write_text(edited, encoding="utf-8")

    assert code_cell_source_sha256(original) == code_cell_source_sha256(changed)


def test_parse_percent_cells_classifies_markdown_and_code() -> None:
    """Sanity check on the underlying cell splitter the hash relies on."""
    cells = parse_percent_cells(_SOURCE)
    is_markdown_flags = [is_md for is_md, _ in cells]
    assert is_markdown_flags == [True, False, True, False]
    assert "import numpy as np" in cells[1][1]
    assert "y = x + 1" in cells[3][1]
