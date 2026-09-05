# SPDX-License-Identifier: BSD-3-Clause
"""Shared, stdlib-only hash of a reproduction notebook's code cells.

``scripts/render_reproduction_notebook.py`` stamps a rendered notebook with
the SHA-256 of the code (not markdown) cells in its jupytext ``.py`` source;
``tools/check_repro_render_fresh.py`` recomputes the same hash to decide
whether a committed render is stale. Both call :func:`code_cell_source_sha256`
so the two can never drift apart from each other — if the render script used
one hashing rule and the freshness guard another, the guard would either
false-positive on every fresh render or false-negative on every stale one.

This module is intentionally stdlib-only (no ``jupytext`` import): the
freshness guard runs in the ``lint``-tier CI job alongside
``tools/check_repro_status.py``, which installs nothing but ``ruff`` and this
repo's own tools. Parsing jupytext's percent format directly, rather than
round-tripping through the ``jupytext`` package, is what keeps that job cheap.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

# A jupytext percent-format cell marker: "# %%" optionally followed by a cell
# tag (e.g. "# %% [markdown]") and/or cell metadata. Only the markdown tag
# matters here -- anything else (a title, a "{"tags": [...]}" json blob) is
# still a code cell for hashing purposes.
_CELL_MARKER = re.compile(r"^# %%(.*)$")


def parse_percent_cells(source: str) -> list[tuple[bool, str]]:
    """Split a jupytext percent-format ``.py`` source into ``(is_markdown, text)`` cells.

    Parameters
    ----------
    source : str
        The full text of a jupytext percent-format ``.py`` file.

    Returns
    -------
    list of (bool, str)
        One entry per cell, in file order: whether the cell is markdown, and
        its body text (everything between this cell marker and the next).
        Content before the first ``# %%`` marker (e.g. jupytext's own header)
        is discarded -- it is not a cell.
    """
    cells: list[tuple[bool, list[str]]] = []
    current_is_md = False
    current_lines: list[str] = []
    started = False

    for line in source.splitlines(keepends=True):
        match = _CELL_MARKER.match(line)
        if match is not None:
            if started:
                cells.append((current_is_md, current_lines))
            current_is_md = "[markdown]" in match.group(1)
            current_lines = []
            started = True
            continue
        if started:
            current_lines.append(line)

    if started:
        cells.append((current_is_md, current_lines))

    return [(is_md, "".join(lines)) for is_md, lines in cells]


def code_cell_source_sha256(py_path: Path) -> str:
    """SHA-256 of the ordered code-cell sources in a jupytext percent-format file.

    Markdown cells are excluded: a prose-only edit (fixing a typo, rewording a
    caveat) must not register as a render-invalidating change, only an edit to
    a code cell's actual source can.

    Parameters
    ----------
    py_path : Path
        Path to the ``01_<slug>.py`` jupytext source.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest over the concatenated code-cell sources,
        each followed by a NUL separator (so cell boundaries cannot collide:
        ``["ab", "c"]`` and ``["a", "bc"]`` hash differently).
    """
    cells = parse_percent_cells(py_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    for is_markdown, text in cells:
        if is_markdown:
            continue
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
