# SPDX-License-Identifier: BSD-3-Clause
"""Shared constants and helpers for NLR block modules."""

from __future__ import annotations

import os
from pathlib import Path

_C_AA_PER_S: float = 2.99792458e18


def _resolve_synthesizer_grid(kind: str) -> str:
    """Resolve a Synthesizer AGN grid path for the grid-backed line blocks.

    Searches ``$TENGRI_SYNTHESIZER_AGN_GRID_DIR`` then the repo-default
    ``data/synthesizer_grids/``. ``kind`` is ``"nlr"`` or ``"blr"``.

    These grids are not packaged with tengri (they ship via
    ``synthesizer-download --agn-test-grids``), so a clear error is raised if
    neither location holds ``test_grid_agn-<kind>.hdf5``.
    """
    fname = f"test_grid_agn-{kind}.hdf5"
    candidates = []
    env = os.environ.get("TENGRI_SYNTHESIZER_AGN_GRID_DIR")
    if env:
        candidates.append(Path(env) / fname)
    candidates.append(Path("data/synthesizer_grids") / fname)
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        f"Synthesizer AGN {kind.upper()} grid '{fname}' not found. Set "
        "$TENGRI_SYNTHESIZER_AGN_GRID_DIR or place it in data/synthesizer_grids/ "
        "(fetch via `synthesizer-download --agn-test-grids`)."
    )
