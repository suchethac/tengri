# SPDX-License-Identifier: BSD-3-Clause
"""Shared skip markers for tests that need optional data bundles.

Used by tests/unit/components/agn/ to skip on CI runners that lack the
GRAHSP HDF5 bundle or FSPS install. Mirrors the pattern already used for
MAPPINGS / SKIRTOR / Cue h5 grids elsewhere in this tree.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

GRAHSP_TEMPLATE = REPO_ROOT / "data" / "grahsp" / "grahsp_templates.h5"

requires_grahsp = pytest.mark.skipif(
    not GRAHSP_TEMPLATE.is_file(),
    reason=f"GRAHSP template bundle not found at {GRAHSP_TEMPLATE} "
    "(run `python tools/build_grahsp_hdf5.py` locally to enable these tests)",
)


def _nenkova_grid() -> Path | None:
    """Locate the Nenkova+2008 CLUMPY torus grid.

    Prefers the vendored grid committed to ``data/`` (so CI exercises the
    torus); falls back to the raw FSPS ``.dat`` via ``$SPS_HOME`` only if the
    vendored grid is absent.
    """
    vendored = REPO_ROOT / "data" / "nenkova08_torus_grid.h5"
    if vendored.is_file():
        return vendored
    sps_home = os.environ.get("SPS_HOME")
    if sps_home:
        dat = Path(sps_home) / "dust" / "Nenkova08_y010_torusg_n10_q2.0.dat"
        if dat.is_file():
            return dat
    return None


_nenkova_path = _nenkova_grid()

requires_nenkova = pytest.mark.skipif(
    _nenkova_path is None,
    reason="Nenkova+2008 torus grid not found (run scripts/build_nenkova_grid.py)",
)
