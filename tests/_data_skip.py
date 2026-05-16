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


def _sps_home_nenkova() -> Path | None:
    sps_home = os.environ.get("SPS_HOME")
    if not sps_home:
        return None
    return Path(sps_home) / "dust" / "Nenkova08_y010_torusg_n10_q2.0.dat"


_nenkova_path = _sps_home_nenkova()

requires_nenkova = pytest.mark.skipif(
    _nenkova_path is None or not _nenkova_path.is_file(),
    reason="Nenkova+2008 torus templates not found ($SPS_HOME unset or missing data)",
)
