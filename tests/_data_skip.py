# SPDX-License-Identifier: BSD-3-Clause
"""Shared skip markers for tests that need optional data bundles.

Used by tests that must skip on CI runners lacking an optional HDF5 grid.

Why the paths live here
-----------------------
``REPO_ROOT`` is computed once, in a module whose own depth is fixed. Every
test file that spelled ``Path(__file__).resolve().parents[N] / "data"`` for
itself had to get ``N`` right for its own depth, and five did not:

* ``tests/components/agn/test_agn_ebv_disc.py`` and
  ``test_agn_ebv_disc_all_models.py`` used ``parents[4]`` where the root is
  ``parents[3]``
* ``tests/contract/test_agn_precompute_kernel_consumer.py``,
  ``test_agn_torus_alt_precompute.py`` and ``test_nebular_shock_kernel_wiring.py``
  used ``parents[4]`` where the root is ``parents[2]`` -- two levels out

Every one resolved above the repository, so ``not (_DATA / grid).exists()`` was
permanently true and **19 tests never ran**, while the grids they wanted were
present in ``data/`` all along. #1431 is the same bug in
``test_agn_cat3d_wind.py``, which still carries the comment explaining it.

A skip that can never be lifted is indistinguishable from a passing test in
every report CI produces. Import a marker from here rather than recomputing a
root.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

#: Optional grids, and the marker that skips when one is absent.
SILVA04_GRID = DATA_DIR / "silva04_torus_grid.h5"
CAT3D_WIND_GRID = DATA_DIR / "cat3d_wind_torus_grid.h5"
MAPPINGS_TEMPLATES = DATA_DIR / "mappings_templates.h5"
SKIRTOR_TEMPLATES = ("skirtor_templates_v2.h5", "skirtor_templates_v3.h5")

requires_silva04 = pytest.mark.skipif(
    not SILVA04_GRID.is_file(),
    reason=f"Silva+04 torus grid not found at {SILVA04_GRID} "
    "(build via scripts/build_silva04_grid.py)",
)

requires_cat3d_wind = pytest.mark.skipif(
    not CAT3D_WIND_GRID.is_file(),
    reason=f"CAT3D-Wind torus grid not found at {CAT3D_WIND_GRID} "
    "(build via scripts/build_cat3d_wind_grid.py)",
)

requires_mappings = pytest.mark.skipif(
    not MAPPINGS_TEMPLATES.is_file(),
    reason=f"MAPPINGS V template grid not found at {MAPPINGS_TEMPLATES} "
    "(run scripts/download_mappings_templates.py)",
)


def has_skirtor() -> bool:
    """True when either SKIRTOR template bundle is present."""
    return any((DATA_DIR / name).is_file() for name in SKIRTOR_TEMPLATES)


GRAHSP_TEMPLATE = DATA_DIR / "grahsp" / "grahsp_templates.h5"

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
    vendored = DATA_DIR / "nenkova08_torus_grid.h5"
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
