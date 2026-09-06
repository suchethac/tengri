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
* ``tests/physics/gradients/test_nebular_gradients.py`` used ``parents[2]``
  where the root is ``parents[3]``, resolving to ``tests/data/`` -- a directory
  that has never existed. All eight of its data-gated tests skipped on every
  machine, CI included, and two of them (Cue, MAPPINGS) want grids that are
  *tracked in git* and therefore present on every runner.

Every one resolved to a directory with no grids in it, so
``not (_DATA / grid).exists()`` was permanently true and **27 tests never ran**,
while the grids they wanted were present in ``data/`` all along. #1431 is the
same bug in ``test_agn_cat3d_wind.py``, which still carries the comment
explaining it.

The failure is silent in both directions: too shallow lands in ``tests/``, too
deep lands above the repository, and neither raises. Only the skip count moves,
and nothing reads it.

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

#: Basename of the Draine+2021 PAHspec grid, built by
#: ``scripts/build_pahspec_hdf5.py``. At 104 MB it is not committed, so no CI
#: runner has a copy.
PAHSPEC_GRID_NAME = "pahspec_draine2021.h5"

#: Both grammar spellings of the one PAHspec component: ``draine2021_pah`` is
#: an alias that resolves to the ``draine2021_pah_ir`` registry key. A gate
#: written against one spelling silently leaves the other ungated.
PAHSPEC_EMISSION_TYPES = frozenset({"draine2021_pah", "draine2021_pah_ir"})


def pahspec_grid_path() -> str | None:
    """The PAHspec grid the Draine+2021 component would load, or ``None``.

    Returns
    -------
    str or None
        Path to the grid, or ``None`` when the component would find nothing.

    Notes
    -----
    Unlike every other grid in this module, this one is **not** gated on
    ``DATA_DIR / name``. ``Draine2021PAHIRSEDComponent.load`` reads
    ``$TENGRI_PAHSPEC_PATH`` first and otherwise calls
    :func:`tengri._data_setup.find_data_str`, whose search walks the ancestors
    of the working directory. Those two locators disagree whenever the checkout
    is a git worktree -- the grid sits in the parent checkout, the component
    loads it, and a ``DATA_DIR``-gated test would skip a run that would have
    passed. Asking the locator the component asks keeps the gate and the
    behavior it guards on the same answer.
    """
    from tengri._data_setup import find_data_str
    from tengri.components.dust.draine2021_pah import PAHSPEC_PATH_ENV

    override = os.environ.get(PAHSPEC_PATH_ENV)
    if override is not None:
        return override if Path(override).is_file() else None
    return find_data_str(PAHSPEC_GRID_NAME)


def has_pahspec() -> bool:
    """True when the Draine+2021 PAHspec grid is loadable on this machine."""
    return pahspec_grid_path() is not None


requires_pahspec = pytest.mark.skipif(
    not has_pahspec(),
    reason=f"Draine+2021 PAHspec grid ({PAHSPEC_GRID_NAME}) not found. Absent it "
    "the component warns and contributes nothing -- the designed response to "
    "missing data (#1278), so its emission cannot be measured here. Build it "
    "with scripts/build_pahspec_hdf5.py or point $TENGRI_PAHSPEC_PATH at a copy.",
)

#: Photoionization grids and weights for the nebular backends.
CLOUDY_GRID_MIST = DATA_DIR / "cloudy_grid_mist.h5"
CUE_WEIGHTS = DATA_DIR / "cue_weights.npz"
CB19_TEMPLATES = DATA_DIR / "cb19_templates.h5"

requires_cloudy_grid = pytest.mark.skipif(
    not CLOUDY_GRID_MIST.is_file(),
    reason=f"CLOUDY MIST grid not found at {CLOUDY_GRID_MIST} "
    "(run scripts/convert_fsps_cloudy_grid.py)",
)

requires_cue_weights = pytest.mark.skipif(
    not CUE_WEIGHTS.is_file(),
    reason=f"Cue NN weights not found at {CUE_WEIGHTS} (tracked in git -- a "
    "missing file here means an incomplete checkout, not an optional bundle)",
)

requires_cb19 = pytest.mark.skipif(
    not CB19_TEMPLATES.is_file(),
    reason=f"CB_19 photoionization grid not found at {CB19_TEMPLATES} "
    "(run scripts/download_cb19_templates.py)",
)


def _cb19_log_u_axis_is_degenerate() -> bool:
    """True when the CB19 grid does not vary along ``log_U``.

    ``tests/conftest.py::pytest_configure`` writes a synthetic CB19 grid when
    the real one is absent, built by broadcasting ten Case B Hbeta ratios across
    the full 7-D shape. It is therefore *constant* along every physical axis by
    construction -- its own comment says "not production-grade, but enough to
    break the degeneracy in plumbing tests".

    Structural tests are fine on it. A gradient test is not: d/d_logU of a
    constant is zero, and comparing that zero against a finite-difference zero
    passes for any implementation. This answers "is there a logU signal here to
    differentiate", which is the actual precondition, rather than guessing
    whether the file on disk is the real download.
    """
    if not CB19_TEMPLATES.is_file():
        return True
    try:
        import h5py
        import numpy as np

        with h5py.File(CB19_TEMPLATES, "r") as f:
            grid = f["grids/SSP/Kroupa01/mu100/line_ratios"]
            # axes: (log_OH, log_age, log_U, log_nH, log_CO, dNO, HbFrac, line)
            return bool(np.ptp(np.asarray(grid[()]), axis=2).max() == 0.0)
    except (OSError, KeyError, ImportError):
        return True


requires_nondegenerate_cb19 = pytest.mark.skipif(
    _cb19_log_u_axis_is_degenerate(),
    reason=f"the CB_19 grid at {CB19_TEMPLATES} is constant along log_U, so a "
    "gradient with respect to neb_logU is identically zero and cannot be "
    "checked against a finite difference. This is the synthetic fixture "
    "tests/conftest.py writes when the real grid is missing -- run "
    "scripts/download_cb19_templates.py for a grid with a live log_U axis.",
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
