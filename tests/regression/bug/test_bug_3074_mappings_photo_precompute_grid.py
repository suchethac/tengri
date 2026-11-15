# SPDX-License-Identifier: BSD-3-Clause
"""The MAPPINGS V stellar precompute reads its grid by field (#3074).

``_load_stellar_grid`` returned a bare tuple and ``precompute`` read
``.line_wavelengths`` on it, so every call raised ``AttributeError``. The lookup
that ``build_lookup`` builds from the same tables had three more defects: the
zeta_O exponent took ``log10`` of the negative ``log10(Z_sun)`` (NaN for every
input), the redshift-like weight along zeta_O dropped out of the sum, and the
SFH row was a hard-coded index instead of the grid's ``cont`` row.

The reference below is an independent multilinear interpolation, so the lookup
is checked against the exact per-line formula rather than against itself.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from tengri.components.nebular import mappings_photo as mp, mappings_photo_precompute as mpp
from tengri.components.nebular._constants import _LOG10_ZSUN, _LSUN_ERG

pytestmark = pytest.mark.regression_bug

_REAL_GRID = Path(mp._DEFAULT_GRID_PATH)
_FILTER_WAVES = [np.linspace(3000.0, 9000.0, 400)]
_FILTER_TRANS = [np.ones(400)]


def _exact_lines(hb_table, ratio_table, axes, point, log_qh):
    """Exact per-line luminosity [Lsun] at ``point`` from multilinear interpolation.

    ``hb_table`` has shape (N_z, N_a, N_u, N_n) and ``ratio_table`` has the same
    leading shape plus a line axis; ``point`` is (zeta_O, log_age, logU, logn).
    """
    hb = RegularGridInterpolator(axes, hb_table)(point)[0]
    ratios = RegularGridInterpolator(axes, ratio_table)(point)[0]
    return ratios * 10.0 ** (hb + log_qh) / _LSUN_ERG


def _finite_cell_midpoint(axes, hb_table, ratio_table):
    """Midpoint of the first grid cell whose 16 corners are finite in every table."""
    finite = np.isfinite(hb_table) & np.all(np.isfinite(ratio_table), axis=-1)
    for idx in itertools.product(*(range(len(ax) - 1) for ax in axes)):
        corner = tuple(slice(i, i + 2) for i in idx)
        if finite[corner].all():
            return tuple(0.5 * (ax[i] + ax[i + 1]) for ax, i in zip(axes, idx))
    raise AssertionError("no fully finite grid cell in the cont row")


@pytest.mark.skipif(not _REAL_GRID.is_file(), reason="flury2024_grids.h5 not present")
def test_precompute_on_the_real_grid_matches_exact_lines():
    """The real sb99 cpr grid precomputes, and its lookup matches the exact per-line sum."""
    loaded = mp._load_stellar_grid(_REAL_GRID, "sb99", "cpr")
    grid = loaded.grid
    cont = loaded.sfh_idx_cont
    zo = np.asarray(grid.zo_axis, dtype=np.float64)
    ages = np.asarray(grid.log_age_yr_axis, dtype=np.float64)
    us = np.asarray(grid.logU_axis, dtype=np.float64)
    ns = np.asarray(grid.logn_axis, dtype=np.float64)
    hb_table = np.asarray(grid.logHB_per_logq, dtype=np.float64)[:, :, cont]
    ratio_table = np.asarray(grid.line_ratios, dtype=np.float64)[:, :, cont]
    axes = (zo, ages, us, ns)
    point = _finite_cell_midpoint(axes, hb_table, ratio_table)

    preint = mpp.precompute(_FILTER_WAVES, _FILTER_TRANS, 0.0, None, grid_path=_REAL_GRID)
    lookup = mpp.build_lookup(preint)
    log_qh = 50.0
    log_z_abs = np.log10(point[0]) + _LOG10_ZSUN
    _, lum = lookup["predict_lines"](log_qh, log_z_abs, *point[1:], neb_fesc=0.0)

    exact = _exact_lines(hb_table, ratio_table, axes, np.asarray(point)[None, :], log_qh)
    np.testing.assert_allclose(np.asarray(lum), exact, rtol=1e-5)
    assert np.all(np.isfinite(np.asarray(lum)))


def _write_linear_stellar_grid(path: Path) -> dict:
    """Write a sb99/cpr group whose tables are linear in every axis, so interpolation is exact.

    The ``inst`` row is a constant that differs from the ``cont`` row, so a lookup that
    picks the wrong SFH row fails the comparison.
    """
    zo = np.array([0.1, 0.5, 1.0, 2.0], dtype=np.float32)
    ages = np.array([6.0, 7.0, 8.0], dtype=np.float32)
    us = np.array([-4.0, -3.0, -2.0], dtype=np.float32)
    ns = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    zz, aa, uu, nn = np.meshgrid(zo, ages, us, ns, indexing="ij")
    cont_hb = (0.1 * zz + 0.2 * aa + 0.3 * uu + 0.4 * nn - 12.0).astype(np.float32)
    inst_hb = np.full_like(cont_hb, 5.0)
    logHB = np.stack([cont_hb, inst_hb], axis=2)  # (N_z, N_a, N_s, N_u, N_n)
    ratio_cont = np.stack(
        [1.0 + 0.5 * zz + 0.1 * aa + 0.2 * uu + 0.3 * nn, 2.0 + 0.2 * zz + 0.4 * aa],
        axis=-1,
    ).astype(np.float32)
    ratio_inst = np.full_like(ratio_cont, 9.0)
    ratios = np.stack([ratio_cont, ratio_inst], axis=2)  # (..., N_s, N_u, N_n, N_lines)
    waves = np.array([5007.0, 6563.0], dtype=np.float32)
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = mp.FLURY_SCHEMA_VERSION
        grp = f.create_group("sb99/cpr")
        grp.create_dataset("z_axis", data=zo)
        grp.create_dataset("log_age_yr_axis", data=ages)
        grp.create_dataset("logU_axis", data=us)
        grp.create_dataset("logn_axis", data=ns)
        grp.create_dataset("sfh_labels", data=np.array(["cont", "inst"], dtype="S8"))
        grp.create_dataset("logHB_per_logq", data=logHB)
        grp.create_dataset("line_ratios", data=ratios)
        grp.create_dataset("line_wavelengths_aa", data=waves)
        grp.create_dataset("line_names", data=np.array(["O3_5007A", "H1r_6563A"], dtype="S40"))
    return {"zo": zo, "ages": ages, "us": us, "ns": ns}


def test_loader_returns_named_fields(tmp_path):
    """``_load_stellar_grid`` returns one typed object whose fields are read by name."""
    path = tmp_path / "linear.h5"
    _write_linear_stellar_grid(path)
    loaded = mp._load_stellar_grid(path, "sb99", "cpr")
    assert isinstance(loaded, mp.MappingsStellarLoad)
    assert loaded.sfh_labels == ["cont", "inst"]
    assert (loaded.sfh_idx_cont, loaded.sfh_idx_inst) == (0, 1)
    assert loaded.grid.logHB_per_logq.shape == (4, 3, 2, 3, 3)


def test_precompute_on_a_synthetic_grid_matches_exact_lines(tmp_path):
    """The precompute runs on a small grid, and the lookup reproduces the linear tables exactly."""
    path = tmp_path / "linear.h5"
    _write_linear_stellar_grid(path)
    preint = mpp.precompute(_FILTER_WAVES, _FILTER_TRANS, 0.0, None, grid_path=path)
    lookup = mpp.build_lookup(preint)

    zo_val, age, logu, logn = 0.7, 6.5, -3.4, 2.6
    log_qh = 48.0
    log_z_abs = np.log10(zo_val) + _LOG10_ZSUN
    waves, lum = lookup["predict_lines"](log_qh, log_z_abs, age, logu, logn, neb_fesc=0.0)

    hb = 0.1 * zo_val + 0.2 * age + 0.3 * logu + 0.4 * logn - 12.0
    ratios = np.array(
        [
            1.0 + 0.5 * zo_val + 0.1 * age + 0.2 * logu + 0.3 * logn,
            2.0 + 0.2 * zo_val + 0.4 * age,
        ]
    )
    expected = ratios * 10.0 ** (hb + log_qh) / _LSUN_ERG

    grid_waves = mp._load_stellar_grid(path, "sb99", "cpr").grid.line_wavelengths
    np.testing.assert_allclose(np.asarray(waves), np.asarray(grid_waves), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(lum), expected, rtol=1e-5)
