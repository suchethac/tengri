# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate tengri's SKIRTOR_mean_3p torus against the AGNfitter-rX grid.

AGNfitter-rX ships ``models/TORUS/SKIRTOR_mean_3p.pickle`` — the Stalevski+2016
SKIRTOR library *averaged over the clumpiness (p, q) and radial parameters*,
leaving (oa, incl, tv). It was converted ONCE to the committed HDF5
``data/skirtor_mean3p_torus_grid.h5`` by ``scripts/build_skirtor_mean3p_grid.py``.
This test reads only that HDF5 (no pickle, no AGNfitter driver) so it runs in CI
rather than skipping (#613). The grid's stored node templates ARE the AGNfitter
reference; the test checks that the runtime component reproduces them.

Key parity result (#614 / #592 B1): the AGNfitter-averaged torus peaks at ~25 µm,
*not* the ~40 µm of tengri's full-grid X-CIGALE ``skirtor`` — the two are
intentionally different reductions of the same model, and this committed block
gives the AGNfitter-matched one.

Interpolation note: the runtime uses node-exact monotone-cubic (PCHIP)
interpolation (``interp_nd_pchip``, shared with the cat3d / slone_netzer components),
so at a grid node it reproduces the stored AGNfitter template to floating-point
precision while keeping C¹-continuous gradients for HMC/geoVI. This replaced the
C²-smooth triweight *smoother*, which mixed in neighbors (~10% peak-normalized
shape residual).

References
----------
.. [1] Stalevski et al. 2016, MNRAS, 458, 2288. arXiv:1602.01954.
.. [2] Martinez-Ramirez et al. 2024, A&A, 688, A46 (AGNfitter-rX). arXiv:2405.12111.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.crossval

_GRID = Path(__file__).resolve().parents[2] / "data" / "skirtor_mean3p_torus_grid.h5"

if not _GRID.is_file():
    pytest.skip(
        f"SKIRTOR_mean_3p grid not found at {_GRID} "
        "(build with: python scripts/build_skirtor_mean3p_grid.py)",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def grid() -> dict:
    import h5py

    with h5py.File(_GRID, "r") as f:
        g = f["skirtor_mean3p"]
        return {
            "oa": np.asarray(g["oa_axis"][:], dtype=np.float64),
            "incl": np.asarray(g["incl_axis"][:], dtype=np.float64),
            "tv": np.asarray(g["tv_axis"][:], dtype=np.float64),
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "template": np.asarray(g["template"][:], dtype=np.float64),
        }


@pytest.fixture(scope="module")
def component():
    from tengri.components.agn.skirtor_agnfitter import create_skirtor_agnfitter_from_grid

    return create_skirtor_agnfitter_from_grid(str(_GRID))


def _peak_norm(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return x / np.nanmax(x[mask])


def test_peak_is_agnfitter_averaged_not_full_grid(grid, component):
    """At (oa=40, incl=30, tv=7) the torus peaks at ~25 µm (AGNfitter-averaged)."""
    wave = grid["wavelength"]
    out = np.asarray(
        component(
            wavelength=jnp.asarray(wave),
            agn_log_lbol=0.0,
            agn_oa_skirtor=40.0,
            agn_incl_skirtor=30.0,
            agn_tv_skirtor=7.0,
            agn_torus_frac=1.0,
        )
    )
    peak_um = wave[np.nanargmax(out)] / 1e4
    assert 20.0 < peak_um < 30.0, (
        f"SKIRTOR_mean_3p peak {peak_um:.1f} µm outside 20-30 µm "
        "(expected ~25 µm AGNfitter-averaged; ~40 µm would mean the full-grid skirtor)."
    )


@pytest.mark.parametrize("i_oa,j_incl,k_tv", [(0, 3, 2), (3, 3, 2), (5, 6, 0), (7, 9, 4)])
def test_node_shape_matches_grid(grid, component, i_oa, j_incl, k_tv):
    """Node-exact PCHIP reproduces the stored AGNfitter template to the float level."""
    wave = grid["wavelength"]
    ref = grid["template"][i_oa, j_incl, k_tv]
    out = np.asarray(
        component(
            wavelength=jnp.asarray(wave),
            agn_log_lbol=0.0,
            agn_oa_skirtor=float(grid["oa"][i_oa]),
            agn_incl_skirtor=float(grid["incl"][j_incl]),
            agn_tv_skirtor=float(grid["tv"][k_tv]),
            agn_torus_frac=1.0,
        )
    )
    mask = (ref > ref.max() * 1e-3) & (out > 0)
    ref_n = _peak_norm(ref, mask)
    out_n = _peak_norm(out, mask)

    # Node-exact: peak coincides and shape matches to ~1e-3.
    assert int(np.nanargmax(ref)) == int(np.nanargmax(out))

    worst = float(np.nanmax(np.abs(out_n[mask] - ref_n[mask]) / ref_n[mask]))
    assert worst < 1e-3, (
        f"Node (oa={grid['oa'][i_oa]}, incl={grid['incl'][j_incl]}, tv={grid['tv'][k_tv]}): "
        f"shape residual {worst:.2e} > 1e-3 (PCHIP should be node-exact)"
    )


def test_parameter_bounds():
    """Priors match the SKIRTOR_mean_3p grid extent."""
    from tengri.components.agn.skirtor_agnfitter_model import SKIRTORAgnfitterTorus

    c = SKIRTORAgnfitterTorus()
    assert (c.oa_skirtor.lo, c.oa_skirtor.hi) == (10.0, 80.0)
    assert (c.incl_skirtor.lo, c.incl_skirtor.hi) == (0.0, 90.0)
    assert (c.tv_skirtor.lo, c.tv_skirtor.hi) == (3.0, 11.0)
