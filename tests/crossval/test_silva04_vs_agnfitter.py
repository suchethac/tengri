# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate tengri's Silva+04 torus against the AGNfitter grid.

The Silva et al. 2004 obscured-torus templates (parameterized by hydrogen
column density ``N_H``) are shipped inside AGNfitter as
``models/TORUS/S04.pickle``. They were converted ONCE to the committed HDF5
``data/silva04_torus_grid.h5`` by ``scripts/build_silva04_grid.py`` (run
``--download`` to fetch the upstream pickle from GitHub — no AGNfitter install
needed). This test reads only that HDF5 (no pickle, no AGNfitter driver) so it
runs in CI rather than skipping (#613). The grid's stored node templates ARE the
AGNfitter reference; the test checks that the runtime component reproduces them.

Interpolation note: the runtime uses node-exact monotone-cubic (PCHIP)
interpolation (``interp_nd_pchip``) over the single ``log N_H`` axis, so at a
grid node it reproduces the stored AGNfitter template to floating-point
precision. This replaced the C²-smooth triweight *smoother*, which averaged
neighboring nodes and smeared the boundary-node shape by ~9% (the same
peak-smear class that moved ``cat3d_wind``/``nenkova_agnfitter`` to node-exact
interpolation). Monotone cubic keeps C¹-continuous gradients for HMC/geoVI
without overshooting on the sparse (5-bin) grid.

References
----------
.. [1] L. Silva, A. Maiolino & G. L. Granato, "Tracing the active galactic
   nucleus component in the infrared-bright galaxy NGC 1068...," MNRAS 355,
   973 (2004). arXiv:astro-ph/0403468.
.. [2] L. N. Martínez-Ramírez et al., "AGNfitter-rx: Modeling the radio-to-X-ray
   spectral energy distributions of AGNs," A&A 688, A46 (2024).
   arXiv:2405.12111. DOI: 10.1051/0004-6361/202449329.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.crossval

_GRID = Path(__file__).resolve().parents[2] / "data" / "silva04_torus_grid.h5"

if not _GRID.is_file():
    pytest.skip(
        f"Silva+04 grid not found at {_GRID} "
        "(build with: python scripts/build_silva04_grid.py --download)",
        allow_module_level=True,
    )

with h5py.File(_GRID, "r") as _f:
    _N_LOG_NH: int = _f["silva04"]["log_nh_axis"].shape[0]

#: Explicit node indices, not a frac -> round(frac * (n - 1)) mapping: the
#: latter never lands on the last node for any frac < 1 (e.g. frac=0.99 on
#: n=60 rounds 58.41 -> 58, one short of index 59), so the worst-behaved edge
#: node was never exercised. See task-1-brief.md item 3.
_NODE_INDICES = [0, _N_LOG_NH // 4, _N_LOG_NH // 2, 3 * _N_LOG_NH // 4, _N_LOG_NH - 1]


@pytest.fixture(scope="module")
def grid() -> dict:
    with h5py.File(_GRID, "r") as f:
        g = f["silva04"]
        return {
            "log_nh": np.asarray(g["log_nh_axis"][:], dtype=np.float64),
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "template": np.asarray(g["template"][:], dtype=np.float64),
        }


@pytest.fixture(scope="module")
def component():
    from tengri.components.agn.silva04 import create_silva04_from_grid

    return create_silva04_from_grid(str(_GRID))


def _peak_norm(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return x / np.nanmax(x[mask])


def _call(component, grid, i):
    """Evaluate the runtime component at grid node ``log_nh[i]``."""
    return np.asarray(
        component(
            wavelength=jnp.asarray(grid["wavelength"]),
            agn_log_lbol=0.0,
            agn_log_nh_silva=float(grid["log_nh"][i]),
            agn_torus_frac=1.0,
        )
    )


def test_peak_in_infrared(grid, component):
    """The reprocessed-dust torus peaks in the IR for a representative N_H."""
    i = grid["log_nh"].size // 2
    out = _call(component, grid, i)
    peak_um = grid["wavelength"][np.nanargmax(out)] / 1e4
    assert 1.0 < peak_um < 200.0, f"Silva+04 peak {peak_um:.1f} µm outside the IR torus band"


@pytest.mark.parametrize("i", _NODE_INDICES)
def test_node_shape_matches_grid(grid, component, i):
    """Runtime component reproduces the stored node template shape (PCHIP node-exact budget).

    Includes the two boundary nodes (``i=0`` and ``i=_N_LOG_NH - 1``), the
    tightest case: the triweight smoother this replaced mixed in a single
    neighbor there (no node on the far side), giving the largest smear (~9%,
    torus.md D2).
    """
    ref = grid["template"][i]
    out = _call(component, grid, i)

    mask = (ref > ref.max() * 1e-3) & (out > 0)
    ref_n = _peak_norm(ref, mask)
    out_n = _peak_norm(out, mask)

    assert abs(np.nanargmax(ref) - np.nanargmax(out)) <= 2

    worst = float(np.nanmax(np.abs(out_n[mask] - ref_n[mask]) / ref_n[mask]))
    assert worst < 1.0e-3, (
        f"Node log_nh={grid['log_nh'][i]:.2f}: shape residual {worst * 100:.4f}% > 0.1% "
        "(PCHIP node-exact budget; floating-point noise only)"
    )


def test_parameter_bounds(grid):
    """Priors cover the Silva+04 grid extent, read from the h5 axis (not a literal)."""
    from tengri.components.agn.silva04_model import Silva04Torus

    c = Silva04Torus()
    lo, hi = float(grid["log_nh"].min()), float(grid["log_nh"].max())
    assert (c.log_nh_silva.lo, c.log_nh_silva.hi) == (lo, hi)
