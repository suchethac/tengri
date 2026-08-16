# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate tengri's Slone & Netzer (2012) disc against the AGNfitter-rX grid.

The SN12 accretion-disc library (tabulated over black-hole mass and Eddington
ratio) is shipped inside AGNfitter-rX at ``models/BBB/SN12.pickle``. It was
converted ONCE to the committed HDF5 ``data/slone_netzer_disc_grid.h5`` by
``scripts/build_slone_netzer_grid.py`` (run ``--download`` to fetch the upstream
pickle from GitHub — no AGNfitter install needed). This test reads only that
HDF5 (no pickle, no AGNfitter driver) so it runs in CI rather than skipping
(#613). The grid's stored node templates ARE the AGNfitter reference; the test
checks that the runtime component reproduces them.

Tolerance note: unlike the triweight-based torus components, the SN12 runtime uses
**node-exact bilinear** interpolation over ``(log M_BH, log Edd)`` (the disc
peak wavelength varies strongly with accretion rate, so a smooth kernel would
smear it). At a grid node bilinear returns the stored template exactly, so the
shape residual here is at the floating-point level, not a kernel budget.

References
----------
.. [1] A. Slone & H. Netzer, "The effect of disc winds on the structure and
   spectrum of accretion discs," MNRAS 426, 656 (2012). arXiv:1207.5077.
.. [2] L. N. Martínez-Ramírez et al., "AGNfitter-rx: Modeling the radio-to-X-ray
   spectral energy distributions of AGNs," A&A 688, A46 (2024).
   arXiv:2405.12111. DOI: 10.1051/0004-6361/202449329.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.crossval

_GRID = Path(__file__).resolve().parents[2] / "data" / "slone_netzer_disc_grid.h5"

if not _GRID.is_file():
    pytest.skip(
        f"Slone & Netzer grid not found at {_GRID} "
        "(build with: python scripts/build_slone_netzer_grid.py --download)",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def grid() -> dict:
    import h5py

    with h5py.File(_GRID, "r") as f:
        g = f["slone_netzer"]
        return {
            "log_mbh": np.asarray(g["log_mbh"][:], dtype=np.float64),
            "log_edd": np.asarray(g["log_edd"][:], dtype=np.float64),
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "template": np.asarray(g["template"][:], dtype=np.float64),
        }


@pytest.fixture(scope="module")
def component():
    from tengri.components.agn.slone_netzer import create_slone_netzer_from_grid

    return create_slone_netzer_from_grid(str(_GRID))


def _peak_norm(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return x / np.nanmax(x[mask])


def _call(component, grid, i, j):
    """Evaluate the runtime component at grid node (log_mbh[i], log_edd[j])."""
    return np.asarray(
        component(
            wavelength=jnp.asarray(grid["wavelength"]),
            agn_log_lbol=0.0,
            agn_log_mbh=float(grid["log_mbh"][i]),
            agn_log_ledd=float(grid["log_edd"][j]),
        )
    )


def test_peak_is_uv_optical(grid, component):
    """A standard accretion disc peaks in the UV/optical (< 1 µm)."""
    out = _call(component, grid, i=grid["log_mbh"].size // 2, j=grid["log_edd"].size // 2)
    peak_um = grid["wavelength"][np.nanargmax(out)] / 1e4
    assert peak_um < 1.0, f"SN12 disc peak {peak_um:.3f} µm is not UV/optical"


@pytest.mark.parametrize("i,j", [(0, 0), (4, 6), (8, 11), (2, 9)])
def test_node_shape_matches_grid(grid, component, i, j):
    """Node-exact bilinear reproduces the stored template to the float level."""
    ref = grid["template"][i, j]
    out = _call(component, grid, i, j)

    mask = (ref > ref.max() * 1e-3) & (out > 0)
    ref_n = _peak_norm(ref, mask)
    out_n = _peak_norm(out, mask)

    # Bilinear is node-exact: peak coincides and shape matches to ~1e-3.
    assert int(np.nanargmax(ref)) == int(np.nanargmax(out))

    worst = float(np.nanmax(np.abs(out_n[mask] - ref_n[mask]) / ref_n[mask]))
    assert worst < 1e-3, (
        f"Node (log_mbh={grid['log_mbh'][i]:.2f}, log_edd={grid['log_edd'][j]:.2f}): "
        f"shape residual {worst:.2e} > 1e-3 (bilinear should be node-exact)"
    )


def test_grid_axis_extent(grid):
    """The committed grid spans the expected SN12 (M_BH, Edd) extent."""
    assert grid["log_mbh"].size == 9
    assert grid["log_edd"].size == 12
    assert grid["log_mbh"].min() < grid["log_mbh"].max()
    assert grid["log_edd"].min() < grid["log_edd"].max()
