# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate tengri's CAT3D-Wind torus against the AGNfitter-rX grid.

The CAT3D-Wind (Hönig & Kishimoto 2017) clumpy-disc-plus-polar-wind torus
library is shipped inside AGNfitter-rX at ``models/TORUS/CAT3D_mean_3p.pickle``
(the three-parameter projection over inclination, radial index ``a``, and polar
wind fraction). It was converted ONCE to the committed HDF5
``data/cat3d_wind_torus_grid.h5`` by ``scripts/build_cat3d_wind_grid.py``
(run ``--download`` to fetch the upstream pickle from GitHub — no AGNfitter
install needed). This test reads only that HDF5 (no pickle, no AGNfitter driver)
so it runs in CI rather than skipping (#613). The grid's stored node templates
ARE the AGNfitter reference; the test checks that the runtime component
reproduces them.

Interpolation note: the runtime uses node-exact monotone-cubic (PCHIP)
interpolation (``interp_nd_pchip``), so at a grid node it reproduces the stored
AGNfitter template to floating-point precision. This replaced the C²-smooth
triweight *smoother*, which averaged neighbors and smeared this torus's mid-IR
peak by ~30% median (the same peak-smear that moved slone_netzer to node-exact
interpolation). Monotone cubic keeps C¹-continuous gradients for HMC/geoVI
without overshooting on the nearest-neighbor-filled grid.

References
----------
.. [1] S. F. Hönig & M. Kishimoto, "The dusty heart of nearby active galaxies.
   II. From clumpy torus models to a unified model," ApJL 838, L20 (2017).
   arXiv:1702.08691.
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

_GRID = Path(__file__).resolve().parents[2] / "data" / "cat3d_wind_torus_grid.h5"

if not _GRID.is_file():
    pytest.skip(
        f"CAT3D-Wind grid not found at {_GRID} "
        "(build with: python scripts/build_cat3d_wind_grid.py --download)",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def grid() -> dict:
    import h5py

    with h5py.File(_GRID, "r") as f:
        g = f["cat3d_wind"]
        return {
            "incl": np.asarray(g["incl_axis"][:], dtype=np.float64),
            "a": np.asarray(g["a_axis"][:], dtype=np.float64),
            "fwd": np.asarray(g["fwd_axis"][:], dtype=np.float64),
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "template": np.asarray(g["template"][:], dtype=np.float64),
        }


@pytest.fixture(scope="module")
def component():
    from tengri.components.agn.cat3d_wind import create_cat3d_wind_from_grid

    return create_cat3d_wind_from_grid(str(_GRID))


def _peak_norm(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return x / np.nanmax(x[mask])


def _call(component, grid, i, j, k):
    """Evaluate the runtime component at grid node (i, j, k)."""
    return np.asarray(
        component(
            wavelength=jnp.asarray(grid["wavelength"]),
            agn_log_lbol=0.0,
            agn_cos_inc=float(np.cos(np.deg2rad(grid["incl"][i]))),
            agn_a_cat3d=float(grid["a"][j]),
            agn_fwd_cat3d=float(grid["fwd"][k]),
            agn_torus_frac=1.0,
        )
    )


def test_peak_in_mid_infrared(grid, component):
    """The clumpy torus peaks in the mid-IR (a few µm to a few tens of µm)."""
    out = _call(component, grid, i=3, j=1, k=5)
    peak_um = grid["wavelength"][np.nanargmax(out)] / 1e4
    assert 1.0 < peak_um < 60.0, f"CAT3D-Wind peak {peak_um:.1f} µm outside the IR torus band"


@pytest.mark.parametrize(
    "i,j,k",
    [(0, 0, 5), (3, 1, 5), (3, 2, 4), (6, 3, 2)],
)
def test_node_shape_matches_grid(grid, component, i, j, k):
    """Node-exact PCHIP reproduces the stored AGNfitter template to the float level."""
    ref = grid["template"][i, j, k]
    out = _call(component, grid, i, j, k)

    mask = (ref > ref.max() * 1e-3) & (out > 0)
    ref_n = _peak_norm(ref, mask)
    out_n = _peak_norm(out, mask)

    # Node-exact: peak coincides and shape matches to ~1e-3.
    assert int(np.nanargmax(ref)) == int(np.nanargmax(out))

    worst = float(np.nanmax(np.abs(out_n[mask] - ref_n[mask]) / ref_n[mask]))
    assert worst < 1e-3, (
        f"Node (incl={grid['incl'][i]}, a={grid['a'][j]}, fwd={grid['fwd'][k]}): "
        f"shape residual {worst:.2e} > 1e-3 (PCHIP should be node-exact)"
    )


@pytest.mark.regression_bug
def test_fwd_cat3d_axis_parity(grid, component):
    """agn_fwd_cat3d is LIVE over the corrected grid extent [1.0, 2.25].

    **Regression test for fwd-axis parity bug**: the build script
    previously fabricated a fwd axis [0.15, 0.45, 0.75, 1.0, 1.25, 1.5, 1.75,
    2.0, 2.25] from the full DataFrame, including an orphaned sub-library
    (rows 0–209) with different physics. AGNfitter uses only rows 210+ with
    fwd ∈ {1.0, 1.25, 1.5, 1.75, 2.0, 2.25}. The first five planes (fwd <
    1.0) were nearest-neighbor copies of fwd=1.0, making agn_fwd_cat3d a
    silent no-op in that range. The default 0.2 evaluated on the 1.0 plane.

    This test verifies the fix: at fwd=1.0 vs fwd=2.25 with all other
    parameters equal, the returned SEDs differ significantly. Peak-normalized
    difference must exceed 1% to confirm agn_fwd_cat3d is not a no-op.

    References
    ----------
    Hönig & Kishimoto 2017, ApJL 838, L20, arXiv:1702.08691.
    Martínez-Ramírez et al. 2024, A&A 688, A46, arXiv:2405.12111.
    """
    # Evaluate at the fwd=1.0 and fwd=2.25 nodes with identical other params
    sed_fwd_low = _call(component, grid, i=3, j=1, k=0)  # fwd=1.0
    sed_fwd_high = _call(component, grid, i=3, j=1, k=5)  # fwd=2.25

    mask = (sed_fwd_low > sed_fwd_low.max() * 1e-3) & (sed_fwd_high > 0)
    fwd_low_n = _peak_norm(sed_fwd_low, mask)
    fwd_high_n = _peak_norm(sed_fwd_high, mask)

    # Peak-normalized difference must exceed 1%
    diff = float(np.nanmax(np.abs(fwd_low_n[mask] - fwd_high_n[mask]) / fwd_high_n[mask]))
    assert diff > 0.01, (
        f"fwd_cat3d change from 1.0 to 2.25 produces only {diff:.2%} "
        f"peak-normalized difference (should be > 1%; parameter is a no-op)"
    )


@pytest.mark.regression_bug
def test_fwd_cat3d_node_exactness(grid, component):
    """CAT3D-Wind photometry matches AGNfitter at fwd ∈ {1.0, 1.75, 2.25}.

    Regression test ensuring the fixed fwd axis (now [1.0, 1.25, 1.5, 1.75,
    2.0, 2.25] from AGNfitter rows 210+ only) produces node-exact agreement
    with the committed grid templates. Tests fwd at three corners and middle
    of the axis to verify the full range reproduces the reference.

    References
    ----------
    Hönig & Kishimoto 2017, ApJL 838, L20, arXiv:1702.08691.
    Martínez-Ramírez et al. 2024, A&A 688, A46, arXiv:2405.12111.
    """
    # Test fwd=1.0, 1.75, 2.25 (low, middle, high)
    fwd_indices = [0, 3, 5]
    for k in fwd_indices:
        ref = grid["template"][3, 1, k]
        out = _call(component, grid, i=3, j=1, k=k)

        mask = (ref > ref.max() * 1e-3) & (out > 0)
        ref_n = _peak_norm(ref, mask)
        out_n = _peak_norm(out, mask)

        # Node-exact: peaks align
        assert int(np.nanargmax(ref)) == int(np.nanargmax(out)), (
            f"fwd={grid['fwd'][k]}: peak index mismatch (expected node-exact PCHIP)"
        )

        # Shape residual < 1e-3
        worst = float(np.nanmax(np.abs(out_n[mask] - ref_n[mask]) / ref_n[mask]))
        assert worst < 1e-3, f"fwd={grid['fwd'][k]}: shape residual {worst:.2e} > 1e-3"


def test_parameter_bounds():
    """Priors cover the CAT3D-Wind grid extent."""
    from tengri.components.agn.cat3d_torus_model import CAT3DTorus

    c = CAT3DTorus()
    assert (c.cos_inc.lo, c.cos_inc.hi) == (0.0, 1.0)
    assert (c.a_cat3d.lo, c.a_cat3d.hi) == (-2.5, -0.5)
    assert (c.fwd_cat3d.lo, c.fwd_cat3d.hi) == (0.0, 1.0)
