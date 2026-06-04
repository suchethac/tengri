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
triweight *smoother*, which averaged neighbours and smeared this torus's mid-IR
peak by ~30% median (the same peak-smear that moved slone_netzer to node-exact
interpolation). Monotone cubic keeps C¹-continuous gradients for HMC/geoVI
without overshooting on the nearest-neighbour-filled grid.

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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

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
    [(0, 0, 5), (3, 1, 5), (3, 2, 8), (6, 3, 10)],
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


def test_parameter_bounds():
    """Priors cover the CAT3D-Wind grid extent."""
    from tengri.components.agn.cat3d_torus_model import CAT3DTorus

    c = CAT3DTorus()
    assert (c.cos_inc.lo, c.cos_inc.hi) == (0.0, 1.0)
    assert (c.a_cat3d.lo, c.a_cat3d.hi) == (-2.5, -0.5)
    assert (c.fwd_cat3d.lo, c.fwd_cat3d.hi) == (0.0, 1.0)
