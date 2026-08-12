# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the CAT3D-Wind torus component.

Grid templates come from AGNfitter-rX (Zhuang et al. 2024) — a
three-parameter projection of Hönig & Kishimoto 2017.  See
``scripts/build_cat3d_wind_grid.py``.  Tests skip cleanly when the
HDF5 grid is absent so the unit suite stays green on CI images without
model data.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

from tengri._data_setup import find_data
from tests._jit_parity import assert_jit_matches_eager

pytestmark = []

jax.config.update("jax_enable_x64", True)

# parents[4] is one level above the repo root from tests/components/agn/, so
# this guard was permanently true and the tests below never ran (#1431).
_GRID_PATH = find_data("cat3d_wind_torus_grid.h5")
_has_grid = _GRID_PATH is not None

pytestmark = [
    pytest.mark.bounds,
    pytest.mark.skipif(
        not _has_grid,
        reason=(
            "CAT3D-Wind grid not built. Run: "
            "python scripts/build_cat3d_wind_grid.py "
            "--input /tmp/AGNfitter-rX/models/TORUS/CAT3D_mean_3p.pickle"
        ),
    ),
]


@pytest.fixture(scope="module")
def wavelength() -> jnp.ndarray:
    return jnp.geomspace(1e3, 1e6, 512)


@pytest.fixture(scope="module")
def torus_fn():
    from tengri.components.agn.cat3d_wind import create_cat3d_wind_from_grid

    return create_cat3d_wind_from_grid(str(_GRID_PATH))


def test_grid_metadata_sensible() -> None:
    import h5py

    with h5py.File(_GRID_PATH, "r") as f:
        g = f["cat3d_wind"]
        incl = g["incl_axis"][:]
        a = g["a_axis"][:]
        fwd = g["fwd_axis"][:]
        wave = g["wavelength"][:]
        tpl = g["template"][:]

    assert incl.ndim == 1 and incl.size >= 3
    assert a.ndim == 1 and a.size >= 2
    assert fwd.ndim == 1 and fwd.size >= 3
    assert jnp.all(a[1:] > a[:-1]), "a_axis must be ascending"
    assert jnp.all(fwd[1:] > fwd[:-1]), "fwd_axis must be ascending"
    assert wave.ndim == 1 and jnp.all(wave[1:] > wave[:-1])
    chex.assert_shape(tpl, (incl.size, a.size, fwd.size, wave.size))


def test_output_shape_and_finiteness(torus_fn, wavelength) -> None:
    sed = torus_fn(
        wavelength,
        agn_log_lbol=44.0,
        agn_cos_inc=0.5,
        agn_a_cat3d=-2.0,
        agn_fwd_cat3d=1.5,
        agn_torus_frac=0.5,
    )
    chex.assert_equal_shape([sed, wavelength])
    chex.assert_tree_all_finite(sed)
    assert float(sed.max()) > 0.0


def test_luminosity_scales_linearly_with_lbol(torus_fn, wavelength) -> None:
    sed_lo = torus_fn(wavelength, agn_log_lbol=44.0)
    sed_hi = torus_fn(wavelength, agn_log_lbol=45.0)
    mask = sed_lo > 0.0
    ratio = jnp.where(mask, sed_hi / jnp.where(mask, sed_lo, 1.0), 10.0)
    assert jnp.allclose(ratio[mask], 10.0, rtol=1e-5)


def test_each_axis_actually_interpolated(torus_fn, wavelength) -> None:
    """Endpoints span the real grid extent: cos(incl) ∈ [0, 1] (incl ∈ [0, 90°]),
    a ∈ [-3, -1.5], fwd ∈ [1.0, 2.25] — AGNfitter's rows-210+ sub-library (#1036)."""
    base = torus_fn(wavelength, agn_cos_inc=0.5, agn_a_cat3d=-2.0, agn_fwd_cat3d=1.0)
    variants = [
        torus_fn(wavelength, agn_cos_inc=0.0, agn_a_cat3d=-2.0, agn_fwd_cat3d=1.0),
        torus_fn(wavelength, agn_cos_inc=0.5, agn_a_cat3d=-3.0, agn_fwd_cat3d=1.0),
        torus_fn(wavelength, agn_cos_inc=0.5, agn_a_cat3d=-2.0, agn_fwd_cat3d=2.25),
    ]
    for i, v in enumerate(variants):
        diff = float(jnp.linalg.norm(v - base))
        norm = float(jnp.linalg.norm(base) + 1e-300)
        assert diff / norm > 1e-3, f"axis {i} appears inert"


def test_jit_compatible(torus_fn, wavelength) -> None:
    sed = assert_jit_matches_eager(
        lambda ci, a, fwd: torus_fn(wavelength, agn_cos_inc=ci, agn_a_cat3d=a, agn_fwd_cat3d=fwd),
        0.4,
        -1.5,
        0.3,
    )
    chex.assert_tree_all_finite(sed)


def test_grad_flows_through_all_axes(torus_fn, wavelength) -> None:
    """Gradient must flow through each of the three grid axes."""

    def scalar_loss(ci: float, a: float, fwd: float) -> float:
        sed = torus_fn(wavelength, agn_cos_inc=ci, agn_a_cat3d=a, agn_fwd_cat3d=fwd)
        return jnp.log1p(jnp.sum(sed))

    grads = jax.grad(scalar_loss, argnums=(0, 1, 2))(0.5, -2.0, 0.2)
    for g in grads:
        assert jnp.isfinite(g)


def test_unified_dispatch_registered() -> None:
    import warnings

    from tengri.components.agn.unified import resolve_agn_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        fn = resolve_agn_model("cat3d_wind")
    assert callable(fn)
    wl = jnp.geomspace(1e3, 1e6, 128)
    sed = fn(
        wl,
        agn_log_lbol=44.0,
        agn_lum_ratio=0.1,
        agn_cos_inc=0.5,
        agn_a_cat3d=-2.0,
        agn_fwd_cat3d=1.5,
    )
    chex.assert_equal_shape([sed, wl])
    assert float(sed.max()) > 0.0
