# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the Silva+04 smooth-torus component.

Grid templates come from AGNfitter (Calistro Rivera et al. 2016); see
``scripts/build_silva04_grid.py``. Tests skip cleanly when the grid HDF5 is
absent so the unit suite stays green on CI images without model data.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
import pytest

from tengri._data_setup import find_data
from tests._grad_parity import assert_grad_matches_fd
from tests._jit_parity import assert_jit_matches_eager

# parents[4] is one level above the repo root from tests/components/agn/, so
# this guard was permanently true and the tests below never ran (#1431).
_GRID_PATH = find_data("silva04_torus_grid.h5")
_has_grid = _GRID_PATH is not None

pytestmark = [
    pytest.mark.bounds,
    pytest.mark.skipif(
        not _has_grid,
        reason=(
            "Silva+04 grid not built. Run: "
            "python scripts/build_silva04_grid.py "
            "--input /tmp/AGNfitter/models/TORUS/S04.pickle"
        ),
    ),
]


@pytest.fixture(scope="module")
def wavelength() -> jnp.ndarray:
    return jnp.geomspace(1e3, 1e6, 512)


@pytest.fixture(scope="module")
def torus_fn():
    from tengri.components.agn.silva04 import create_silva04_from_grid

    return create_silva04_from_grid(str(_GRID_PATH))


def test_grid_metadata_sensible() -> None:
    import h5py

    with h5py.File(_GRID_PATH, "r") as f:
        g = f["silva04"]
        log_nh = g["log_nh_axis"][:]
        wave = g["wavelength"][:]
        tpl = g["template"][:]

    assert log_nh.ndim == 1 and log_nh.size >= 5
    assert jnp.all(log_nh[1:] > log_nh[:-1]), "log_nh_axis must be ascending"
    assert wave.ndim == 1 and wave.size >= 128
    assert jnp.all(wave[1:] > wave[:-1]), "wavelength must be ascending"
    chex.assert_shape(tpl, (log_nh.size, wave.size))
    # Silva+04 grid typically spans obscured column densities.
    assert float(log_nh.min()) >= 20.0
    assert float(log_nh.max()) <= 26.0


def test_output_shape_and_finiteness(torus_fn, wavelength) -> None:
    sed = torus_fn(
        wavelength,
        agn_log_lbol=44.0,
        agn_log_nh_silva=23.0,
        agn_torus_frac=0.5,
    )
    chex.assert_equal_shape([sed, wavelength])
    chex.assert_tree_all_finite(sed)
    assert float(sed.max()) > 0.0


def test_luminosity_scales_linearly_with_lbol(torus_fn, wavelength) -> None:
    sed_lo = torus_fn(wavelength, agn_log_lbol=44.0, agn_log_nh_silva=23.0)
    sed_hi = torus_fn(wavelength, agn_log_lbol=45.0, agn_log_nh_silva=23.0)
    mask = sed_lo > 0.0
    ratio = jnp.where(mask, sed_hi / jnp.where(mask, sed_lo, 1.0), 10.0)
    assert jnp.allclose(ratio[mask], 10.0, rtol=1e-5)


def test_torus_frac_scales_linearly(torus_fn, wavelength) -> None:
    sed_half = torus_fn(wavelength, agn_log_lbol=44.0, agn_torus_frac=0.5)
    sed_full = torus_fn(wavelength, agn_log_lbol=44.0, agn_torus_frac=1.0)
    mask = sed_half > 0.0
    ratio = jnp.where(mask, sed_full / jnp.where(mask, sed_half, 1.0), 2.0)
    assert jnp.allclose(ratio[mask], 2.0, rtol=1e-5)


def test_nh_axis_actually_interpolated(torus_fn, wavelength) -> None:
    """Interpolation must produce distinct SEDs for distinct log_N_H values."""
    sed_low = torus_fn(wavelength, agn_log_nh_silva=22.5)
    sed_high = torus_fn(wavelength, agn_log_nh_silva=24.5)
    diff = jnp.linalg.norm(sed_high - sed_low)
    norm = jnp.linalg.norm(sed_low) + 1e-300
    assert float(diff / norm) > 1e-3


def test_jit_compatible(torus_fn, wavelength) -> None:
    sed = assert_jit_matches_eager(lambda nh: torus_fn(wavelength, agn_log_nh_silva=nh), 23.0)
    chex.assert_tree_all_finite(sed)


def test_grad_flows_through_log_nh(torus_fn, wavelength) -> None:
    """Gradient must flow through ``agn_log_nh_silva`` (triweight is C²)."""

    def scalar_loss(nh: float) -> float:
        sed = torus_fn(wavelength, agn_log_nh_silva=nh)
        return jnp.log1p(jnp.sum(sed))

    g = assert_grad_matches_fd(scalar_loss, 23.0)
    assert jnp.isfinite(g)
    assert jnp.any(g != 0.0), (
        "`g` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )


def test_unified_dispatch_registered() -> None:
    import warnings

    from tengri.components.agn.unified import resolve_agn_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        fn = resolve_agn_model("silva04")
    assert callable(fn)
    wl = jnp.geomspace(1e3, 1e6, 128)
    sed = fn(wl, agn_log_lbol=44.0, agn_lum_ratio=0.1, agn_log_nh_silva=23.0)
    chex.assert_equal_shape([sed, wl])
    assert float(sed.max()) > 0.0
