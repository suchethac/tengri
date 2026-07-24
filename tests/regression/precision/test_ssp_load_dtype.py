# SPDX-License-Identifier: BSD-3-Clause
"""``load_ssp_data(dtype=...)`` — the opt-in for a 32-bit host grid (#1206).

The SSP flux cube is the model's largest single array. ``jax.enable_x64``
governs only JAX arrays, so a pure-float32 forward pass still loads the grid as
float64 numpy. ``dtype=jnp.float32`` halves that host footprint and makes the
pipeline 32-bit from disk to output. It is applied regardless of the x64 flag —
the point is to force 32-bit even when x64 is on — and is safe now that the
~1e42 ``stellar_mass_scale`` and ~1e56 ``nion`` scale seams are carried in log
space (before that, a float32 grid overflowed them silently: #1099).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

pytestmark = pytest.mark.regression_bug

_FLOAT_FIELDS = ("ssp_wave", "ssp_flux", "ssp_lg_age_gyr", "ssp_lgmet", "ssp_mass_remaining")


def _write_tiny_ssp(path):
    """A minimal but valid DSPS-format grid: a few ages, mets, wavelengths."""
    h5py = pytest.importorskip("h5py")
    n_age, n_met, n_wave = 4, 3, 32
    wave = np.linspace(1e3, 1e5, n_wave)
    # A per-Msun flux at a realistic magnitude so nothing is contrived.
    flux = np.full((n_met, n_age, n_wave), 1e-14)
    with h5py.File(path, "w") as f:
        f.create_dataset("ssp_wave", data=wave)
        f.create_dataset("ssp_flux", data=flux)
        f.create_dataset("ssp_lg_age_gyr", data=np.linspace(-3.0, 1.0, n_age))
        f.create_dataset("ssp_lgmet", data=np.linspace(-3.0, -1.5, n_met))
        # Include mass_remaining so the loader does not invoke the synthesizer.
        f.create_dataset("ssp_mass_remaining", data=np.full((n_met, n_age), 0.7))
    return path


@pytest.fixture
def tiny_ssp_file(tmp_path):
    return _write_tiny_ssp(tmp_path / "ssp_tiny_test.h5")


def test_dtype_float32_makes_every_float_array_float32(tiny_ssp_file):
    """Every float field — loaded and mass_remaining — comes back float32."""
    with jax.enable_x64(True):  # force the hard case: x64 on, dtype must still win
        ssp = load_ssp_data(str(tiny_ssp_file), dtype=jnp.float32)

    for field in _FLOAT_FIELDS:
        arr = getattr(ssp, field)
        assert arr is not None, f"setup: {field} missing"
        assert arr.dtype == jnp.float32, (
            f"{field} is {arr.dtype}, not float32 — dtype= did not reach it "
            "(a float64 field defeats the whole point: the grid is not halved)"
        )


def test_default_follows_working_precision(tiny_ssp_file):
    """Omitting dtype keeps the historical behavior: working precision (#1099)."""
    with jax.enable_x64(True):
        ssp64 = load_ssp_data(str(tiny_ssp_file))
    assert ssp64.ssp_flux.dtype == jnp.float64, (
        "default load must upcast to float64 under x64 — the #1099 guard against "
        "silent stellar_mass_scale overflow depends on it"
    )
    with jax.enable_x64(False):
        ssp32 = load_ssp_data(str(tiny_ssp_file))
    assert ssp32.ssp_flux.dtype == jnp.float32, "default under x64-off is float32"


def test_float32_grid_halves_the_flux_cube(tiny_ssp_file):
    """The host footprint of the flux cube is exactly halved — the memory point."""
    with jax.enable_x64(True):
        f64 = load_ssp_data(str(tiny_ssp_file))
        f32 = load_ssp_data(str(tiny_ssp_file), dtype=jnp.float32)
    b64 = int(np.asarray(f64.ssp_flux).nbytes)
    b32 = int(np.asarray(f32.ssp_flux).nbytes)
    assert b32 * 2 == b64, f"flux cube {b32} B (f32) is not half of {b64} B (f64)"


def test_float32_grid_preserves_values_to_float32_precision(tiny_ssp_file):
    """Casting is lossless to float32's own precision — no behavior change."""
    with jax.enable_x64(True):
        f64 = load_ssp_data(str(tiny_ssp_file))
        f32 = load_ssp_data(str(tiny_ssp_file), dtype=jnp.float32)
    ref = np.asarray(f64.ssp_flux, dtype=np.float64)
    got = np.asarray(f32.ssp_flux, dtype=np.float64)
    peak = float(np.abs(ref).max())
    assert peak > 0.0, "setup: flux is all zero"
    assert float(np.abs(got - ref).max() / peak) < 1e-6, (
        "float32 grid departs from float64 by more than float32 precision"
    )
