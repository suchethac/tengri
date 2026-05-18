"""Snapshot regression: predict_photometry/spectrum output across modes.

Pins the numerical output of every prediction mode under a fixed param
dict so that ADR-0004's kernel-strategy refactor cannot accidentally
re-route a call (e.g. compositional → hybrid) without breaking these
tests. Compositional ≡ exact bit-identical (closure-A); hybrid has a
documented 0.5% stellar tolerance — we compare each mode to itself
across runs, not across modes.

The first run writes the npz under ``tests/_data/kernel_snapshots/``.
Subsequent runs assert ``allclose(rtol=0, atol=0)`` — any drift fails.
Re-baseline with::

    TENGRI_REBASELINE_KERNEL_SNAPSHOTS=1 pytest \
        tests/integration/test_kernel_output_snapshots.py

All tests skip gracefully when the SSP data file is missing.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.forward.sed_model import SEDModel
from tengri.observation.filters import load_filter_set
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Uniform

# ── Skip guard ────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_SNAP_DIR = Path(__file__).resolve().parents[1] / "_data" / "kernel_snapshots"

pytestmark = pytest.mark.skipif(
    not _SSP_FILE.is_file(),
    reason=f"SSP data file not found: {_SSP_FILE}",
)

_FILTER_NAMES = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
_REBASELINE = bool(os.environ.get("TENGRI_REBASELINE_KERNEL_SNAPSHOTS"))


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ssp_data():
    return load_ssp_data(str(_SSP_FILE))


@pytest.fixture(scope="module")
def filters():
    return load_filter_set(_FILTER_NAMES)


@pytest.fixture(scope="module")
def model(ssp_data, filters):
    """Star-forming photometry model at fixed z=0.05."""
    spec = Parameters(
        mean_sfh_type="tsnorm",
        sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5.0),
        sfh_tsnorm_skew=Uniform(-1.0, 1.0),
        sfh_tsnorm_trunc=Uniform(1.0, 10.0),
        met_logzsol=Uniform(-2.0, 0.2),
        dust_tau_bc=Uniform(0.0, 3.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        dust_slope=-0.7,
        redshift=0.05,
    )
    return SEDModel(spec=spec, ssp_data=ssp_data, filters=filters)


@pytest.fixture(scope="module")
def fixed_params():
    """Stable param dict — same numbers every time."""
    return {
        "sfh_tsnorm_log_peak_sfr": 0.7,
        "sfh_tsnorm_peak_lbt_gyr": 3.0,
        "sfh_tsnorm_width_gyr": 1.5,
        "sfh_tsnorm_skew": 0.2,
        "sfh_tsnorm_trunc": 3.0,
        "met_logzsol": -0.3,
        "dust_tau_bc": 0.4,
        "dust_tau_diff": 0.2,
    }


# ── Snapshot helpers ──────────────────────────────────────────────


def _snap_path(name: str) -> Path:
    return _SNAP_DIR / f"{name}.npy"


def _assert_snapshot(name: str, arr) -> None:
    """Write on rebaseline, compare otherwise."""
    arr = np.asarray(arr)
    path = _snap_path(name)
    if _REBASELINE or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
        if not _REBASELINE:
            pytest.skip(f"baseline written: {path.relative_to(_SNAP_DIR.parent.parent)}")
        return
    expected = np.load(path)
    assert arr.shape == expected.shape, (
        f"snapshot {name} shape changed: {arr.shape} vs {expected.shape}"
    )
    # rtol=0, atol=0 — bit-identical across runs of the same code path.
    np.testing.assert_array_equal(arr, expected, err_msg=f"snapshot drift in {name}")


# ── Photometry snapshots ──────────────────────────────────────────


@pytest.mark.parametrize("mode", ["auto", "compositional", "hybrid", "exact"])
def test_photometry_snapshot(model, fixed_params, mode):
    flux = model.predict_photometry(fixed_params, mode=mode)
    _assert_snapshot(f"photometry_{mode}", jnp.asarray(flux))


# ── Spectrum snapshots ────────────────────────────────────────────


@pytest.fixture(scope="module")
def wave_obs():
    return jnp.linspace(3000.0, 10000.0, 200)


@pytest.mark.parametrize("mode", ["auto", "compositional", "exact"])
def test_spectrum_snapshot(model, fixed_params, wave_obs, mode):
    flux = model.predict_spectrum(fixed_params, wave_obs, mode=mode)
    _assert_snapshot(f"spectrum_{mode}", jnp.asarray(flux))


# ── Strategy routing observable ──────────────────────────────────


def test_list_available_kernels_populated(model):
    """The refactor's visibility win: every kernel has a status entry."""
    log = model.list_available_kernels()
    assert "exact_rest_sed" in log
    assert "compositional_rest_sed" in log
    assert "compositional_photometry" in log
    # Every recorded entry is "ok" or a documented failure prefix.
    for name, status in log.items():
        assert status == "ok" or status.startswith(("build_failed:", "build_returned_none")), (
            f"{name}: {status}"
        )
