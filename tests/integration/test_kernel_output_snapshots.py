# SPDX-License-Identifier: BSD-3-Clause
"""Snapshot regression: predict_photometry/spectrum numerical output.

Pins the numerical output of the forward predict paths under a fixed
param dict so refactors cannot silently drift the numbers. We snapshot
two photometry paths — the default exact path and the build-time
``approx=WavePrecomp(...)`` LUT path — each compared to *itself* across
runs (the LUT path is an approximation, so it has its own baseline), plus
the exact spectrum path.

The first run writes ``*.npy`` under ``tests/_data/kernel_snapshots/``.
Subsequent runs assert bit-identical output — any drift fails.
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
from tengri.forward.sed_model import SEDModel, WavePrecomp
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
def waveprecomp_model(ssp_data, filters):
    """Same model under the build-time WavePrecomp LUT (approx=)."""
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
    return SEDModel(
        spec=spec,
        ssp_data=ssp_data,
        filters=filters,
        approx=WavePrecomp(z_min=0.0, z_max=1.0, n_z=100),
    )


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


def test_photometry_snapshot_exact(model, fixed_params):
    """Default (exact) path — pins predict_photometry output bit-for-bit."""
    flux = model.predict_photometry(fixed_params)
    _assert_snapshot("photometry_exact", jnp.asarray(flux))


def test_photometry_snapshot_waveprecomp(waveprecomp_model, fixed_params):
    """WavePrecomp LUT path (approx=) — pinned to itself (it is an approximation)."""
    flux = waveprecomp_model.predict_photometry(fixed_params)
    _assert_snapshot("photometry_waveprecomp", jnp.asarray(flux))
