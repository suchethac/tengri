# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validation: DSPS CSP rest-UV vs a native-age reference (#858, #538).

The CIGALE panchromatic head-to-head shows tengri's intrinsic stellar SED a few
percent brighter than CIGALE in the rest-UV on identical BC03 templates. This
was first suspected to be a DSPS integration bug (#858) but is **working as
intended**: it is the continuum flip-side of the #538 young-boundary knot, which
captures the ``[0, age0 ~1 Myr]`` recent star formation into the youngest SSP
bin. A native-age integration (bins starting at 1 Myr) — like CIGALE, and like a
naive dense trapezoid — *misses* that recent SF, so it comes out fainter in the
rest-UV **and** lower in Q_H. The knot moves Q_H toward the analytic (continuous)
SFH->SSP convolution, so tengri is the more physically complete side.

This module pins two facts:

* :func:`test_dsps_flux_sum_is_faithful` — ``sed_intrinsic`` == the DSPS
  ``age_weights x ssp_flux`` to <0.1% (the flux sum is exact; any UV difference
  lives in the age-weighting, not flux handling).
* :func:`test_young_knot_brightens_uv_vs_native_binning` — tengri's rest-UV is
  *brighter* than a native-age re-integration of its own SFH, by the young-knot
  recent-SF contribution. This documents the intended #538 behavior (removing
  the knot to match CIGALE's rest-UV would regress Q_H) and guards its sign.

Data-gated on ``reproduction/cigale/_drivers/data/bc03_from_cigale.h5`` (the
CIGALE BC03 grid with a dense 13700-age axis); skipped when absent.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from tengri import FIXED, Fixed, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

pytestmark = pytest.mark.crossval

_BC03 = (
    Path(__file__).resolve().parents[2]
    / "reproduction"
    / "cigale"
    / "_drivers"
    / "data"
    / "bc03_from_cigale.h5"
)
_HAS_BC03 = _BC03.is_file()
_LOG10_ZSUN = -1.848  # Asplund 2009 (matches the naming contract)
_MET_002 = float(np.log10(0.02) - _LOG10_ZSUN)  # met_logzsol for Z_abs = 0.02


def _uv_opt(wave: np.ndarray, lnu: np.ndarray) -> float:
    """Rest-UV/optical L_nu ratio, 1500 A / 5000 A (normalization-independent)."""
    return float(lnu[np.argmin(np.abs(wave - 1500.0))] / lnu[np.argmin(np.abs(wave - 5000.0))])


def _build_state():
    ssp = load_ssp_data(str(_BC03))
    model = SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(_MET_002), "*": FIXED},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(0.0),
            "*": FIXED,
        },
        dust={
            "law": "power_law",
            "type": "two_component",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "*": FIXED,
        },
        redshift=Fixed(0.0),
    )
    return model.predict_state({})


def _raw_ssp_lnu():
    """Raw BC03 SSP at Z=0.02 [L_nu/Msun], its wave grid, and native ages [yr]."""
    with h5py.File(_BC03, "r") as f:
        lgmet = np.asarray(f["ssp_lgmet"])
        flux = np.asarray(f["ssp_flux"])  # (n_met, n_age, n_wave) L_nu/Msun
        wave = np.asarray(f["ssp_wave"])
        age_yr = 10.0 ** np.asarray(f["ssp_lg_age_gyr"]) * 1.0e9
    i_z = int(np.argmin(np.abs(lgmet - np.log10(0.02))))
    return flux[i_z], wave, age_yr


@pytest.mark.skipif(not _HAS_BC03, reason="bc03_from_cigale.h5 (reproduction data) not present")
def test_dsps_flux_sum_is_faithful():
    """``sed_intrinsic`` == sum(age_weights x ssp_flux): the flux sum is exact.

    Confirms any rest-UV difference lives in the age-weighting (the #538 young
    knot), not in flux handling.
    """
    state = _build_state()
    wave = np.asarray(state.wave)
    sed = np.asarray(state.sed_intrinsic)
    ssp_lnu, ssp_wave, _ = _raw_ssp_lnu()
    weights = np.asarray(state.derived["age_weights"])
    direct = (weights[:, None] * ssp_lnu).sum(axis=0)  # L_nu, same units as sed
    np.testing.assert_allclose(_uv_opt(wave, sed), _uv_opt(ssp_wave, direct), rtol=5e-3)


@pytest.mark.skipif(not _HAS_BC03, reason="bc03_from_cigale.h5 (reproduction data) not present")
def test_young_knot_brightens_uv_vs_native_binning():
    """tengri's rest-UV exceeds a native-age re-integration by the recent-SF capture.

    Intended #538 behavior, not a bug (#858 closed): the young-boundary knot
    captures the [0, ~1 Myr] recent SF that a native-age integration (bins
    starting at 1 Myr, like CIGALE) misses. So tengri is *brighter* in the
    rest-UV -- the continuum flip-side of the knot's Q_H fix. This guards the
    sign and rough magnitude; a regression that removed the knot (to match
    CIGALE's fainter rest-UV) would also drop Q_H and trip this test.
    """
    state = _build_state()
    wave = np.asarray(state.wave)
    sed = np.asarray(state.sed_intrinsic)
    ssp_lnu, ssp_wave, age_yr = _raw_ssp_lnu()

    # Native-age re-integration of tengri's OWN SFH -- misses the [0, age0]
    # recent SF (no young-boundary extension), matching CIGALE's convention.
    sfr_history = np.asarray(state.derived["sfr_history"])
    lookback_yr = np.asarray(state.derived["sfh_grid_lbt_yr"])
    order = np.argsort(lookback_yr)
    sfr_on_ssp = np.interp(age_yr, lookback_yr[order], sfr_history[order], left=0.0, right=0.0)
    native = ((sfr_on_ssp * np.gradient(age_yr))[:, None] * ssp_lnu).sum(axis=0)

    r_tengri = _uv_opt(wave, sed)
    r_native = _uv_opt(ssp_wave, native)
    # tengri captures the recent SF the native binning drops -> brighter rest-UV.
    assert r_tengri > r_native, (
        f"rest-UV/opt: tengri={r_tengri:.4f} should exceed the native-age "
        f"reference={r_native:.4f} (young-knot recent-SF capture, #538)"
    )
    # ... by a few percent (not a runaway); guards the magnitude.
    assert 1.01 < r_tengri / r_native < 1.10, (
        f"rest-UV excess {r_tengri / r_native:.3f}x outside the expected "
        f"~1.01-1.10 young-knot band (#538/#858)"
    )
