"""Cross-validation: DSPS CSP integration accuracy in the rest-UV (#858).

Building the CIGALE panchromatic head-to-head surfaced a systematic
rest-UV offset: tengri's intrinsic stellar SED is a few percent too blue vs
CIGALE on **identical** BC03 templates. Traced to the DSPS composite-population
age-weighting (the SFH -> SSP-age mass mapping), not the flux interpolation and
not an SFH-parametrisation convention:

* tengri ``sed_intrinsic`` reproduces its own ``age_weights x ssp_flux`` to
  <0.1% (the flux sum is faithful) -- guarded by
  :func:`test_dsps_flux_sum_is_faithful`.
* but a **dense** trapezoidal re-integration of tengri's own published
  ``sfr_history`` over the SSP's native ages gives a rest-UV ~4% fainter, and
  that dense value matches CIGALE's ``bc03`` module exactly (UV/opt 1500/5000 A:
  tengri 0.1953, dense reference 0.1872, CIGALE 0.1872). tengri over-weights the
  young UV-bright stars.

This module pins both facts. The dense-reference agreement is expected to
**fail until #858 is fixed** (marked ``xfail``); when the age-weighting is
corrected the UV ratio converges to the dense reference and the marker flips.

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
    """Rest-UV/optical L_nu ratio, 1500 A / 5000 A (normalisation-independent)."""
    return float(lnu[np.argmin(np.abs(wave - 1500.0))] / lnu[np.argmin(np.abs(wave - 5000.0))])


def _build_state():
    ssp = load_ssp_data(str(_BC03))
    model = SEDModel.build(
        ssp_data=ssp,
        stellar={"logzsol": Fixed(_MET_002), "*": FIXED},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(0.0),
            "*": FIXED,
        },
        dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
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

    Isolates the #858 offset to the age-weighting rather than flux handling.
    """
    state = _build_state()
    wave = np.asarray(state.wave)
    sed = np.asarray(state.sed_intrinsic)
    ssp_lnu, ssp_wave, _ = _raw_ssp_lnu()
    weights = np.asarray(state.derived["age_weights"])
    direct = (weights[:, None] * ssp_lnu).sum(axis=0)  # L_nu, same units as sed
    np.testing.assert_allclose(_uv_opt(wave, sed), _uv_opt(ssp_wave, direct), rtol=5e-3)


@pytest.mark.skipif(not _HAS_BC03, reason="bc03_from_cigale.h5 (reproduction data) not present")
@pytest.mark.xfail(
    strict=True,
    reason="#858: DSPS age-weighting over-weights young stars -> rest-UV ~4% too bright "
    "vs a dense re-integration of the same SFH (which matches CIGALE).",
)
def test_dsps_csp_matches_dense_reintegration_uv():
    """tengri's rest-UV must match a dense re-integration of its OWN SFH.

    Convention-free: uses tengri's published ``sfr_history`` re-integrated over
    the SSP's native ages. Currently the ratio is ~0.96 (tengri too blue); the
    dense reference equals CIGALE. Flips to pass when #858 is fixed.
    """
    state = _build_state()
    wave = np.asarray(state.wave)
    sed = np.asarray(state.sed_intrinsic)
    ssp_lnu, ssp_wave, age_yr = _raw_ssp_lnu()

    sfr_history = np.asarray(state.derived["sfr_history"])
    lookback_yr = np.asarray(state.derived["sfh_grid_lbt_yr"])
    order = np.argsort(lookback_yr)
    sfr_on_ssp = np.interp(age_yr, lookback_yr[order], sfr_history[order], left=0.0, right=0.0)
    reference = ((sfr_on_ssp * np.gradient(age_yr))[:, None] * ssp_lnu).sum(axis=0)

    r_tengri = _uv_opt(wave, sed)
    r_reference = _uv_opt(ssp_wave, reference)
    np.testing.assert_allclose(
        r_tengri,
        r_reference,
        rtol=1e-2,
        err_msg=(
            f"rest-UV/opt: tengri={r_tengri:.4f} vs dense re-integration="
            f"{r_reference:.4f} ({r_tengri / r_reference:.3f}x); DSPS over-weights "
            f"the young UV (#858)"
        ),
    )
