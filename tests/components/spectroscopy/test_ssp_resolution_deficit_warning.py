# SPDX-License-Identifier: BSD-3-Clause
"""Contract: warn, don't silently clamp, where sigma_inst^2 < sigma_lib^2 (#2518).

``apply_lsf`` still floors the deficit to zero in-jit (the only numerically
safe behavior inside a traced kernel); the promise checked here is the
build-time (pre-trace) warning naming which pixels the model cannot resolve
through the SSP library's own template resolution, so the regime is
surfaced rather than hidden.
"""

import warnings

import jax.numpy as jnp
import pytest

from tengri import Fixed, Observation, Parameters, Spectroscopy
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.forward.sed_model import SEDModel

pytestmark = pytest.mark.contract

_C_KMS = 299792.458
_FWHM_TO_SIGMA = 2.3548200450309493


def _minimal_spec():
    return Parameters(
        sfh_tsnorm_log_total_mass=Fixed(1.0),
        sfh_tsnorm_peak_lbt_gyr=Fixed(0.5),
        sfh_tsnorm_width_gyr=Fixed(0.3),
        sfh_tsnorm_skew=Fixed(0.0),
        sfh_tsnorm_trunc=Fixed(3.0),
        met_logzsol=Fixed(-0.3),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.0),
    )


def _resolution_for(sigma_inst_kms: float) -> float:
    return _C_KMS / (_FWHM_TO_SIGMA * sigma_inst_kms)


def test_warns_when_instrument_narrower_than_miles_blue_end():
    """sigma_inst=50 km/s over the MILES blue end (~89-91 km/s) must warn.

    Every pixel here has sigma_inst^2 < sigma_lib(lambda)^2, so the model
    cannot resolve through the library's own template resolution at all in
    this window; the deficit is silently invisible without this warning
    (#2518's own motivating example).
    """
    ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")
    wave_obs = jnp.linspace(3600.0, 4200.0, 50)
    obs = Observation(
        spectroscopy=Spectroscopy(wave_obs=wave_obs, resolution=_resolution_for(50.0))
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SEDModel(_minimal_spec(), ssp, observation=obs)

    matches = [
        w
        for w in caught
        if issubclass(w.category, UserWarning)
        and "instrument resolution narrower than" in str(w.message)
    ]
    assert len(matches) == 1
    assert "fsps_prsc_miles_chabrier" in str(matches[0].message)
    assert "50/50" in str(matches[0].message) or "100.0%" in str(matches[0].message)


def test_no_deficit_warning_when_instrument_resolves_the_library():
    """sigma_inst=300 km/s (coarse) safely exceeds MILES everywhere in this window."""
    ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")
    wave_obs = jnp.linspace(3600.0, 7300.0, 50)
    obs = Observation(
        spectroscopy=Spectroscopy(wave_obs=wave_obs, resolution=_resolution_for(300.0))
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SEDModel(_minimal_spec(), ssp, observation=obs)

    matches = [
        w
        for w in caught
        if issubclass(w.category, UserWarning)
        and "instrument resolution narrower than" in str(w.message)
    ]
    assert matches == []


def test_warns_when_falling_back_to_flat_scalar():
    """A library with no documented curve warns once, naming the flat fallback."""
    ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")
    ssp_no_curve = ssp._replace(ssp_resolution_kms=None)
    wave_obs = jnp.linspace(3600.0, 7300.0, 50)
    obs = Observation(
        spectroscopy=Spectroscopy(wave_obs=wave_obs, resolution=2000.0, sigma_lib_kms=70.0)
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SEDModel(_minimal_spec(), ssp_no_curve, observation=obs)

    matches = [
        w
        for w in caught
        if issubclass(w.category, UserWarning) and "no per-wavelength LSF curve" in str(w.message)
    ]
    assert len(matches) == 1
    assert "sigma_lib_kms=70.0" in str(matches[0].message)
