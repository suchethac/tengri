# SPDX-License-Identifier: BSD-3-Clause
"""Regression for #3409: radio and X-ray custom bands predict through a full model.

A model with a radio star-formation component and an X-ray component is observed
through a 1.4 GHz top-hat, the shipped VLA L-band curve, and a 2-10 keV band given
in keV. The photometry must be finite and positive. The 1.4 GHz flux density must
equal the model's rest-frame L_nu at the rest frequency 1.4 GHz (1 + z), with the
(1 + z) K-correction and 4 pi d_L^2 dimming, to 1 %.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.observation import Photometry
from tengri.observation.filters.custom import register_filter, unregister_filter
from tengri.utils.cosmology import luminosity_distance

Z = 0.5
C_AA = 2.99792458e18

_RADIO_TOPHAT = "bug3409_vla_l_tophat"
_RADIO_NARROW = "bug3409_radio_narrow"
_XRAY_KEV = "bug3409_xray_2_10kev"
_SHIPPED_VLA = "VLA_L_1p4GHz"
_NAMES = [_RADIO_TOPHAT, _RADIO_NARROW, _XRAY_KEV, _SHIPPED_VLA]


@pytest.fixture
def custom_bands():
    """Register the custom bands for one test and remove them afterwards."""
    nu_full = np.linspace(1.0, 2.0, 201)
    register_filter(_RADIO_TOPHAT, nu_full, np.ones_like(nu_full), wave_unit="GHz", overwrite=True)
    nu_narrow = np.linspace(1.393, 1.407, 201)
    register_filter(
        _RADIO_NARROW, nu_narrow, np.ones_like(nu_narrow), wave_unit="GHz", overwrite=True
    )
    e_kev = np.linspace(2.0, 10.0, 201)
    register_filter(_XRAY_KEV, e_kev, np.ones_like(e_kev), wave_unit="keV", overwrite=True)
    yield
    for name in (_RADIO_TOPHAT, _RADIO_NARROW, _XRAY_KEV):
        unregister_filter(name)


def _build(ssp):
    obs = Photometry.from_names(_NAMES)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="BakedInBackend")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={
                "type": "delayed",
                "sfh_delayed_log_total_mass": Fixed(10.0),
                "sfh_delayed_tau_gyr": Fixed(1.0),
                "sfh_delayed_age_gyr": Fixed(5.0),
            },
            dust_attenuation={
                "type": "single_component",
                "law": "calzetti",
                "tau_v": Fixed(0.5),
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dh02_ce01", "all_params": Fixed(DEFAULT)},
            radio={
                "sf": {"type": "bell2003_split"},
                "agn": {"type": "none"},
                "all_params": Fixed(DEFAULT),
            },
            xray={"type": "yang20", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(Z),
        )


def _radio_lnu_at(state, nu_rest_hz: float) -> float:
    """Model L_nu [erg/s/Hz] at a rest frequency, log-log interpolated on the radio grid."""
    wave = np.asarray(state.wave, dtype=float)
    lnu = np.asarray(state.derived["sed_radio"], dtype=float)
    nu = C_AA / wave
    order = np.argsort(nu)
    nu, lnu = nu[order], lnu[order]
    positive = lnu > 0.0
    log_val = np.interp(np.log10(nu_rest_hz), np.log10(nu[positive]), np.log10(lnu[positive]))
    return float(10.0**log_val)


@pytest.fixture
def predicted(synthetic_ssp_wide, custom_bands):
    model = _build(synthetic_ssp_wide)
    params = model.spec.sample(jax.random.PRNGKey(0))
    phot = np.asarray(model.predict_photometry(params), dtype=float)
    state = model.predict_state(params)
    return phot, state


def test_radio_and_xray_photometry_is_finite_and_positive(predicted):
    phot, _ = predicted
    assert phot.shape == (len(_NAMES),)
    assert np.all(np.isfinite(phot))
    assert np.all(phot > 0.0)


def test_shipped_vla_l_band_matches_custom_tophat_to_one_percent(predicted):
    phot, _ = predicted
    shipped = phot[_NAMES.index(_SHIPPED_VLA)]
    custom = phot[_NAMES.index(_RADIO_TOPHAT)]
    assert shipped == pytest.approx(custom, rel=1e-2)


def test_radio_narrow_band_flux_matches_rest_lnu_with_k_correction(predicted):
    phot, state = predicted
    observed = phot[_NAMES.index(_RADIO_NARROW)]
    nu_rest = 1.4e9 * (1.0 + Z)
    d_l_cm = float(luminosity_distance(Z))
    expected = (1.0 + Z) * _radio_lnu_at(state, nu_rest) / (4.0 * np.pi * d_l_cm**2)
    assert observed == pytest.approx(expected, rel=1e-2)
