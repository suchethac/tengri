# SPDX-License-Identifier: BSD-3-Clause
"""Contract: ``bell2003`` and ``bell2003_split`` share the thermal term of Bell's total (#2590).

Bell (2003) Eq. 1 defines ``q_IR`` on the TOTAL 1.4 GHz luminosity. Through the pipeline,
``bell2003_split`` is the same model as ``bell2003``: the synchrotron term is the calibrated
total minus the Murphy et al. (2011) thermal luminosity at 1.4 GHz, and the thermal term is
added beside it, so the sum at 1.4 GHz is the calibration and the thermal emission is counted
once. The stand-alone function :func:`tengri.radio.radio_sfr_bell2003_split` keeps the
AGNFITTER-RX 90/10 construction with its own slopes.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.radio.component import RadioSEDComponentConfig
from tengri.observation import Photometry
from tengri.parameters.registry import registry

pytestmark = pytest.mark.contract

#: Matches ``tengri.components.radio.radio._RADIO_WAVE_MIN_AA`` (1 mm).
_RADIO_WAVE_MIN_AA = 1.0e7

#: Declared default for radio_q_ir from the registry (Bell 2003: 2.64).
_Q_IR = registry().get("radio_q_ir").prior.value

#: Declared default for radio_alpha_sf from the registry (typical 0.8).
_ALPHA_SF = registry().get("radio_alpha_sf").prior.value


def _build(ssp_data, sf_type: str):
    """A dusty, energy-balanced galaxy with a real (nonzero) L_ir, SF radio only."""
    obs = Photometry.from_names(["sdss_g", "sdss_r"])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="BakedInBackend")
        return SEDModel.build(
            ssp_data=ssp_data,
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
                "sf": {"type": sf_type},
                "agn": {"type": "none"},
                "radio_q_ir": Fixed(_Q_IR),
                "radio_alpha_sf": Fixed(_ALPHA_SF),
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.0),
        )


def _expected_sed(wave, l_ir):
    """Bell total shared with Murphy+2011 free-free, from the literature formulas."""
    nu = 2.99792458e18 / wave
    sfr = 3.88e-44 * l_ir  # Murphy+2011 Eq. 4 [Msun/yr]
    ff = (1.0 / 4.6e-28) * (nu / 1.0e9) ** -0.1 * sfr  # Eq. 11, T_e = 1e4 K
    ff_ref = (1.0 / 4.6e-28) * 1.4**-0.1 * sfr
    total_ref = l_ir / (3.75e12 * 10.0**_Q_IR)  # Bell Eq. 1
    return (total_ref - ff_ref) * (nu / 1.4e9) ** (-_ALPHA_SF) + ff


@pytest.mark.parametrize("sf_type", ["bell2003", "bell2003_split"])
def test_component_output_is_the_shared_total(ssp_data_fsps, sf_type):
    """End-to-end ``sed_radio`` equals (total - thermal(1.4 GHz)) power law + thermal."""
    model = _build(ssp_data_fsps, sf_type)
    state = model.predict_state(model.spec.sample(jax.random.PRNGKey(0)))
    wave = np.asarray(state.wave)
    sed_radio = np.asarray(state.derived["sed_radio"])
    mask = wave > _RADIO_WAVE_MIN_AA
    assert np.any(mask), "no radio-band wavelength points in the model's grid"
    want = _expected_sed(wave[mask], float(state.derived["L_ir"]))
    np.testing.assert_allclose(sed_radio[mask], want, rtol=1e-8)


def test_split_spelling_is_bit_identical_to_bell2003(ssp_data_fsps):
    outs = []
    for sf_type in ("bell2003", "bell2003_split"):
        model = _build(ssp_data_fsps, sf_type)
        state = model.predict_state(model.spec.sample(jax.random.PRNGKey(0)))
        outs.append(np.asarray(state.derived["sed_radio"]))
    np.testing.assert_array_equal(outs[0], outs[1])


def test_split_config_resolves_like_bell2003():
    """Unset resolves to True for both spellings; an explicit True or False is accepted."""
    for mode in ("bell2003", "bell2003_split"):
        assert RadioSEDComponentConfig(sfr_mode=mode).include_freefree is True
        assert RadioSEDComponentConfig(sfr_mode=mode, include_freefree=True).include_freefree
        assert (
            RadioSEDComponentConfig(sfr_mode=mode, include_freefree=False).include_freefree
            is False
        )
