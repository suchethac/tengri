# SPDX-License-Identifier: BSD-3-Clause
"""``bell2003_split`` beside a free-free-bearing nebular backend: one thermal term (#2764).

Bell (2003, ApJ 586, 794, Eq. 1) calibrates the TOTAL 1.4 GHz luminosity,

    L_1.4 = L_IR / (3.75e12 Hz x 10^q).

``bell2003_split`` (AGNfitter-rX) splits that total (1 - f_th) / f_th into a
nu^-0.75 and a nu^-0.1 power law, f_th = 0.1. A nebular continuum that carries
free-free (``cue``, ``cloudy_grid``) is a second owner of the same thermal
emission, so the split keeps only its synchrotron and the nebular term supplies
the thermal one. Without such a backend the split is unchanged.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.radio import radio as R
from tengri.observation import Photometry

pytestmark = pytest.mark.regression_bug

_C_AA = 2.99792458e18
_Q = 2.64
_F_TH = 0.10
_ALPHA_NT = 0.75


def test_function_without_thermal_is_the_synchrotron_share():
    """include_thermal=False keeps (1 - f_th) of the Bell total at 1.4 GHz and nothing else."""
    l_ir = 1e44
    cal = l_ir / (3.75e12 * 10.0**_Q)
    nu = np.array([0.15e9, 1.4e9, 30e9])
    wave = _C_AA / nu
    full = np.asarray(R.radio_sfr_bell2003_split(wave, l_ir, _Q))
    sync = np.asarray(R.radio_sfr_bell2003_split(wave, l_ir, _Q, include_thermal=False))
    np.testing.assert_allclose(full[1], cal, rtol=1e-12)
    np.testing.assert_allclose(sync, (1 - _F_TH) * cal * (nu / 1.4e9) ** -_ALPHA_NT, rtol=1e-12)
    thermal = full - sync
    np.testing.assert_allclose(thermal, _F_TH * cal * (nu / 1.4e9) ** -0.1, rtol=1e-9)


def _build(ssp, neb):
    obs = Photometry.from_names(["sdss_g", "sdss_r"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            redshift=Fixed(0.0),
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
            neb=neb,
            radio={
                "sf": {"type": "bell2003_split"},
                "agn": {"type": "none"},
                "radio_q_ir": Fixed(_Q),
                "all_params": Fixed(DEFAULT),
            },
        )


def _node(wave, nu):
    """Index of the grid node nearest ``nu`` (compare at nodes: no interpolation error)."""
    return int(np.argmin(np.abs(np.log(wave) - np.log(_C_AA / nu))))


@pytest.mark.parametrize("backend", ["none", "cue"])
def test_one_thermal_term_in_the_built_model(ssp_data_bc03, backend):
    if backend == "cue":
        from tengri._data_setup import find_data_str

        if find_data_str("cue_weights.npz") is None:
            pytest.skip("Cue weights file not found")
        neb = {"type": "cue", "all_params": Fixed(DEFAULT)}
    else:
        neb = {"type": "none"}
    model = _build(ssp_data_bc03, neb)
    state = model.predict_state(model.spec.sample(jax.random.PRNGKey(0)))
    wave = np.asarray(state.wave)
    radio = np.asarray(state.derived["sed_radio"])
    cal = float(state.derived["L_ir"]) / (3.75e12 * 10.0**_Q)

    idx = [_node(wave, f) for f in (0.5e9, 1.4e9, 5e9)]
    nu = _C_AA / wave[idx]
    got = radio[idx]
    if backend == "none":
        # Split alone: synchrotron plus its own thermal law, total = Bell's L_1.4.
        expect = cal * ((1 - _F_TH) * (nu / 1.4e9) ** -_ALPHA_NT + _F_TH * (nu / 1.4e9) ** -0.1)
        np.testing.assert_allclose(got, expect, rtol=1e-9)
        ref = _node(wave, 1.4e9)
        # the node nearest 1.4 GHz: total there is the calibration times the power-law offset
        nu_ref = _C_AA / wave[ref]
        at_ref = (1 - _F_TH) * (nu_ref / 1.4e9) ** -_ALPHA_NT + _F_TH * (nu_ref / 1.4e9) ** -0.1
        assert radio[ref] == pytest.approx(cal * at_ref, rel=1e-9)
    else:
        # Radio block holds the synchrotron only; the nebular continuum is the thermal term.
        np.testing.assert_allclose(got, (1 - _F_TH) * cal * (nu / 1.4e9) ** -_ALPHA_NT, rtol=1e-9)
        ref = _node(wave, 1.4e9)
        neb_ff = float(np.asarray(state.derived["sed_nebular"])[ref])
        assert neb_ff > 0.0
        # One thermal term: Bell's total with the nebular free-free in place of the
        # split's 10 % share. The doubled model sat at (1.0 + neb/cal) = 1.06 and above;
        # one thermal term gives 0.9 + neb/cal, 0.981 for this case (the modelled nebular
        # free-free share is not Bell's empirical 10 %).
        assert 0.95 < (radio[ref] + neb_ff) / cal < 1.00
