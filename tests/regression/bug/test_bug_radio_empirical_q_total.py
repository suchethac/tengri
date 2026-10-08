# SPDX-License-Identifier: BSD-3-Clause
"""Delvecchio+2021 and McCheyne+2022 q_IR are TOTAL-radio calibrations (#2590 follow-up).

Both relations are fitted to observed total radio luminosity against total IR luminosity,
so the calibrated total at the relation's own reference frequency is

    L_tot(nu_ref) = L_IR / (3.75e12 Hz x 10^q(M*, z)),                 (Delvecchio Eq. 1)

and the emitted synchrotron is what remains after the Murphy+2011 free-free term at the same
frequency is taken out. With the free-free term on the radio block the public SED
(``sed_radio`` + ``sed_nebular``) therefore reproduces the calibration at nu_ref, for every
(M*, z, L_IR). The earlier construction multiplied the total by the Bell (2003) n(L)
suppression on top of a q(M*, z) that already carries the mass and luminosity trend,
which counted that correction twice and missed the calibration by -11 % to +12 %.

Every expectation in this file is written from the paper equation, never by calling the
code under test:

* Delvecchio+2021, arXiv:2010.05510, Eq. 5 (multi-parametric q_IR(M*, z) fit, 1.4 GHz,
  AGN-corrected total radio), coefficients 2.646 +/- 0.024, -0.023 +/- 0.008, 0.148 +/- 0.013.
* McCheyne+2022, A&A 662, A100, Sect. 5.2 joint fit (150 MHz): q_TIR = 1.98 (1+z)^0.02
  - 0.22 (log M* - 10.45), valid for z < 0.4 and M* > 10^10.45. The text was read from the
  paper's Leiden accepted manuscript as quoted in the tengri issue thread (#2805); the
  public path below grids M* outside that validity range as a formula check only.
* Murphy+2011 Eqs. 4 and 11 for the free-free term (3.88e-44 Msun/yr per erg/s of L_IR,
  2.174e27 erg/s/Hz per Msun/yr at 1 GHz, nu^-0.1 at T_e = 1e4 K).
"""

from __future__ import annotations

import numpy as np
import pytest

import tengri
from tengri.config.exceptions import ConfigError
from tengri.utils.physics_constants import L_SUN as _L_SUN

pytestmark = pytest.mark.regression_bug

_C_AA = 2.99792458e18  # Angstrom/s
_LOG_M_GRID = (9.5, 10.0, 10.5, 11.0)
_Z_GRID = (0.0, 1.0)
_LOG_TOTAL_MASS_OFFSET = 0.2246  # log_total_mass - log_mstar at age 5 Gyr, measured

# (sfr_mode, paper q(M*, z), nu_ref [Hz], synchrotron index used by the mode)
_MODES = {
    "delvecchio2021": (
        lambda logm, z: 2.646 * (1.0 + z) ** (-0.023) - 0.148 * (logm - 10.0),
        1.4e9,
    ),
    "mccheyne2022": (
        # McCheyne+2022 Sect. 5.2 joint fit, pivot log M* = 10.45 (paper text, Eq. 4 definition)
        lambda logm, z: 1.98 * (1.0 + z) ** 0.02 - 0.22 * (logm - 10.45),
        1.5e8,
    ),
}


# ---- the literature, written out here ----------------------------------------------


def _total_ref(q, l_ir):
    """L_tot at the calibration frequency [erg/s/Hz] (Delvecchio Eq. 1)."""
    return l_ir / (3.75e12 * 10.0**q)


def _murphy_ff(nu, l_ir):
    """Murphy+2011 free-free L_nu [erg/s/Hz] at T_e = 1e4 K, alpha_ff = -0.1."""
    sfr = 3.88e-44 * l_ir  # Murphy Eq. 4, Msun/yr
    return (1.0 / 4.6e-28) * (nu / 1.0e9) ** (-0.1) * sfr  # Murphy Eq. 11, inverted


# ---- public path --------------------------------------------------------------------

_NEBULAR = {
    "none": {"type": "none"},
    "cue": {"type": "cue", "all_params": tengri.Fixed(tengri.DEFAULT)},
}


def _build(ssp, mode, *, log_total_mass, age, z, neb):
    return tengri.SEDModel.build(
        ssp_data=ssp,
        sfh={
            "type": "delayed",
            "tau_gyr": tengri.Fixed(1.0),
            "age_gyr": tengri.Fixed(age),
            "log_total_mass": tengri.Fixed(log_total_mass),
            "all_params": tengri.Fixed(tengri.DEFAULT),
        },
        dust_attenuation={
            "law": "calzetti",
            "type": "two_component",
            "tau_bc": tengri.Fixed(0.0),
            "tau_diff": tengri.Fixed(1.0),
            "all_params": tengri.Fixed(tengri.DEFAULT),
        },
        dust_emission={"type": "dl14", "all_params": tengri.Fixed(tengri.DEFAULT)},
        radio={
            "sf": {"type": mode, "all_params": tengri.Fixed(tengri.DEFAULT)},
            "agn": {"type": "none"},
            "all_params": tengri.Fixed(tengri.DEFAULT),
        },
        neb=_NEBULAR[neb],
        redshift=tengri.Fixed(z),
    ).predict_state({})


def _l_nu_at(state, key, nu):
    """Log-interpolate a rest-frame L_nu [erg/s/Hz] component at frequency ``nu`` [Hz]."""
    wave_nu = _C_AA / np.asarray(state.wave, dtype=np.float64)
    vals = np.asarray(state.derived[key], dtype=np.float64)
    order = np.argsort(wave_nu)
    return float(
        np.exp(
            np.interp(
                np.log(nu),
                np.log(wave_nu[order]),
                np.log(np.maximum(vals[order], 1e-300)),
            )
        )
    )


def _radio_ir_input(state):
    """The IR luminosity [erg/s] the radio block receives, read from the published state."""
    # The windowed IR input is published as radio_L_ir_input [L_sun] when present.
    if "radio_L_ir_input" in state.derived:
        return float(np.asarray(state.derived["radio_L_ir_input"])) * _L_SUN
    return float(np.asarray(state.derived["L_ir"]))


def _public_state(ssp, mode, neb, log_total_mass, age, z):
    """Build the public model for one mode and return (state, logM*, L_IR,in)."""
    st = _build(ssp, mode, log_total_mass=log_total_mass, age=age, z=z, neb=neb)
    return st, float(np.asarray(st.derived["log_mstar"])), _radio_ir_input(st)


def _cases():
    for mode in _MODES:
        for logm in _LOG_M_GRID:
            for z in _Z_GRID:
                for age in (1.0, 5.0):
                    yield mode, logm, z, age


@pytest.mark.parametrize(("mode", "logm", "z", "age"), list(_cases()))
def test_radio_block_alone_reproduces_the_total_calibration(ssp_data_fsps, mode, logm, z, age):
    """No nebular: sed_radio at nu_ref is the paper's total, thermal term included."""
    q_of, nu_ref = _MODES[mode]
    st, got_logm, l_ir = _public_state(
        ssp_data_fsps, mode, "none", logm + _LOG_TOTAL_MASS_OFFSET, age, z
    )
    assert got_logm == pytest.approx(logm, abs=0.3)  # realized M*; the offset depends on age
    total = _total_ref(q_of(got_logm, z), l_ir)
    got = _l_nu_at(st, "sed_radio", nu_ref)
    assert got / total == pytest.approx(1.0, abs=1e-3), (
        f"{mode} logM*={got_logm:.3f} z={z} L_IR={l_ir:.3g}: total/calibration = {got / total:.5f}"
    )


@pytest.mark.parametrize(("mode", "logm", "z", "age"), list(_cases()))
def test_radio_block_with_cue_nebular_is_the_total_minus_the_murphy_share(
    ssp_data_fsps, mode, logm, z, age
):
    """Cue on: the radio block keeps (1 - f_th) of the calibration, f_th the Murphy share."""
    q_of, nu_ref = _MODES[mode]
    st, got_logm, l_ir = _public_state(
        ssp_data_fsps, mode, "cue", logm + _LOG_TOTAL_MASS_OFFSET, age, z
    )
    total = _total_ref(q_of(got_logm, z), l_ir)
    murphy = _murphy_ff(nu_ref, l_ir)
    got = _l_nu_at(st, "sed_radio", nu_ref)
    assert got == pytest.approx(total - murphy, rel=1e-3)


@pytest.mark.parametrize(("mode", "logm", "z", "age"), list(_cases()))
def test_radio_plus_cue_nebular_is_the_fixed_share_residual_band(
    ssp_data_fsps, mode, logm, z, age
):
    """Sum over the calibration = (1 - f_th) + L_ff,neb / L_cal at nu_ref.

    The radio block subtracts the Murphy share f_th; the nebular backend adds its own
    free-free, which differs from f_th. The residual is therefore L_ff,neb(nu_ref) over the
    calibration, published by the nebular continuum, not a free fit.
    """
    q_of, nu_ref = _MODES[mode]
    st, got_logm, l_ir = _public_state(
        ssp_data_fsps, mode, "cue", logm + _LOG_TOTAL_MASS_OFFSET, age, z
    )
    total = _total_ref(q_of(got_logm, z), l_ir)
    murphy = _murphy_ff(nu_ref, l_ir)
    neb_ff = _l_nu_at(st, "sed_nebular", nu_ref)
    summed = _l_nu_at(st, "sed_radio", nu_ref) + neb_ff
    assert summed == pytest.approx(total - murphy + neb_ff, rel=1e-3)
    # The residual is measured, not zero: Cue's share differs from Murphy's.
    assert 0.0 < neb_ff / total < 0.3


# ---- negative synchrotron: the refusal covers the new modes ---------------------------


def _build_radio_box(ssp, mode, sf=None):
    return tengri.SEDModel.build(
        ssp_data=ssp,
        sfh={"type": "const", "all_params": tengri.Fixed(tengri.DEFAULT)},
        dust_attenuation={"type": "two_component", "law": "calzetti"},
        dust_emission={"type": "draine_li2014"},
        radio={"sf": {"type": mode, **(sf or {})}, "agn": {"type": "none"}},
        redshift=tengri.Fixed(0.0),
    )


@pytest.mark.parametrize("mode", list(_MODES))
def test_declared_free_box_of_the_new_modes_builds(ssp_data_fsps, mode):
    _build_radio_box(ssp_data_fsps, mode, sf={"all_params": tengri.FREE})


def test_delvecchio_q0_above_the_limit_is_refused_at_build(ssp_data_fsps):
    with pytest.raises(ConfigError, match=r"q_\*"):
        _build_radio_box(
            ssp_data_fsps,
            "delvecchio2021",
            sf={"radio_delv_q0": tengri.Uniform(1.8, 3.6)},
        )


def test_mccheyne_default_box_is_below_its_limit(ssp_data_fsps):
    _build_radio_box(
        ssp_data_fsps,
        "mccheyne2022",
        sf={"radio_mcch_q0": tengri.Uniform(1.0, 3.0)},
    )
