# SPDX-License-Identifier: BSD-3-Clause
"""Energy closure of the Lyman continuum when Cue counts every stellar age.

Cue re-emits the ionizing photons of every age it counts as nebular lines and continuum.
The stellar SED must give up the same photons: under ``lyc_reprocessed_by='all'`` (the
default with Cue) it does. Under ``'young'`` only ages below ``t_birth_yr`` are removed,
so Cue counting older ages creates their energy a second time (94-96% of it for a
quiescent galaxy); that pairing is refused at build time.

Measured on the old-LyC FSPS grid (``tests/_old_lyc_ssp.py``, 5e40 photons/s/Msun above
1 Gyr; fesc = fdust = 0, no dust): LyC energy removed from the stellar SED vs nebular
energy (lines + continuum) emitted.
"""

from __future__ import annotations

import warnings
from functools import cache

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, Observation, Photometry, SEDModel
from tengri.config.exceptions import ConfigError
from tengri.utils.physics_constants import C_AA
from tests._old_lyc_ssp import FSPS_PRSC, ssp_with_old_lyc

pytestmark = pytest.mark.skipif(
    not FSPS_PRSC.is_file(), reason="needs data/fsps_prsc_miles_chabrier.h5"
)

_LYMAN_AA = 911.76
#: (tau_gyr, age_gyr): quenched, and a mixed population with old stars carrying ~half of Q_H.
_CASES = [(0.3, 10.0), (1.0, 10.0)]
#: Measured nebular/removed energy ratio under all ages + 'all': 0.944 and 0.973. The rest
#: of the removed LyC energy is not re-emitted as lines or continuum within the SED grid.
_MIN_REEMITTED = 0.90


@cache
def _ssp():
    return ssp_with_old_lyc()


def _build(neb, reprocessed_by=None, fdust_frac=0.0):
    dust = {
        "law": "power_law",
        "type": "two_component",
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    }
    if reprocessed_by is not None:
        dust["lyc_reprocessed_by"] = reprocessed_by
    if neb is None:
        neb_group = {"type": "none"}
    else:
        neb_group = {
            "type": "cue",
            "neb_fesc": Fixed(0.0),
            "neb_fdust_frac": Fixed(fdust_frac),
            "neb_logZ_gas": Fixed(0.0),
            "neb_logU": Fixed(-2.0),
            "all_params": Fixed(DEFAULT),
            **neb,
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=_ssp(),
            observation=Observation(photometry=Photometry.from_names(["des_g", "des_r"])),
            sfh={
                "type": "delayed",
                "tau_gyr": FREE,
                "age_gyr": FREE,
                "log_total_mass": FREE,
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation=dust,
            met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
            neb=neb_group,
            redshift=Fixed(0.0),
        )


def _params(tau, age):
    return {
        "sfh_delayed_tau_gyr": jnp.asarray(tau),
        "sfh_delayed_age_gyr": jnp.asarray(age),
        "sfh_delayed_log_total_mass": jnp.asarray(10.0),
    }


def _energy(wave, lnu, mask=None):
    """Integral of L_nu over frequency [erg/s] by the trapezoid rule in wavelength."""
    integrand = np.asarray(lnu, dtype=np.float64) * C_AA / wave**2
    if mask is not None:
        integrand = np.where(mask, integrand, 0.0)
    return float(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(wave)))


@cache
def _ledger(tau, age, neb_key, reprocessed_by):
    """(raw LyC energy, LyC left in the SED, nebular energy) [erg/s] for one configuration."""
    neb = dict(neb_key) if neb_key is not None else None
    raw = _build(None).predict_state(_params(tau, age))
    wave = np.asarray(raw.wave, dtype=np.float64)
    e_raw = _energy(wave, raw.sed_intrinsic, wave < _LYMAN_AA)
    state = _build(neb, reprocessed_by).predict_state(_params(tau, age))
    wave_out = np.asarray(state.wave, dtype=np.float64)
    e_left = _energy(wave_out, state.sed_intrinsic, wave_out < _LYMAN_AA)
    e_neb = _energy(wave_out, state.derived["sed_nebular"])
    return e_raw, e_left, e_neb


@pytest.mark.conservation
@pytest.mark.parametrize("reprocessed_by", [None, "all"], ids=["default", "explicit_all"])
@pytest.mark.parametrize("case", _CASES, ids=lambda c: f"tau{c[0]}_age{c[1]}")
def test_all_ages_cue_closes_the_lyman_continuum_budget(case, reprocessed_by):
    """Every LyC photon removed from the stellar SED is re-emitted, no more.

    The stellar SED keeps no flux below 912 A, and the nebular energy is the removed energy
    times the re-emitted fraction (measured 0.944 and 0.973), never above it.
    """
    e_raw, e_left, e_neb = _ledger(*case, (), reprocessed_by)
    removed = e_raw - e_left
    assert e_left == 0.0
    assert removed == pytest.approx(e_raw, rel=1e-12)
    assert e_neb <= removed * 1.0 + 1e-3 * removed, "nebular energy created from nothing"
    assert e_neb >= _MIN_REEMITTED * removed


@pytest.mark.conservation
@pytest.mark.parametrize("case", _CASES, ids=lambda c: f"tau{c[0]}_age{c[1]}")
def test_young_split_with_a_matching_cutoff_closes_too(case):
    """``'young'`` is consistent when Cue counts only the ages the split removes (<= t_birth)."""
    e_raw, e_left, e_neb = _ledger(*case, (("lyc_max_age_yr", 1.0e7),), "young")
    removed = e_raw - e_left
    assert removed > 0.0
    assert e_neb <= removed * 1.001
    assert e_neb >= _MIN_REEMITTED * removed


@pytest.mark.regression_bug
def test_young_with_all_ages_cue_is_refused_and_names_both_fixes():
    """The double count itself: 'young' + every age raises, with the two fixes in the message."""
    with pytest.raises(ConfigError) as err:
        _build({}, "young")
    text = str(err.value)
    assert "double-counts" in text
    assert "'lyc_reprocessed_by': 'all'" in text
    assert "lyc_max_age_yr <= t_birth_yr" in text


@pytest.mark.regression_bug
def test_young_with_a_cutoff_above_the_former_cut_is_refused():
    with pytest.raises(ConfigError, match="double-counts"):
        _build({"lyc_max_age_yr": 5.0e8}, "young")


def _spec(neb_extra, reprocessed_by):
    from tengri.parameters.groups import parse_groups

    return parse_groups(
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT), **neb_extra},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "lyc_reprocessed_by": reprocessed_by,
        },
        redshift=Fixed(0.1),
    )


@pytest.mark.regression_bug
def test_former_cut_with_young_warns_once_instead_of_raising():
    """10-100 Myr gap of the former cut: documented, one warning, no error."""
    from tengri.parameters import parameters as parameters_module

    parameters_module._CUE_YOUNG_GAP_WARNED = False
    with pytest.warns(UserWarning, match="10-100 Myr energy gap"):
        spec = _spec({"lyc_max_age_yr": 1.0e8}, "young")
    assert spec.dust_lyc_reprocessed_by == "young"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _spec({"lyc_max_age_yr": 1.0e8}, "young")


@pytest.mark.regression_bug
def test_default_is_all_for_cue_and_young_otherwise():
    assert _build({}).spec.dust_lyc_reprocessed_by == "all"
    assert "[default: 'all' for cue]" in _build({}).spec.summary_str()
    assert "[user]" in _build({}, "all").spec.summary_str()


@pytest.mark.conservation
def test_dust_credit_is_the_dust_share_of_the_whole_lyman_continuum():
    """Under 'all' the dust credit is fdust x the LyC of every age (fesc = 0)."""
    tau, age, fdust = 1.0, 10.0, 0.3
    e_raw, _, _ = _ledger(tau, age, (), None)
    state = _build({}, None, fdust_frac=fdust).predict_state(_params(tau, age))
    credit = 10.0 ** float(np.asarray(state.derived["log_L_lyc_dust"]))
    assert credit == pytest.approx(fdust * e_raw, rel=2e-2)
