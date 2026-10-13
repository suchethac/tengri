# SPDX-License-Identifier: BSD-3-Clause
"""Cue counts ionizing photons from every stellar age by default; the 100 Myr cut is an option.

Until now Cue's Q_H and ionizing-spectrum shape were built from SSP bins younger than
100 Myr only (``log10(age/yr) <= 8``). A quiescent galaxy therefore got ~zero nebular
emission, while ``derived['log_nion']`` (and CIGALE) counted every age. The default is now
every age; ``neb={'type': 'cue', 'lyc_max_age_yr': 1e8}`` restores the cut exactly.

The tracked FSPS/Padova grid has no old-star Lyman continuum, so these tests add a 1e5 K
post-AGB-like continuum to its >= 1 Gyr bins (``tests/_old_lyc_ssp.py``), normalized to
5e40 photons/s/Msun (MIST grids carry 3e40-1e41 there).

Data-gated: needs ``data/fsps_prsc_miles_chabrier.h5`` and ``data/cue_weights.npz``.
"""

from __future__ import annotations

import warnings
from functools import cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, Observation, Photometry, SEDModel
from tengri.components.nebular._recombination_coeffs import lyc_dust_escape_factor
from tests._old_lyc_ssp import FSPS_PRSC, ssp_with_old_lyc

_HALPHA_AA = 6564.61
#: Case B, Osterbrock & Ferland (2006) Table 4.4, T = 1e4 K, n_e = 100 cm^-3 [erg per photon].
_CASE_B_ERG = 1.37e-12
_MASS = 10.0  # log10(M_formed / Msun)
_LOG_Z_ABS = -1.848  # solar metallicity, log10(Z) absolute
_CUE_WEIGHTS = Path("data/cue_weights.npz")

#: Quenched population: delayed SFH, tau = 0.3 Gyr, age 10 Gyr (the star formation ended long ago).
_QUENCHED = (0.3, 10.0)

#: ``Halpha`` and ``nion`` of the 100 Myr cut, measured on the unmodified tree (commit 71021d4eb)
#: on the old-LyC grid above. {(tau_gyr, age_gyr): (L_Halpha [erg/s], Q_H [photons/s])}.
_CUT_BASELINE = {
    (0.3, 10.0): (9.56681297974834e29, 5.022708261307399e50),
    (1.0, 10.0): (1.1611224273751623e39, 1.3372329010187178e51),
    (3.0, 1.0): (4.544649942429051e42, 3.269331004026739e54),
    (1.0, 5.0): (8.968849307789703e40, 6.499872e52),
}
#: Combine output of the cut at the last case above (tau = 1, age = 5): frozen from the tree
#: before the change, float64 arithmetic.
_CUT_COMBINE = {
    "gas_logqion": 52.81579674083042,
    "ionspec_index1": 22.010348216280047,
    "ionspec_index2": 10.796958660763432,
    "ionspec_index3": 6.447468707832534,
    "ionspec_index4": 2.629120636340638,
    "ionspec_logLratio1": 2.6227516560031034,
    "ionspec_logLratio2": 0.41155866134053554,
    "ionspec_logLratio3": 0.6628467330311736,
}

pytestmark = pytest.mark.skipif(
    not (FSPS_PRSC.is_file() and _CUE_WEIGHTS.is_file()),
    reason="needs data/fsps_prsc_miles_chabrier.h5 and data/cue_weights.npz",
)


@cache
def _ssp():
    return ssp_with_old_lyc()


@cache
def _model(lyc_max_age_yr=None, fesc=0.0, fdust_frac=0.0):
    neb = {
        "type": "cue",
        "neb_fesc": Fixed(fesc),
        "neb_fdust_frac": Fixed(fdust_frac),
        "neb_logZ_gas": Fixed(0.0),
        "neb_logU": Fixed(-2.0),
        "all_params": Fixed(DEFAULT),
    }
    if lyc_max_age_yr is not None:
        neb["lyc_max_age_yr"] = lyc_max_age_yr
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
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.0),
                "all_params": Fixed(DEFAULT),
            },
            met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
            neb=neb,
            redshift=Fixed(0.0),
        )


def _params(tau, age):
    return {
        "sfh_delayed_tau_gyr": jnp.asarray(tau),
        "sfh_delayed_age_gyr": jnp.asarray(age),
        "sfh_delayed_log_total_mass": jnp.asarray(_MASS),
    }


def _halpha_and_qh(model, tau, age):
    """``(L(Halpha) [erg/s], Q_H [photons/s])`` of the model at one delayed-SFH point."""
    derived = model.predict_state(_params(tau, age)).derived
    waves = np.asarray(derived["line_waves"], dtype=np.float64)
    lums = np.asarray(derived["line_lums"], dtype=np.float64)
    i = int(np.argmin(np.abs(waves - _HALPHA_AA)))
    return float(lums[i]), float(np.asarray(derived["nion"]))


@pytest.mark.regression_paper
def test_quenched_halpha_is_case_b_times_the_all_age_qh():
    """Quenched galaxy: ``L(Halpha) = 1.37e-12 erg * Q_H(all ages)`` within 2%.

    Osterbrock & Ferland (2006) Table 4.4 (Case B, T = 1e4 K, n_e = 100 cm^-3).
    ``nion`` is the stellar-published Q_H, which has always counted every age. With the
    100 Myr cut Cue returned ~1e-9 of this for the same population.
    """
    halpha, q_h = _halpha_and_qh(_model(), *_QUENCHED)
    ratio = halpha / (_CASE_B_ERG * q_h)
    assert abs(ratio - 1.0) < 0.02, (
        f"L(Halpha)/(1.37e-12 erg * Q_H) = {ratio:.4f} for the quenched population; "
        "Cue is not counting the old-star Lyman continuum"
    )


@pytest.mark.regression_paper
def test_default_is_all_ages_and_explicit_all_is_identical():
    """``lyc_max_age_yr='all'`` is the default, not a different code path."""
    assert _halpha_and_qh(_model("all"), *_QUENCHED) == _halpha_and_qh(_model(), *_QUENCHED)


@pytest.mark.regression_bug
@pytest.mark.parametrize("case", list(_CUT_BASELINE), ids=lambda c: f"tau{c[0]}_age{c[1]}")
def test_cut_option_reproduces_the_100_myr_result(case):
    """``lyc_max_age_yr=1e8`` gives the pre-change Halpha and Q_H, to float noise.

    Pinned from the unmodified tree. The quenched case is ~zero Halpha by construction
    (the cut leaves only the exponential tail of the star formation).
    """
    halpha, q_h = _halpha_and_qh(_model(1.0e8), *case)
    want_halpha, want_q_h = _CUT_BASELINE[case]
    assert halpha == pytest.approx(want_halpha, rel=1e-6)
    assert q_h == pytest.approx(want_q_h, rel=1e-5)


@pytest.mark.regression_bug
def test_cut_option_combine_is_bit_identical_to_the_old_arithmetic():
    """The float64 Q_H and shape combine under the cut equal the frozen pre-change values."""
    model = _model(1.0e8)
    derived = model.predict_state(_params(1.0, 5.0)).derived
    got = model._nebular_backend._compute_weighted_cue_params(
        jnp.asarray(derived["age_weights"]),
        jnp.log10(jnp.asarray(derived["ssp_ages_yr"])),
        _LOG_Z_ABS,
    )
    for key, want in _CUT_COMBINE.items():
        assert float(got[key]) == pytest.approx(want, rel=1e-13, abs=1e-13), key


@pytest.mark.regression_bug
def test_quenched_population_has_emission_by_default_and_none_under_the_cut():
    """The failure this option exists for: a quiescent population gets a nebular spectrum."""
    all_ages, _ = _halpha_and_qh(_model(), *_QUENCHED)
    cut, _ = _halpha_and_qh(_model(1.0e8), *_QUENCHED)
    assert cut / all_ages < 1e-6
    # 5e40 photons/s/Msun x 1e10 Msun x ~0.9 of the mass formed before the last 3 Gyr
    assert all_ages > 1e38


#: Relative change of Halpha from the 100 Myr cut to all ages, measured on the old-LyC grid.
#: {(tau_gyr, age_gyr): change}. The old-star share of Q_H is 5e-7 (tau 3, age 1 Gyr) to
#: 0.12% (tau 3, age 5 Gyr); tau 1 Gyr at 5 Gyr is a mixed population (0.74%).
_YOUNG_SF_CHANGE = {
    (3.0, 1.0): 4.6171309242559744e-07,
    (1.0, 1.0): 8.576091550249743e-07,
    (3.0, 5.0): 0.0012259490058386113,
}
_MIXED_CHANGE = {(1.0, 5.0): 0.00742887834618533}


def _change_from_the_cut(case):
    cut, _ = _halpha_and_qh(_model(1.0e8), *case)
    full, _ = _halpha_and_qh(_model(), *case)
    return full / cut - 1.0


@pytest.mark.regression_bug
@pytest.mark.parametrize("case", list(_YOUNG_SF_CHANGE), ids=lambda c: f"tau{c[0]}_age{c[1]}")
def test_star_forming_histories_change_by_at_most_half_a_percent(case):
    """Star-forming SFHs: Halpha moves by < 0.5% from the cut to all ages, by the pinned amount."""
    change = _change_from_the_cut(case)
    assert 0.0 <= change < 0.005, f"{change:+.5%}"
    assert change == pytest.approx(_YOUNG_SF_CHANGE[case], rel=1e-2, abs=2e-7)


@pytest.mark.regression_bug
@pytest.mark.parametrize("case", list(_MIXED_CHANGE), ids=lambda c: f"tau{c[0]}_age{c[1]}")
def test_mixed_population_change_is_the_pinned_old_star_share(case):
    """tau 1 Gyr at 5 Gyr (old stars carry 0.7% of the photons) moves Halpha by that share."""
    assert _change_from_the_cut(case) == pytest.approx(_MIXED_CHANGE[case], rel=1e-2)


@pytest.mark.gradient
def test_gradients_are_finite_with_old_populations_counted():
    """d(L_Halpha)/d(SFH parameters) is finite and follows the mass for a quenched SFH.

    L_Halpha is linear in the formed mass, so dL/dlog10(M) = ln(10) L exactly.
    """
    model = _model()

    def halpha(p):
        return model.predict_properties(p, names=("halpha",))["halpha"]

    params = _params(*_QUENCHED)
    grads = jax.grad(halpha)(params)
    for name, g in grads.items():
        assert np.isfinite(float(g)), name
    assert float(grads["sfh_delayed_log_total_mass"]) == pytest.approx(
        np.log(10.0) * float(halpha(params)), rel=1e-4
    )


@pytest.mark.gradient
def test_combine_gradient_wrt_weights_is_finite_with_zero_weight_bins():
    """The log-domain combine over every age has finite d/d(weights), zero-weight bins included."""
    model = _model()
    backend = model._nebular_backend
    derived = model.predict_state(_params(*_QUENCHED)).derived
    log_ages = jnp.log10(jnp.asarray(derived["ssp_ages_yr"]))
    weights = jnp.asarray(derived["age_weights"])

    def gas_qion(w):
        return backend._compute_weighted_cue_params(w, log_ages, _LOG_Z_ABS)["gas_logqion"]

    def shape_index(w):
        return backend._compute_weighted_cue_params(w, log_ages, _LOG_Z_ABS)["ionspec_index4"]

    for fn in (gas_qion, shape_index):
        g = np.asarray(jax.grad(fn)(weights))
        assert np.all(np.isfinite(g))
    assert (np.asarray(weights) == 0.0).any(), "fixture lost its zero-weight bins"


@pytest.mark.conservation
@pytest.mark.parametrize("case", [(0.3, 10.0), (1.0, 5.0), (3.0, 1.0)], ids=str)
def test_cue_qh_and_stellar_log_nion_are_the_same_quantity(case):
    """Cue's population ``gas_logqion`` equals the stellar-published ``log_nion``.

    The same ionizing photons the stellar SED loses to ``neb_fesc`` / ``neb_fdust_frac`` are
    the ones Cue re-emits as lines: the two Q_H values must agree to the table interpolation.
    """
    model = _model()
    derived = model.predict_state(_params(*case)).derived
    got = model._nebular_backend._compute_weighted_cue_params(
        jnp.asarray(derived["age_weights"]),
        jnp.log10(jnp.asarray(derived["ssp_ages_yr"])),
        _LOG_Z_ABS,
    )["gas_logqion"]
    assert float(got) == pytest.approx(float(derived["log_nion"]), abs=0.02)


@pytest.mark.conservation
def test_escape_and_dust_share_scale_lines_identically_in_both_modes():
    """``neb_fesc`` / ``neb_fdust_frac`` scale Q_H by one factor, whatever the age cut."""
    for cut in (None, 1.0e8):
        base, _ = _halpha_and_qh(_model(cut), 1.0, 5.0)
        scaled, _ = _halpha_and_qh(_model(cut, fesc=0.3, fdust_frac=0.2), 1.0, 5.0)
        # absolute dust share = (1 - fesc) * fdust_frac (#2436); lines carry the CIGALE k-factor
        k = float(lyc_dust_escape_factor(0.3, (1.0 - 0.3) * 0.2))
        assert scaled / base == pytest.approx(k, rel=1e-6), cut


@pytest.mark.limit
def test_empty_population_gets_no_emission_and_a_finite_gradient():
    """No stars at all: ``gas_logqion`` is the -99 'no photons' value, everything finite."""
    backend = _model()._nebular_backend
    log_ages = jnp.log10(
        jnp.asarray(backend._ssp_log_age_yr * 0.0 + 10.0**backend._ssp_log_age_yr)
    )

    def combine(w):
        return backend._compute_weighted_cue_params(w, log_ages, _LOG_Z_ABS)

    zeros = jnp.zeros(log_ages.shape)
    out = combine(zeros)
    assert float(out["gas_logqion"]) == -99.0
    assert all(np.isfinite(float(v)) for v in out.values())
    g = jax.grad(lambda w: combine(w)["gas_logqion"])(zeros)
    assert np.all(np.isfinite(np.asarray(g)))
