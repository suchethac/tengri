# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate tengri's AGN informative priors against AGNfitter-rX.

Each of the eight functions in ``tengri.parameters.agn_priors`` (also
reachable as ``tengri.agn.priors``) is a transcription of one branch of
AGNfitter-rX's ``functions/PRIORS_AGNfitter.py`` (tag ``AGNfitter-rX_v0.1``).
This file transcribes the SAME upstream branches independently, directly in
numpy (not by importing tengri's implementation), and compares numeric
output at >= 5 inputs per prior, including the branch/regime boundaries,
with a tolerance of ``max|delta| < 1e-9``.

A previous implementation in this module (three differently-named functions,
removed) had a contract test that compared its own formula to itself
(``tests/contract/test_agn_priors.py`` before this rewrite) and so never
caught that it misstated every upstream branch it claimed to reproduce; this
file exists specifically to close that gap.

Reference: ``functions/PRIORS_AGNfitter.py``, AGNfitter-rX tag
``AGNfitter-rX_v0.1`` (upstream source path recorded in the audit that
motivated this test:
``~/.claude/jobs/da5e6ce4/tmp/AGNfitter-rX/functions/PRIORS_AGNfitter.py``).
Tolerance: ``max|delta| < 1e-9`` (numpy float64 vs tengri's JAX float64).

References
----------
.. [1] L. N. Martinez-Ramirez et al., "AGNFITTER-RX: Modeling the
   radio-to-X-ray spectral energy distributions of AGNs," A&A, 688, A46
   (2024). arXiv:2405.12111. doi:10.1051/0004-6361/202449329.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.parameters.agn_priors import (
    AGNFITTER_HARD_REJECT,
    prior_agn_fraction,
    prior_energy_balance,
    prior_ir_syn_fraction,
    prior_ir_xrays,
    prior_low_agn_fraction,
    prior_midir_uv,
    prior_stellar_mass,
    prior_uv_xrays,
)

pytestmark = pytest.mark.crossval

_TOL = 1e-9
_UP_REJECT = -9999.0


def _up_gaussian(mu, sigma, par):
    """Transcription of ``Gaussian_prior`` (PRIORS_AGNfitter.py:428-430)."""
    return np.log(1.0 / (np.sqrt(2.0 * np.pi) * sigma)) - 0.5 * ((par - mu) / sigma) ** 2


def _up_characteristic_mag(z):
    """Transcription of ``characteristic_mag`` (PRIORS_AGNfitter.py:172,406)."""
    return -35.4 * (1.0 + z) ** 0.524 / (1.0 + (1.0 + z) ** 0.678)


# ===========================================================================
# 1. prior_energy_balance vs upstream prior_energy_balance (:78-106)
# ===========================================================================
def _up_energy_balance(l_gal_att, l_sb_emit, mode):
    if l_sb_emit < l_gal_att:
        return _UP_REJECT
    if mode == "flexible":
        return 0.0
    frac = np.log10(l_sb_emit / l_gal_att)
    return _up_gaussian(0.0, 0.1, frac)


@pytest.mark.parametrize(
    "l_gal_att, l_sb_emit, mode",
    [
        (1.0e44, 1.0e44, "flexible"),  # boundary: exact equality, not rejected
        (1.0e44, 1.0e44, "restrictive"),  # boundary: exact equality -> Gaussian peak
        (1.0e44, 0.5e44, "flexible"),  # Lsb < Lgal -> hard reject, BOTH modes
        (1.0e44, 0.5e44, "restrictive"),  # same reject, restrictive mode
        (1.0e44, 2.0e44, "flexible"),  # physical, flexible -> flat 0
        (1.0e44, 2.0e44, "restrictive"),  # physical, restrictive -> Gaussian
        (5.0e42, 5.0e42 * 10**0.05, "restrictive"),  # small positive offset
    ],
)
def test_prior_energy_balance_matches_upstream(l_gal_att, l_sb_emit, mode):
    expected = _up_energy_balance(l_gal_att, l_sb_emit, mode)
    got = float(prior_energy_balance(l_gal_att, l_sb_emit, mode=mode))
    assert abs(got - expected) < _TOL, f"got={got} expected={expected}"
    if expected == _UP_REJECT:
        assert got == AGNFITTER_HARD_REJECT == -9999.0


# ===========================================================================
# 2. prior_stellar_mass vs upstream prior_stellar_mass (:201-210)
# ===========================================================================
def _up_stellar_mass(ga):
    return _up_gaussian(4.5, 1.5, ga)


@pytest.mark.parametrize("ga", [-5.0, 0.0, 3.0, 4.5, 10.0])
def test_prior_stellar_mass_matches_upstream(ga):
    expected = _up_stellar_mass(ga)
    got = float(prior_stellar_mass(ga))
    assert abs(got - expected) < _TOL


# ===========================================================================
# 3. prior_agn_fraction vs upstream prior_AGNfraction (:109-199, tail :167-198)
# ===========================================================================
def _up_agn_fraction(bbb_flux_1500, gal_flux_1500, data_flux_1500, dlum, z):
    agn_frac_1500 = np.log10(bbb_flux_1500 / gal_flux_1500)
    lumfactor = 4.0 * np.pi * dlum**2
    data_lum_1500 = lumfactor * data_flux_1500
    abs_mag_data = 51.6 - 2.5 * np.log10(data_lum_1500)
    char_mag = _up_characteristic_mag(z)
    if abs_mag_data > (char_mag - 1.0):
        return _up_gaussian(-2.0, 2.0, agn_frac_1500)
    if agn_frac_1500 < 0:
        return _UP_REJECT
    return _up_gaussian(2.0, 2.0, agn_frac_1500)


def _flux_1500_for_abs_mag(abs_mag_data, dlum):
    """Solve for data_flux_1500 giving the requested abs_mag_data at this dlum."""
    lumfactor = 4.0 * np.pi * dlum**2
    data_lum_1500 = 10.0 ** ((51.6 - abs_mag_data) / 2.5)
    return data_lum_1500 / lumfactor


_Z = 2.0
_DLUM = 1.0
_CHAR_MAG = _up_characteristic_mag(_Z)
# Faint (galaxy-regime) and bright (QSO-regime) 1500A data points, straddling
# each function's own threshold (char_mag - 1 for prior_agn_fraction,
# char_mag - 3 for prior_low_agn_fraction).
_FLUX_GALAXY_REGIME = _flux_1500_for_abs_mag(_CHAR_MAG - 1.0 + 2.0, _DLUM)  # faint -> galaxy
_FLUX_QSO_REGIME = _flux_1500_for_abs_mag(_CHAR_MAG - 1.0 - 2.0, _DLUM)  # bright -> QSO
_FLUX_LOW_NARROW_REGIME = _flux_1500_for_abs_mag(_CHAR_MAG - 3.0 + 2.0, _DLUM)  # faint -> narrow
_FLUX_LOW_WIDE_REGIME = _flux_1500_for_abs_mag(_CHAR_MAG - 3.0 - 2.0, _DLUM)  # bright -> wide
_FLUX_AT_AGN_BOUNDARY = _flux_1500_for_abs_mag(_CHAR_MAG - 1.0, _DLUM)  # exact regime boundary


@pytest.mark.parametrize(
    "bbb_flux_1500, gal_flux_1500, data_flux_1500",
    [
        (1.0e-27, 1.0e-25, _FLUX_GALAXY_REGIME),  # galaxy regime, AGNfrac1500<0
        (1.0e-23, 1.0e-25, _FLUX_GALAXY_REGIME),  # galaxy regime, AGNfrac1500>0
        (1.0e-27, 1.0e-25, _FLUX_QSO_REGIME),  # QSO regime, AGNfrac1500<0 -> reject
        (1.0e-23, 1.0e-25, _FLUX_QSO_REGIME),  # QSO regime, AGNfrac1500>0 -> Gaussian
        (1.0e-25, 1.0e-25, _FLUX_AT_AGN_BOUNDARY),  # exact abs_mag_data boundary
    ],
)
def test_prior_agn_fraction_matches_upstream(bbb_flux_1500, gal_flux_1500, data_flux_1500):
    expected = _up_agn_fraction(bbb_flux_1500, gal_flux_1500, data_flux_1500, _DLUM, _Z)
    got = float(prior_agn_fraction(bbb_flux_1500, gal_flux_1500, data_flux_1500, _DLUM, _Z))
    assert abs(got - expected) < _TOL, f"got={got} expected={expected}"
    if expected == _UP_REJECT:
        assert got == AGNFITTER_HARD_REJECT == -9999.0


# ===========================================================================
# 4. prior_low_agn_fraction vs upstream prior_low_AGNfraction (:360-425, tail :412-423)
# ===========================================================================
def _up_low_agn_fraction(bbb_flux_1500, gal_flux_1500, data_flux_1500, dlum, z):
    agn_frac_1500 = np.log10(bbb_flux_1500 / gal_flux_1500)
    lumfactor = 4.0 * np.pi * dlum**2
    data_lum_1500 = lumfactor * data_flux_1500
    abs_mag_data = 51.6 - 2.5 * np.log10(data_lum_1500)
    char_mag = _up_characteristic_mag(z)
    if abs_mag_data > (char_mag - 3.0):
        return _up_gaussian(-2.0, 0.5, agn_frac_1500)
    return _up_gaussian(-2.0, 2.0, agn_frac_1500)


@pytest.mark.parametrize(
    "bbb_flux_1500, gal_flux_1500, data_flux_1500",
    [
        (1.0e-27, 1.0e-25, _FLUX_LOW_NARROW_REGIME),  # narrow (sigma=0.5) regime
        (1.0e-23, 1.0e-25, _FLUX_LOW_NARROW_REGIME),  # narrow regime, other sign
        (1.0e-27, 1.0e-25, _FLUX_LOW_WIDE_REGIME),  # wide (sigma=2) regime
        (1.0e-23, 1.0e-25, _FLUX_LOW_WIDE_REGIME),  # wide regime, other sign
        (
            1.0e-25,
            1.0e-25,
            _flux_1500_for_abs_mag(_CHAR_MAG - 3.0, _DLUM),
        ),  # exact regime boundary
    ],
)
def test_prior_low_agn_fraction_matches_upstream(bbb_flux_1500, gal_flux_1500, data_flux_1500):
    expected = _up_low_agn_fraction(bbb_flux_1500, gal_flux_1500, data_flux_1500, _DLUM, _Z)
    got = float(prior_low_agn_fraction(bbb_flux_1500, gal_flux_1500, data_flux_1500, _DLUM, _Z))
    assert abs(got - expected) < _TOL, f"got={got} expected={expected}"


def test_prior_agn_fraction_and_low_agn_fraction_are_genuinely_different():
    """D7(b): a prior implementation conflated these two upstream functions.

    Same inputs, different sigma sets and different thresholds -- assert the
    two tengri functions do NOT collapse to the same number away from a
    coincidental point.
    """
    args = (1.0e-23, 1.0e-25, _FLUX_GALAXY_REGIME, _DLUM, _Z)
    a = float(prior_agn_fraction(*args))
    b = float(prior_low_agn_fraction(*args))
    assert abs(a - b) > 0.1


# ===========================================================================
# 5. prior_ir_syn_fraction vs upstream prior_IR_SYNfraction (:212-254, core :237-254)
# ===========================================================================
def _up_ir_syn_fraction(
    data_flux_rad, data_nu_rad, data_flux_ir, data_nu_ir, sb_flux_ir, syn_flux_ir
):
    syn_exp_ir = data_flux_rad * ((10**data_nu_ir / 10**data_nu_rad) ** (-0.75))
    syn_frac_ir = np.log10(syn_flux_ir / sb_flux_ir)
    ratio = data_flux_ir / syn_exp_ir
    if ratio < 2.0:
        return _up_gaussian(2.0, 2.0, syn_frac_ir)
    if ratio > 2.0:
        return _up_gaussian(-2.0, 2.0, syn_frac_ir)
    return None  # upstream leaves this branch undefined (neither if nor elif fires)


_NU_RAD = 9.0  # log10(Hz), ~1 GHz
_NU_IR = 12.5  # log10(Hz), cold-dust peak
_DATA_FLUX_RAD = 1.0e-27
_SYN_EXP_IR = _DATA_FLUX_RAD * ((10**_NU_IR / 10**_NU_RAD) ** (-0.75))


@pytest.mark.parametrize(
    "data_flux_ir, sb_flux_ir, syn_flux_ir",
    [
        (0.5 * _SYN_EXP_IR, 1.0e-25, 1.0e-27),  # ratio < 2 (radio explains IR)
        (0.5 * _SYN_EXP_IR, 1.0e-25, 1.0e-23),  # ratio < 2, other SYNfrac sign
        (10.0 * _SYN_EXP_IR, 1.0e-25, 1.0e-27),  # ratio > 2 (starburst dominates)
        (10.0 * _SYN_EXP_IR, 1.0e-25, 1.0e-23),  # ratio > 2, other SYNfrac sign
        (3.0 * _SYN_EXP_IR, 1.0e-25, 1.0e-25),  # ratio > 2, SYNfrac = 0
    ],
)
def test_prior_ir_syn_fraction_matches_upstream(data_flux_ir, sb_flux_ir, syn_flux_ir):
    expected = _up_ir_syn_fraction(
        _DATA_FLUX_RAD, _NU_RAD, data_flux_ir, _NU_IR, sb_flux_ir, syn_flux_ir
    )
    assert expected is not None
    got = float(
        prior_ir_syn_fraction(
            _DATA_FLUX_RAD, _NU_RAD, data_flux_ir, _NU_IR, sb_flux_ir, syn_flux_ir
        )
    )
    assert abs(got - expected) < _TOL, f"got={got} expected={expected}"


def test_prior_ir_syn_fraction_boundary_is_documented_not_upstream_matched():
    """At ratio == 2 exactly, upstream's own branch (neither ``if`` nor ``elif``
    fires) is undefined -- there is nothing to cross-validate against. This
    pins tengri's own documented tie-break (the ``< 2`` branch's complement,
    mu=-2) instead of asserting agreement with an undefined upstream value.
    """
    data_flux_ir = 2.0 * _SYN_EXP_IR  # exact ratio == 2 boundary
    got = float(
        prior_ir_syn_fraction(_DATA_FLUX_RAD, _NU_RAD, data_flux_ir, _NU_IR, 1.0e-25, 1.0e-25)
    )
    expected_tie_break = _up_gaussian(-2.0, 2.0, 0.0)  # syn_frac_ir = log10(1) = 0
    assert abs(got - expected_tie_break) < _TOL


# ===========================================================================
# 6. prior_uv_xrays vs upstream prior_UV_xrays (:256-299, core :258-265,291-297)
# ===========================================================================
def _up_uv_xrays(log_l2500a_data, log_l2kev_data):
    beta, gamma = 0.643, 6.8734
    log_l2500a_model = (log_l2kev_data - gamma) / beta
    ratio = log_l2500a_data - log_l2500a_model
    return _up_gaussian(0.0, 0.4, ratio)


@pytest.mark.parametrize(
    "log_l2kev_data, offset",
    [
        (28.0, 0.0),  # exact alpha_ox relation, ratio=0 (Gaussian peak)
        (28.0, 0.4),  # +1 sigma offset
        (28.0, -0.4),  # -1 sigma offset
        (30.0, 0.0),  # different L_2kev, still ratio=0
        (30.0, 0.8),  # +2 sigma offset
    ],
)
def test_prior_uv_xrays_matches_upstream(log_l2kev_data, offset):
    beta, gamma = 0.643, 6.8734
    log_l2500a_model = (log_l2kev_data - gamma) / beta
    log_l2500a_data = log_l2500a_model + offset
    expected = _up_uv_xrays(log_l2500a_data, log_l2kev_data)
    got = float(prior_uv_xrays(log_l2500a_data, log_l2kev_data))
    assert abs(got - expected) < _TOL, f"got={got} expected={expected}"


# ===========================================================================
# 7. prior_ir_xrays vs upstream prior_IR_XRays (:302-324, core :309-322)
# ===========================================================================
def _up_ir_xrays(log_f2_10kev_data, nulnu_6um):
    x = np.log10(nulnu_6um / 1e41)
    model = 22.9494264 + 1.024 * x - 0.047 * x**2
    ratio = log_f2_10kev_data - model
    return _up_gaussian(0.0, 0.5, ratio)


@pytest.mark.parametrize(
    "nulnu_6um, offset",
    [
        (1.0e41, 0.0),  # x=0, ratio=0
        (1.0e41, 0.5),  # x=0, +1 sigma
        (1.0e41, -0.5),  # x=0, -1 sigma
        (1.0e43, 0.0),  # x=2, ratio=0
        (1.0e39, 1.0),  # x=-2, +2 sigma
    ],
)
def test_prior_ir_xrays_matches_upstream(nulnu_6um, offset):
    x = np.log10(nulnu_6um / 1e41)
    model = 22.9494264 + 1.024 * x - 0.047 * x**2
    log_f2_10kev_data = model + offset
    expected = _up_ir_xrays(log_f2_10kev_data, nulnu_6um)
    got = float(prior_ir_xrays(log_f2_10kev_data, nulnu_6um))
    assert abs(got - expected) < _TOL, f"got={got} expected={expected}"


# ===========================================================================
# 8. prior_midir_uv vs upstream prior_midIR_UV (:327-357, core :333-355)
#
# CRITICAL fix (review round 1): upstream's prior_midIR_UV computes
# x = log10(tor_flux_6microns * lumfactor) - 27.30103, where
# tor_flux_6microns * lumfactor is a SPECIFIC luminosity L_nu(6um)
# [erg/s/Hz] -- NOT nu*L_nu. prior_IR_XRays instead explicitly forms
# nuLnu_6microns = 10**13.69897 * tor_flux_6microns * lumfactor (nu*L_nu,
# erg/s) before its own x = log10(nuLnu_6microns/1e41). A first version of
# this test (and of prior_midir_uv itself) fed the SAME nu*L_nu input to
# both formulas but re-used upstream's L_nu-calibrated -27.30103 constant
# directly on log10(nulnu_6um), silently off by log10(nu_6um) = 13.69897 in
# x (e.g. nulnu_6um=1e44: x=16.7 instead of x=3.0). Fixed by transcribing
# upstream's two call sites on their OWN units and proving algebraic
# equivalence below (13.69897 + 27.30103 = 41 exactly).
# ===========================================================================
_NU_6UM_HZ = 10.0**13.69897  # PRIORS_AGNfitter.py:306,331: "6 microns = 13.69897 log(Hz)"


def _up_midir_uv_from_l_nu_6um(log_l2500a_bbmodel, l_nu_6um):
    """Transcription of upstream's LITERAL formula (PRIORS_AGNfitter.py:333),
    which operates on tor_flux_6microns*lumfactor -- a SPECIFIC luminosity
    L_nu(6um) [erg/s/Hz], NOT nu*L_nu."""
    x = np.log10(l_nu_6um) - 27.30103
    model = (16.2530786 + 1.024 * x - 0.047 * x**2) / 0.643
    ratio = log_l2500a_bbmodel - model
    return _up_gaussian(0.0, 0.6, ratio)


def _up_midir_uv(log_l2500a_bbmodel, nulnu_6um):
    """Transcription on tengri's public nu*L_nu [erg/s] input, converting to
    upstream's own L_nu-based x internally via the 6-micron frequency."""
    l_nu_6um = nulnu_6um / _NU_6UM_HZ
    return _up_midir_uv_from_l_nu_6um(log_l2500a_bbmodel, l_nu_6um)


def test_prior_midir_uv_x_matches_upstream_specific_luminosity_path():
    """Prove the fix directly against upstream's own tor_flux_6microns*lumfactor
    path (PRIORS_AGNfitter.py:331-333), converting to nu*L_nu explicitly via
    the 6-micron frequency (:306,331), rather than against tengri's own
    formula. 13.69897 + 27.30103 = 41 exactly, so upstream's L_nu-based
    x = log10(L_nu_6um) - 27.30103 and tengri's nu*L_nu-based
    x = log10(nuLnu_6um/1e41) are the SAME number for the SAME physical torus
    flux."""
    assert abs((13.69897 + 27.30103) - 41.0) < _TOL
    l_nu_6um = 3.7e29  # arbitrary torus specific luminosity [erg/s/Hz]
    x_upstream_l_nu_path = np.log10(l_nu_6um) - 27.30103
    nulnu_6um = _NU_6UM_HZ * l_nu_6um  # convert to nu*L_nu [erg/s], as prior_IR_XRays does
    x_tengri_nulnu_path = np.log10(nulnu_6um / 1e41)
    assert abs(x_tengri_nulnu_path - x_upstream_l_nu_path) < _TOL

    # And prior_midir_uv, given nulnu_6um, reproduces the model prediction
    # upstream's own L_nu-based x would give:
    model_upstream = (
        16.2530786 + 1.024 * x_upstream_l_nu_path - 0.047 * x_upstream_l_nu_path**2
    ) / 0.643
    got = float(prior_midir_uv(model_upstream, nulnu_6um))  # ratio == 0 exactly
    expected = _up_gaussian(0.0, 0.6, 0.0)
    assert abs(got - expected) < _TOL


def test_prior_ir_xrays_and_prior_midir_uv_agree_on_x():
    """Both priors share tengri's private ``_x_from_nulnu_6um`` helper -- for
    the SAME nu*L_nu(6um) input, the Stern-correlation variable ``x`` each
    uses internally must be identical (this was the exact quantity that
    silently disagreed before the fix)."""
    from tengri.parameters.agn_priors import _x_from_nulnu_6um

    for nulnu_6um in (1.0e41, 1.0e43, 1.0e45):
        x = float(_x_from_nulnu_6um(nulnu_6um))
        model_ir_xrays = 22.9494264 + 1.024 * x - 0.047 * x**2
        model_midir_uv = (16.2530786 + 1.024 * x - 0.047 * x**2) / 0.643
        expected_peak_ir_xrays = _up_gaussian(0.0, 0.5, 0.0)
        expected_peak_midir_uv = _up_gaussian(0.0, 0.6, 0.0)
        got_ir_xrays = float(prior_ir_xrays(model_ir_xrays, nulnu_6um))
        got_midir_uv = float(prior_midir_uv(model_midir_uv, nulnu_6um))
        assert abs(got_ir_xrays - expected_peak_ir_xrays) < _TOL
        assert abs(got_midir_uv - expected_peak_midir_uv) < _TOL


@pytest.mark.parametrize(
    "nulnu_6um, offset",
    [
        (1.0e44, 0.0),  # ratio=0 (Gaussian peak)
        (1.0e44, 0.6),  # +1 sigma
        (1.0e44, -0.6),  # -1 sigma
        (1.0e45, 0.0),  # different torus luminosity, still ratio=0
        (1.0e43, 1.2),  # +2 sigma, different torus luminosity
    ],
)
def test_prior_midir_uv_matches_upstream(nulnu_6um, offset):
    x = np.log10(nulnu_6um / 1e41)
    model = (16.2530786 + 1.024 * x - 0.047 * x**2) / 0.643
    log_l2500a_bbmodel = model + offset
    expected = _up_midir_uv(log_l2500a_bbmodel, nulnu_6um)
    got = float(prior_midir_uv(log_l2500a_bbmodel, nulnu_6um))
    assert abs(got - expected) < _TOL, f"got={got} expected={expected}"


def test_prior_midir_uv_disagrees_sharply_with_equal_luminosities():
    """D7(c): a prior implementation compared L_mir and L_uv directly.

    At equal nominal log-luminosities (the direct-comparison "no penalty"
    point), the actual Stern+alpha_ox composite relation predicts a UV
    luminosity nowhere near that value -- the two formulas are not
    approximations of each other at any normalization.
    """
    log_l2500a_bbmodel = 45.0
    nulnu_6um = 10**45.0
    x = np.log10(nulnu_6um / 1e41)
    log_l2500a_tomodel = (16.2530786 + 1.024 * x - 0.047 * x**2) / 0.643
    # The composite relation predicts ~30.5 dex at this torus flux, ~14.5 dex
    # away from the naive "equal luminosities" point -- nowhere near it on
    # any astrophysical scale.
    assert abs(log_l2500a_tomodel - 45.0) > 5.0
    got = float(prior_midir_uv(log_l2500a_bbmodel, nulnu_6um))
    direct_comparison_lp = _up_gaussian(0.0, 0.6, 0.0)  # what a naive L_mir==L_uv model gives
    assert abs(got - direct_comparison_lp) > 50.0  # log-prior units: (14.5/0.6)^2/2 ~ 292
