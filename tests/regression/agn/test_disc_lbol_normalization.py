# SPDX-License-Identifier: BSD-3-Clause
"""Regression: physical discs are luminosity-first (#846, ADR-0020).

The physical accretion discs (``multicolor``, ``kubota_done``) derive the
Eddington ratio from the requested ``agn_log_lbol`` and ``agn_log_mbh``
(lambda_Edd = L_bol / L_Edd), so the disc shape is self-consistent with L_bol
and the delivered bolometric luminosity matches the request. ``agn_log_ledd`` is
retired (no effect on these blocks); setting it on a composable model raises a
build-time warning.

Data-free (pure disc physics); runs in CI.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

from tengri.components.agn._params import DEFAULT_AGN_COS_INC
from tengri.components.agn.disc import kubota_done_disc, multicolor_disc
from tengri.utils.physics_constants import L_SUN

# Wavelength grid spanning the hard X-ray corona (0.01 A) to the far-IR (5 mm),
# so the trapezoidal bolometric integral captures the full SED (the hot corona
# emits below ~1 A; a UV-optical grid would miss it and understate L_bol).
_WAVE = jnp.geomspace(0.01, 5.0e6, 12000)


def _l_bol_delivered(l_nu):
    nu = 2.99792458e18 / _WAVE  # Hz (c in A/s)
    order = jnp.argsort(nu)
    return float(jnp.trapezoid(np.asarray(l_nu)[order], np.asarray(nu)[order]))


def _p(disc_fn, log_lbol, cos_inc):
    """Delivered power [erg/s] at ``cos_inc`` (M_BH = 1e8, a = 0)."""
    l_nu = disc_fn(
        _WAVE, agn_log_lbol=log_lbol, agn_log_mbh=8.0, agn_a_spin=0.0, agn_cos_inc=cos_inc
    )
    return _l_bol_delivered(l_nu)


@pytest.mark.parametrize("disc_fn", [multicolor_disc, kubota_done_disc])
@pytest.mark.parametrize("log_lbol", [11.0, 12.0, 12.4])
def test_disc_delivers_requested_lbol(disc_fn, log_lbol):
    """``agn_log_lbol`` is the accretion power: the SED integrates to it at cos i = 0.5.

    The discs return ``L_nu(i) = 2 cos i D_nu + H_nu`` with
    ``int (D_nu + H_nu) dnu = L_acc = 10**agn_log_lbol L_sun``, so the line-of-sight power
    equals ``L_acc`` at ``cos i = 0.5`` (M_BH = 1e8), to <1%.
    """
    ratio = _p(disc_fn, log_lbol, 0.5) / (10.0**log_lbol * L_SUN)
    assert abs(ratio - 1.0) < 0.01, (
        f"{disc_fn.__name__} delivered L_bol ratio {ratio:.4f} (want 1.00 +/- 1%) "
        f"at agn_log_lbol={log_lbol}, cos i = 0.5"
    )


@pytest.mark.parametrize("log_lbol", [11.0, 12.0, 12.4])
def test_multicolor_default_inclination_is_two_cos_i_times_lbol(log_lbol):
    """A pure disc (no corona) integrates to ``2 cos i L_acc`` at the default inclination.

    ``D_nu`` carries all of ``L_acc``, so ``int L_nu(i) dnu = 2 cos i L_acc``
    (2 cos 30 deg = 1.732 at the default ``DEFAULT_AGN_COS_INC``), to the 1% of the
    energy test above.
    """
    ratio = _p(multicolor_disc, log_lbol, DEFAULT_AGN_COS_INC) / (10.0**log_lbol * L_SUN)
    assert ratio == pytest.approx(2.0 * DEFAULT_AGN_COS_INC, rel=0.01)


@pytest.mark.parametrize("log_lbol", [11.0, 12.0, 12.4])
def test_kubota_done_default_inclination_matches_disc_plus_corona_budget(log_lbol):
    """``int L_nu(i) dnu = 2 cos i D + H`` with D, H measured from two evaluations.

    ``P(c) = 2 c D + H`` is linear in ``c``: ``P(1) - P(0) = 2 D`` and ``P(0) = H``. The
    parts close the budget, ``D + H = L_acc`` (1%), and the power at the default inclination
    equals ``2 cos i D + H`` (the ratio to ``L_acc`` is ``(2 cos i D + H) / (D + H)``).
    """
    p0 = _p(kubota_done_disc, log_lbol, 0.0)
    p1 = _p(kubota_done_disc, log_lbol, 1.0)
    d_disc, h_corona = 0.5 * (p1 - p0), p0
    l_acc = 10.0**log_lbol * L_SUN
    assert (d_disc + h_corona) / l_acc == pytest.approx(1.0, abs=0.01)
    expected = (2.0 * DEFAULT_AGN_COS_INC * d_disc + h_corona) / (d_disc + h_corona)
    delivered = _p(kubota_done_disc, log_lbol, DEFAULT_AGN_COS_INC) / (d_disc + h_corona)
    assert delivered == pytest.approx(expected, rel=1e-6)
    assert 1.0 < delivered < 2.0 * DEFAULT_AGN_COS_INC


@pytest.mark.parametrize("disc_fn", [multicolor_disc, kubota_done_disc])
def test_disc_agn_log_ledd_is_noop(disc_fn):
    """agn_log_ledd no longer affects the physical discs (derived from L_bol)."""
    wave = jnp.geomspace(50.0, 1.0e5, 3000)
    ref = disc_fn(wave, agn_log_lbol=12.0, agn_log_mbh=8.0, agn_log_ledd=-1.0)
    for ledd in (-3.0, -0.3, 0.0):
        alt = disc_fn(wave, agn_log_lbol=12.0, agn_log_mbh=8.0, agn_log_ledd=ledd)
        assert jnp.allclose(ref, alt, rtol=1e-10), (
            f"{disc_fn.__name__}: agn_log_ledd={ledd} changed the SED (should be ignored)"
        )


def test_disc_peak_blueshifts_with_lbol_at_fixed_mass():
    """Higher L_bol at fixed M_BH → higher lambda_Edd → hotter T_in → bluer peak."""
    wave = jnp.geomspace(50.0, 1.0e5, 4000)
    peak = {}
    for log_lbol in (11.2, 12.2):
        lnu = np.asarray(multicolor_disc(wave, agn_log_lbol=log_lbol, agn_log_mbh=8.0))
        sel = (np.asarray(wave) > 100.0) & (np.asarray(wave) < 1.0e4) & (lnu > 0)
        nu_lnu = lnu * (2.99792458e18 / np.asarray(wave))
        peak[log_lbol] = np.asarray(wave)[sel][np.argmax(nu_lnu[sel])]
    assert peak[12.2] < peak[11.2], (
        f"higher L_bol should peak bluer: peak(12.2)={peak[12.2]:.0f} A vs "
        f"peak(11.2)={peak[11.2]:.0f} A"
    )


def test_agn_log_ledd_warns_when_set_on_composable():
    """Setting agn_log_ledd on a composable multicolor/kubota model warns at
    construction (jit-safe, Parameters.__init__)."""
    from tengri import Fixed, Parameters

    with pytest.warns(UserWarning, match="agn_log_ledd has no effect"):
        Parameters(
            agn_model="composable",
            agn_disc_block="multicolor",
            agn_log_ledd=Fixed(-0.5),
        )


def test_agn_log_ledd_default_does_not_warn():
    """The default agn_log_ledd (unset) must not warn."""
    from tengri import Parameters

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        Parameters(agn_model="composable", agn_disc_block="multicolor")
