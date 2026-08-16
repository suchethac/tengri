# SPDX-License-Identifier: BSD-3-Clause
"""Stellar Lyman-continuum absorption under two-component dust (young-only vs all).

Regression for #824 (no negative flux / no leak below 912 Å) plus the young-only
default. ``neb_fesc`` is a *birth-cloud* escape fraction, so by default only the
young/birth-cloud stellar LyC is reprocessed; the old/diffuse stellar LyC passes
through (matches bagpipes ``model_galaxy``, which zeros only ``spectrum_bc[<912]``).
``DustSEDComponentConfig.lyc_absorb_all=True`` absorbs *all* stellar LyC
(old + young), matching FSPS (``frac_obrun``) and CIGALE (absorbed_old+young).

The nebular component publishes ``lyc_transmission = where(λ<912, neb_fesc, 1)``;
the two-component dust applies it — young-weighted by default, uniform under
``lyc_absorb_all`` — to the per-age ``lnu_age`` reconstruction, so:

* ``predict()`` never goes negative below 912 Å (the #824 phantom is gone), and
* below 912 the surviving stellar LyC is young×fesc + old (default), or
  (young+old)×fesc (absorb_all).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust.two_component import DustSEDComponent, DustSEDComponentConfig
from tengri.protocols.component import ForwardState
from tests._bounds import assert_non_negative

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]

# Grid straddling the Lyman limit (first two points are LyC, λ < 912 Å).
_WAVE = jnp.array([500.0, 800.0, 1000.0, 1500.0, 5500.0])
_LYC = np.asarray(_WAVE) < 912.0
# t_birth = 1e7 yr (default). Age 1e4 is fully young (y≈1 to ~1e-10), 1e10 fully
# old (y≈0); the smooth sigmoid never gives *exactly* 0/1, so assertions below
# use relative fractions rather than exact zeros.
_AGES = jnp.array([1.0e4, 1.0e8, 1.0e10])
_YOUNG, _OLD = 0, 2


def _state(fesc: float, *, young: float, old: float, publish_lyc: bool = True) -> ForwardState:
    """Photoionized-nebular output: stellar in the young and/or old age bin.

    The nebular component masks the summed ``sed_intrinsic`` LyC uniformly by
    ``fesc`` and publishes ``lyc_transmission``.
    """
    lnu_age = jnp.zeros((_AGES.shape[0], _WAVE.shape[0]))
    lnu_age = lnu_age.at[_YOUNG, :].set(young).at[_OLD, :].set(old)
    full = jnp.sum(lnu_age, axis=0)
    lyc_mask = _WAVE < 912.0
    sed_intrinsic = jnp.where(lyc_mask, full * fesc, full)
    derived = {"lnu_age": lnu_age, "ssp_ages_yr": _AGES, "sed_nebular": jnp.zeros_like(_WAVE)}
    if publish_lyc:
        derived["lyc_transmission"] = jnp.where(lyc_mask, fesc, jnp.ones_like(_WAVE))
    return ForwardState(wave=_WAVE, sed_intrinsic=sed_intrinsic, derived=derived)


def _apply(state, *, absorb_all=False):
    comp = DustSEDComponent(
        config=DustSEDComponentConfig(
            law_bc="calzetti", law_diff="calzetti", lyc_absorb_all=absorb_all
        )
    )
    # tau_bc=0 -> young and old both see only the diffuse screen (age-independent
    # transmission), so the below-912 result isolates the LyC absorption logic.
    return comp.apply(state, {"dust_tau_bc": 0.0, "dust_tau_diff": 1.0})


def test_no_negative_flux_below_912():
    """#824: predict() must never go negative below 912 Å, in either mode."""
    for absorb_all in (False, True):
        for fesc in (0.0, 0.5):
            out = _apply(_state(fesc, young=1e29, old=1e29), absorb_all=absorb_all)
            assert_non_negative(np.asarray(out.sed_intrinsic), name="output")


def test_default_keeps_old_lyc_absorbs_young():
    """Default (young-only): fesc=0 removes young LyC, keeps old LyC below 912."""
    # Young only -> below 912 nearly all absorbed at fesc=0 (>99.9% for y≈1).
    sda_y0 = np.asarray(_apply(_state(0.0, young=1e29, old=0.0)).derived["sed_dust_attenuated"])
    sda_y1 = np.asarray(_apply(_state(1.0, young=1e29, old=0.0)).derived["sed_dust_attenuated"])
    assert np.all(sda_y0[_LYC] < 1e-3 * sda_y1[_LYC])  # young LyC absorbed
    # Old only -> below 912 KEPT, and independent of fesc.
    sda_o0 = np.asarray(_apply(_state(0.0, young=0.0, old=1e29)).derived["sed_dust_attenuated"])
    sda_o1 = np.asarray(_apply(_state(1.0, young=0.0, old=1e29)).derived["sed_dust_attenuated"])
    assert np.all(sda_o0[_LYC] > 0.0)
    # fesc-independent to within the young indicator's tail. The indicator is a
    # LOGISTIC (unified with the exact dust screen in #1122; it used to be a 2.3x
    # sharper base-10 sigmoid here), and a logistic never reaches exactly zero: a
    # 1e10 yr population keeps y = 4.5e-5 of birth-cloud membership. The dust screen
    # has always assumed exactly that, so the LyC must too. The old rtol=1e-6 was
    # calibrated to the sharper curve, i.e. to the bug.
    np.testing.assert_allclose(sda_o0[_LYC], sda_o1[_LYC], rtol=1e-3)


def test_absorb_all_zeros_all_lyc():
    """lyc_absorb_all=True: fesc=0 removes BOTH young and old LyC below 912."""
    out = _apply(_state(0.0, young=1e29, old=1e29), absorb_all=True)
    sda = np.asarray(out.derived["sed_dust_attenuated"])
    assert np.allclose(sda[_LYC], 0.0)
    assert np.all(np.asarray(out.sed_intrinsic)[~_LYC] > 0.0)  # >912 untouched


def test_young_lyc_scales_with_fesc():
    """Young LyC survival is linear in fesc (default).

    ``sda(fesc) = young·T·(1 - y·(1 - fesc))``, so
    ``(sda(fesc) - sda(0)) / (sda(1) - sda(0)) = fesc`` exactly — a ratio that
    cancels the small ``(1 - y)`` sigmoid residual.
    """
    young_only = lambda f: np.asarray(  # noqa: E731
        _apply(_state(f, young=1e29, old=0.0)).derived["sed_dust_attenuated"]
    )
    s0, s1 = young_only(0.0)[_LYC], young_only(1.0)[_LYC]
    for fesc in (0.3, 0.7):
        frac = (young_only(fesc)[_LYC] - s0) / (s1 - s0)
        np.testing.assert_allclose(frac, fesc, rtol=1e-5)


def test_absent_lyc_transmission_passes_through():
    """No published lyc_transmission (BakedIn) -> LyC unmasked, no error."""
    out = _apply(_state(1.0, young=1e29, old=1e29, publish_lyc=False))
    total = np.asarray(out.sed_intrinsic)
    assert np.all(total >= 0.0)
    assert np.all(total[_LYC] > 0.0)  # LyC present (gas absorption not applied)
