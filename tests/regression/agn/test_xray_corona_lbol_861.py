# SPDX-License-Identifier: BSD-3-Clause
"""Regression: X-ray corona must not exceed the disc that powers it (#861).

The ``yang20`` corona sets its 2 keV luminosity from the disc 2500 A luminosity
via the empirical ``alpha_ox``-L_2500 anti-correlation (Just+2007). That relation
rises without bound as L_2500 falls and turns **positive** below
``log10(L_2500) ~ 19`` -- i.e. ``L_2keV > L_2500``, an X-ray corona brighter than
its own disc, which drove the total X-ray past ``L_bol`` at low AGN luminosity
(``L_X / L_bol ~ 1230`` at the reproduction §10 setup).

The fix (:data:`tengri.components.xray.xray._ALPHA_OX_CALIB_RANGE`) clamps
``log10(L_2500)`` to each relation's fitted range before applying the slope, so
``alpha_ox`` is held at the faintest-calibrated (physical) value below the range
and never goes positive. These tests guard the clamp (root cause) and the
resulting corona <= disc bound, and confirm realistic AGN are unchanged.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.xray.xray import (
    _ALPHA_OX_CALIB_RANGE,
    alpha_ox_from_l2500,
    xray_total,
)

pytestmark = [pytest.mark.regression_bug]


@pytest.mark.parametrize("relation", ["just2007", "lusso_risaliti_2016", "lusso_risaliti_2017"])
def test_alpha_ox_never_positive_below_calibration_range(relation):
    """alpha_ox stays negative (X-ray fainter than the disc UV) at any L_2500.

    A positive alpha_ox means ``L_2keV > L_2500`` -- the #861 pathology. Below
    the fitted range the clamp holds the boundary value, so alpha_ox never
    crosses zero however faint the disc.
    """
    lo, _hi = _ALPHA_OX_CALIB_RANGE[relation]
    for log_l2500 in [10.0, 15.0, 19.0, 25.0, lo]:
        aox = float(alpha_ox_from_l2500(10.0**log_l2500, relation=relation))
        assert aox < 0.0, f"{relation}: alpha_ox={aox:.3f} >= 0 at log L_2500={log_l2500}"


@pytest.mark.parametrize("relation", ["just2007", "lusso_risaliti_2016", "lusso_risaliti_2017"])
def test_alpha_ox_clamped_below_floor_unchanged_above(relation):
    """Below the fitted floor alpha_ox is held; inside the range it is untouched."""
    lo, hi = _ALPHA_OX_CALIB_RANGE[relation]
    floor_val = float(alpha_ox_from_l2500(10.0**lo, relation=relation))
    # far below the floor -> identical to the floor value (clamped)
    assert float(alpha_ox_from_l2500(10.0 ** (lo - 8.0), relation=relation)) == pytest.approx(
        floor_val, abs=1e-9
    )
    # inside the range -> steeper (more negative), i.e. the relation is live
    mid = 0.5 * (lo + hi)
    assert float(alpha_ox_from_l2500(10.0**mid, relation=relation)) < floor_val


def test_corona_2kev_below_disc_2500_at_low_luminosity():
    """End-to-end through ``xray_total``: L_2keV(corona) < L_2500 (the disc).

    Uses the pathological low disc luminosity from #861 with a negligible galaxy
    (tiny SFR / M_star) so the binary + hot-gas channels vanish and the 2 keV
    flux is the corona. ``log_nh`` is set low so we compare the *intrinsic*
    corona, not the absorbed one.
    """
    l2500 = 3.17e17  # erg/s/Hz -- the §10 reproduction value (log ~ 17.5)
    wave_2kev = jnp.array([12.398 / 2.0])  # 2 keV in Angstrom
    sed = xray_total(
        wave_2kev,
        sfr=1e-30,
        stellar_mass=1.0,
        l_2500_30deg=l2500,
        log_nh=16.0,  # effectively unabsorbed at 2 keV
    )
    l_2kev = float(np.asarray(sed)[0])
    assert 0.0 < l_2kev < l2500, (
        f"corona L_2keV={l_2kev:.3e} must be below the disc L_2500={l2500:.3e} "
        f"(alpha_ox < 0); got ratio {l_2kev / l2500:.3e} (#861)"
    )


def test_realistic_agn_corona_unchanged():
    """A quasar-luminosity disc (log L_2500 ~ 30, inside the range) is untouched."""
    l2500 = 1.0e30
    # inside the fitted range -> equals the raw (unclamped) Just+2007 value
    raw = -0.137 * np.log10(l2500) + 2.638
    assert float(alpha_ox_from_l2500(l2500)) == pytest.approx(raw, abs=1e-9)
