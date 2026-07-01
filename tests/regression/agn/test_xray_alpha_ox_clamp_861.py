# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #861 — clamp the alpha_ox-L_2500 relation to its
calibration window so the AGN X-ray corona never extrapolates to a *positive*
alpha_ox at sub-AGN luminosities (which made L_X exceed L_bol).

Just et al. 2007 (ApJ 665, 1004), Eq. 3::

    alpha_ox = -0.137 * log10(L_2500) + 2.638,   valid 28 <~ log10(L_2500) <~ 33.

Below log10(L_2500) = 2.638 / 0.137 = 19.3 the *un-clamped* fit returns
alpha_ox > 0, i.e. L_2keV = L_2500 * 10^(alpha_ox / 0.3838) > L_2500 — the 2 keV
corona brighter than the 2500 A disc, which drives the total X-ray luminosity
above L_bol. Observed in the CIGALE reproduction (§10) at a unit-normalized AGN
(log10(L_2500) ~ 17.5): alpha_ox = +0.24 and L_X ~ 1200 * L_bol.

The fix clamps log10(L_2500) to ``[LOG_L2500_CALIB_MIN, LOG_L2500_CALIB_MAX]``
before applying the empirical relation.
"""

from __future__ import annotations

import pytest

from tengri.components.xray.xray import (
    LOG_L2500_CALIB_MAX,
    LOG_L2500_CALIB_MIN,
    alpha_ox_from_l2500,
)

pytestmark = [pytest.mark.regression_bug]

ALL_RELATIONS = ("just2007", "lusso_risaliti_2016", "lusso_risaliti_2017")


@pytest.mark.parametrize("relation", ALL_RELATIONS)
@pytest.mark.parametrize("log_l2500", [10.0, 15.0, 17.5, 19.3, 25.0])
def test_alpha_ox_never_positive_below_calibration(relation, log_l2500):
    """alpha_ox stays negative for any L_2500, including far below the fit range.

    Regression for #861: at log10(L_2500) ~ 17.5 the un-clamped just2007 fit
    returned alpha_ox = +0.24, making the corona brighter than the disc UV.
    """
    aox = float(alpha_ox_from_l2500(10.0**log_l2500, relation=relation))
    assert aox < 0.0, f"{relation}: alpha_ox={aox:+.3f} at log10(L_2500)={log_l2500}"


def test_l_2kev_below_l_2500_at_low_luminosity():
    """The corona is never brighter than the disc UV: L_2keV < L_2500.

    L_2keV / L_2500 = 10^(alpha_ox / 0.3838). The un-clamped fit gave 4.2 at
    log10(L_2500) = 17.5; the clamp keeps alpha_ox <= -1.198 -> ratio < 1e-3.
    """
    for log_l2500 in (12.0, 17.5, 25.0):
        aox = float(alpha_ox_from_l2500(10.0**log_l2500))
        ratio = 10.0 ** (aox / 0.3838)
        assert ratio < 1.0, f"L_2keV/L_2500={ratio:.2e} at log10(L_2500)={log_l2500}"


def test_alpha_ox_clamped_to_floor_below_range():
    """Below the calibration floor alpha_ox equals its value at the floor."""
    aox_floor = float(alpha_ox_from_l2500(10.0**LOG_L2500_CALIB_MIN))
    aox_below = float(alpha_ox_from_l2500(10.0**17.5))
    assert aox_below == pytest.approx(aox_floor, rel=1e-6)
    # Just+2007 at the floor: -0.137 * 28 + 2.638 = -1.198
    assert aox_floor == pytest.approx(-1.198, abs=1e-3)


def test_alpha_ox_unchanged_in_valid_range():
    """No regression inside [28, 33]: the raw Just+2007 fit is returned."""
    # log10(L_2500) = 30 -> -0.137 * 30 + 2.638 = -1.472
    assert float(alpha_ox_from_l2500(1.0e30)) == pytest.approx(-1.472, abs=1e-3)
    # Clamp is a no-op at the upper endpoint.
    expected_max = -0.137 * LOG_L2500_CALIB_MAX + 2.638
    assert float(alpha_ox_from_l2500(10.0**LOG_L2500_CALIB_MAX)) == pytest.approx(
        expected_max, abs=1e-3
    )
