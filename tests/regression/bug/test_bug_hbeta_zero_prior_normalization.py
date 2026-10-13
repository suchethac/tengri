# SPDX-License-Identifier: BSD-3-Clause
"""A CLOUDY grid with a vanishing H-beta must raise, not normalize by 1.0.

Bug: ``observation/eline_priors.py`` replaced a sub-``1e-30`` H-beta luminosity
with ``1.0`` ("safety fallback"). Every prior mean was then divided by 1.0
instead of by H-beta, so the line ratios came out in units of erg/s rather than
of H-beta with no error. A grid whose H-beta is that faint is corrupt; the
function must refuse it and name the value.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tengri.observation.eline_priors import cloudy_grid_line_priors

pytestmark = pytest.mark.regression_bug


def _grid_with_hbeta_log_lum(hbeta_log_lum: float) -> SimpleNamespace:
    """A 2x2x2 grid of three lines; line 0 is H-beta at the given log10 luminosity."""
    lum = np.full((2, 2, 2, 3), 0.5)
    lum[..., 0] = hbeta_log_lum
    return SimpleNamespace(
        line_luminosity=lum,
        line_log_met=np.array([-1.0, 0.0]),
        line_log_age=np.array([6.0, 8.0]),
        line_log_U=np.array([-3.5, -2.0]),
        line_wavelengths=np.array([4862.68, 5008.24, 6564.61]),
    )


def test_faint_hbeta_raises_and_names_the_value():
    """H-beta at 1e-40 erg/s is refused; the message carries the value."""
    grid = _grid_with_hbeta_log_lum(-40.0)
    with pytest.raises(ValueError, match=r"Hbeta luminosity 1\.000e-40"):
        cloudy_grid_line_priors(grid, log_z=-0.5, neb_logU=-3.0)


def test_healthy_hbeta_still_normalizes_to_unity():
    """A grid with a real H-beta keeps H-beta as the unit reference."""
    grid = _grid_with_hbeta_log_lum(38.0)
    means, _ = cloudy_grid_line_priors(grid, log_z=-0.5, neb_logU=-3.0)
    assert float(means[0]) == pytest.approx(1.0)
