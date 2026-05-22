# SPDX-License-Identifier: BSD-3-Clause
"""Tests for ``tengri.analysis.feature_strengths.clip_feature``.

The "clipped feature strength" Fclip is defined in Draine, Li, Hensley
et al. 2021 (ApJ 917 3, arXiv:2011.07046), Section 9.1: integrate
nu*P_nu between two clip wavelengths (lambda1, lambda2) above a linear
baseline drawn between the two endpoints.

These tests exercise the integrator on synthetic spectra where the
answer is analytic, before we apply it to the loaded PAHspec templates.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.analysis.feature_strengths import clip_feature, total_ir_power

pytestmark = pytest.mark.bounds


@pytest.mark.unit
def test_constant_continuum_has_zero_clip():
    """A flat nu*P_nu continuum has Fclip = 0 (baseline subtracts it)."""
    wave_um = np.geomspace(3.0, 4.0, 200)
    nu_pnu = np.full_like(wave_um, 1.0e-24)

    fclip = clip_feature(wave_um, nu_pnu, lam1_um=3.09, lam2_um=3.52)

    assert abs(fclip) < 1e-30


@pytest.mark.unit
def test_linear_continuum_has_zero_clip():
    """A linear-in-ln(nu) continuum is exactly the baseline; Fclip = 0."""
    # Baseline is linear in nu*P_nu vs ln(nu); build that explicitly.
    wave_um = np.geomspace(3.0, 4.0, 200)
    ln_nu = np.log(1.0 / wave_um)  # arbitrary up to a constant
    nu_pnu = 1.0e-24 + 2.0e-24 * (ln_nu - ln_nu[0])

    fclip = clip_feature(wave_um, nu_pnu, lam1_um=3.09, lam2_um=3.52)

    # Tolerance: discrete trapezoid vs analytic linear integral.
    assert abs(fclip) < 1e-29


@pytest.mark.unit
def test_box_feature_above_continuum():
    """A box-shaped excess of known area above zero baseline integrates
    to its analytic area in d ln nu space."""
    wave_um = np.geomspace(3.0, 4.0, 1001)
    # Zero continuum + a constant excess of A=1e-23 erg/s/H between
    # lambda in [3.2, 3.4] microns.
    nu_pnu = np.zeros_like(wave_um)
    box = (wave_um >= 3.2) & (wave_um <= 3.4)
    nu_pnu[box] = 1.0e-23

    # Clip endpoints sit *outside* the box so endpoints have value 0,
    # i.e. the linear baseline is identically zero.
    fclip = clip_feature(wave_um, nu_pnu, lam1_um=3.0, lam2_um=4.0)

    # Analytic: int nu*P_nu d ln nu = A * ln(lam_b / lam_a) for box
    # of value A between lam_a and lam_b (since d ln nu = -d ln lam).
    expected = 1.0e-23 * np.log(3.4 / 3.2)
    np.testing.assert_allclose(fclip, expected, rtol=2e-3)


@pytest.mark.unit
def test_total_ir_power_constant_spectrum():
    """Total IR = integral of nu*P_nu d ln nu = A * ln(lam_max/lam_min)
    for a flat nu*P_nu spectrum of value A."""
    wave_um = np.geomspace(1.0, 1000.0, 2000)
    A = 3.5e-24
    nu_pnu = np.full_like(wave_um, A)

    ftir = total_ir_power(wave_um, nu_pnu)
    expected = A * np.log(1000.0 / 1.0)

    np.testing.assert_allclose(ftir, expected, rtol=1e-3)


@pytest.mark.unit
def test_clip_feature_input_validation():
    wave_um = np.array([3.0, 3.5, 4.0])
    nu_pnu = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        clip_feature(wave_um, nu_pnu, lam1_um=4.0, lam2_um=3.0)
    with pytest.raises(ValueError):
        clip_feature(wave_um, nu_pnu, lam1_um=2.0, lam2_um=3.5)  # below grid
    with pytest.raises(ValueError):
        clip_feature(wave_um[:-1], nu_pnu, lam1_um=3.1, lam2_um=3.9)  # mismatch
