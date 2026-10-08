# SPDX-License-Identifier: BSD-3-Clause
"""Meiksin (2006) IGM: tengri against the paper and against CIGALE's ``igm_transmission`` (#2785).

Both codes implement Meiksin (2006, MNRAS 365, 807). Their continuum terms are the
same closed forms (Eq. 5 for the optically thin forest, Eq. 7 for the Lyman-limit
systems) and agree to ~1e-5 in T. They differ in the Lyman-series lines, where
CIGALE 2025.1 (``pcigale/sed_modules/redshifting.py``) departs from the paper in
two ways, both for the absorber redshift ``z_n = lambda_obs / lambda_n - 1``:

1. ``tau_alpha`` of the n >= 3 lines is evaluated at the *source* redshift (lines 60-81),
   not at ``z_n``. Meiksin's Table 1 gives ``tau_n / tau_alpha`` as a function of ``z_n``
   and Eqs. 2-3 give ``tau_alpha`` at the redshift of the absorbing gas.
2. The Lyman-alpha amplitude switches between Eq. 2 (z < 4) and Eq. 3 (z > 4) on the
   *source* redshift (``if redshift <= 4``), so a source at z > 4 applies Eq. 3 to
   absorbers at z_n < 4, where the paper's Eq. 2 holds.

The paper's own Table 2 values at 1730 A (observed) arbitrate: tengri follows them.
This module pins (a) tengri against an independent numpy transcription of the
paper's equations, (b) CIGALE against the same transcription with the two readings
above, (c) the tengri minus CIGALE difference as the difference of those two
readings, and (d) the issue's numbers.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import erfc

from tengri.components.igm.meiksin06 import igm_transmission_meiksin06

pytestmark = [pytest.mark.crossval, pytest.mark.regression_paper]

_LYMAN_LIMIT_AA = 912.0
_FACT = {3: 0.348, 4: 0.179, 5: 0.109, 6: 0.0722, 7: 0.0508, 8: 0.0373, 9: 0.0283}
_REDSHIFTS = (3.0, 5.0, 7.0)
_WAVE_AA = np.linspace(900.0, 9000.0, 600)  # the grid of the issue
# Gamma(2 - beta, 1) = Gamma(1/2, 1) = sqrt(pi) erfc(1)
_GAMMA_HALF_1 = math.sqrt(math.pi) * float(erfc(1.0))


def _lam_n(n: int) -> float:
    return _LYMAN_LIMIT_AA / (1.0 - 1.0 / n**2)


def _tau_alpha(z):
    """Meiksin (2006) Eq. 2 (z < 4) and Eq. 3 (z > 4)."""
    z = np.asarray(z, dtype=float)
    return np.where(z < 4.0, 0.00211 * (1.0 + z) ** 3.7, 0.00058 * (1.0 + z) ** 4.5)


def _paper_tau_lines(lam_obs, z, *, amplitude_at, branch_at):
    """Sum of the Lyman-series optical depths, Table 1 and Eq. 4 (n = 2..30).

    ``amplitude_at`` / ``branch_at`` are ``"absorber"`` (the paper) or ``"source"``
    (CIGALE 2025.1): the redshift at which ``tau_alpha`` is evaluated for n >= 3 and the
    redshift that selects Eq. 2 or Eq. 3 for n = 2. Eq. 4 scales the n >= 10 lines from
    ``tau_9`` at ``z_9``, as both codes do.
    """
    z_src = np.full_like(lam_obs, z)
    total = np.zeros_like(lam_obs)
    tau = {}
    z_n = {n: lam_obs / _lam_n(n) - 1.0 for n in range(2, 31)}
    z2 = z_n[2]
    tau[2] = (
        _tau_alpha(z2)
        if branch_at == "absorber"
        else (0.00211 * (1.0 + z2) ** 3.7 if z <= 4.0 else 0.00058 * (1.0 + z2) ** 4.5)
    )
    for n in range(3, 10):
        zn = z_n[n]
        amp = _tau_alpha(zn if amplitude_at == "absorber" else z_src)
        late = 1.0 / 6.0 if n <= 5 else 1.0 / 3.0
        expo = np.where(zn < 3.0, 1.0 / 3.0, late)
        tau[n] = amp * _FACT[n] * (0.25 * (1.0 + zn)) ** expo
    for n in range(10, 31):
        tau[n] = tau[9] * 720.0 / (n * (n * n - 1.0))
    for n, t_n in tau.items():
        total = total + np.where((z_n[n] >= 0.0) & (z_n[n] < z), t_n, 0.0)
    return total


def _paper_tau_continuum(lam_obs, z):
    """Eq. 5 forest plus Eq. 7 LLS (beta = gamma = 1.5, N0 = 0.25, ten terms), damped as
    ``(1 + z_L)^2.75`` below the Lyman limit (O'Meara et al. 2013, as in CIGALE)."""
    n0, beta, gam = 0.25, 1.5, 1.5

    def at_or_above_limit(ratio):  # ratio = lambda / lambda_L, 0 <= z_L < z
        zl = ratio - 1.0
        t_igm = 0.805 * (1.0 + zl) ** 3 * (1.0 / (1.0 + zl) - 1.0 / (1.0 + z))
        s2 = sum(
            (beta - 1.0) / (k + 1.0 - beta) * (-1.0) ** k / math.factorial(k) for k in range(10)
        )
        pref = n0 / (4.0 + gam - 3.0 * beta)
        brack = (1.0 + z) ** (-3.0 * (beta - 1.0) + gam + 1.0) * ratio ** (
            3.0 * (beta - 1.0)
        ) - ratio ** (gam + 1.0)
        t_lls = pref * (_GAMMA_HALF_1 - math.exp(-1.0) - s2) * brack
        for k in range(1, 11):
            coef = (beta - 1.0) / ((3.0 * k - gam - 1.0) * (k + 1.0 - beta))
            t_lls = t_lls - n0 * coef * (-1.0) ** k / math.factorial(k) * (
                (1.0 + z) ** (gam + 1.0 - 3.0 * k) * ratio ** (3.0 * k) - ratio ** (gam + 1.0)
            )
        return t_igm + t_lls

    ratio = lam_obs / _LYMAN_LIMIT_AA
    above = (ratio >= 1.0) & (ratio - 1.0 < z)
    t_above = np.where(above, at_or_above_limit(np.where(above, ratio, 1.0)), 0.0)
    edge = at_or_above_limit(1.0)
    return np.where(ratio < 1.0, edge * ratio**2.75, t_above)


def _paper_transmission(lam_obs, z, *, amplitude_at="absorber", branch_at="absorber"):
    lines = _paper_tau_lines(lam_obs, z, amplitude_at=amplitude_at, branch_at=branch_at)
    return np.exp(-(lines + _paper_tau_continuum(lam_obs, z)))


def _tengri(lam_obs, z):
    return np.asarray(igm_transmission_meiksin06(jnp.asarray(lam_obs), z))


def _pcigale(lam_obs, z):
    pcigale_redshifting = pytest.importorskip("pcigale.sed_modules.redshifting")
    return np.asarray(pcigale_redshifting.igm_transmission(lam_obs / 10.0, z))


@pytest.mark.parametrize("z", _REDSHIFTS)
def test_tengri_matches_paper_equations(z):
    """tengri equals the transcription of Eqs. 2-5, 7 and Table 1 to TOL_TENGRI in T.

    The residual is the 0.2788 vs Gamma(1/2, 1) = 0.278806 constant and the ten-term
    truncation of Eq. 7's series (term n = 9 is 1.6e-7 of the n = 0 term).
    """
    diff = np.abs(_tengri(_WAVE_AA, z) - _paper_transmission(_WAVE_AA, z))
    assert diff.max() < 3e-6, f"z={z}: max|dT|={diff.max():.2e}"


@pytest.mark.parametrize("z", _REDSHIFTS)
def test_pcigale_matches_paper_with_source_redshift_lines(z):
    """CIGALE equals the same equations with its two source-redshift readings.

    Residual (<= 3e-5 in T): CIGALE's ``np.interp`` at z_L = 0 for the damping
    normalization on a 600-point grid, against the closed-form edge value.
    """
    pcigale = _pcigale(_WAVE_AA, z)
    reading = _paper_transmission(_WAVE_AA, z, amplitude_at="source", branch_at="source")
    assert np.abs(pcigale - reading).max() < 3e-5, f"z={z}"


@pytest.mark.parametrize("z", _REDSHIFTS)
def test_difference_is_the_two_line_readings(z):
    """tengri minus CIGALE equals paper minus source-redshift reading, to 3e-5; size pinned."""
    t, p = _tengri(_WAVE_AA, z), _pcigale(_WAVE_AA, z)
    paper = _paper_transmission(_WAVE_AA, z)
    source = _paper_transmission(_WAVE_AA, z, amplitude_at="source", branch_at="source")
    assert np.abs((t - p) - (paper - source)).max() < 3e-5
    expected_max = {3.0: 0.0445, 5.0: 0.0469, 7.0: 0.0199}[z]
    assert np.abs(t - p).max() == pytest.approx(expected_max, abs=2e-4)


@pytest.mark.parametrize(("z", "expected"), [(3.0, 0.0), (5.0, 8.37e-3), (7.0, 8.6e-6)])
def test_lyalpha_branch_reading_effect(z, expected):
    """Selecting Eq. 2 / Eq. 3 on the source redshift changes T by 8.4e-3 at z = 5.

    At z = 3 every z_n < 3 and both readings use Eq. 2; at z = 7 the n >= 2 optical depths
    already leave T ~ 0 where the Lyman-alpha absorbers have z_n < 4.
    """
    a = _paper_transmission(_WAVE_AA, z, branch_at="source")
    b = _paper_transmission(_WAVE_AA, z, branch_at="absorber")
    assert np.abs(a - b).max() == pytest.approx(expected, abs=2e-5)


@pytest.mark.parametrize("z", _REDSHIFTS)
def test_codes_agree_redward_of_lyman_beta(z):
    """With only Lyman-alpha active (rest >= 1026 A) the two codes agree to 1e-10."""
    redward = 1026.0 * (1.0 + z) <= _WAVE_AA
    lam = _WAVE_AA[redward]
    assert lam.size > 50
    assert np.abs(_tengri(lam, z) - _pcigale(lam, z)).max() < 1e-10


def test_issue_point_1730_angstrom_z5():
    """T(1730 A, z = 5): Meiksin Table 2 0.003277; tengri 0.003283; CIGALE 0.000909 (3.6x low)."""
    lam = np.array([1730.0])
    t, p = float(_tengri(lam, 5.0)[0]), float(_pcigale(lam, 5.0)[0])
    assert t == pytest.approx(0.003277, rel=3e-3)
    assert p == pytest.approx(0.000909, rel=1e-3)
    assert t / p == pytest.approx(3.61, rel=1e-2)
