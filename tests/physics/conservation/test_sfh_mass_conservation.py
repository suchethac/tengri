# SPDX-License-Identifier: BSD-3-Clause
"""SFH mass conservation tests — trapezoid integral vs target log_total_mass.

After the 2026-05-25 normalization refactor, every parametric SFH callable
obeys the contract::

    trapezoid(sfr, t_lookback) == 10**log_total_mass     (within trapezoid noise)

This module spot-checks that contract for a few callables. The broader
parametric coverage lives in :mod:`tests.physics.conservation.test_sfh_normalization_contract`.

The pre-refactor formulas like ``peak_sfr * tau * (1 - exp(-T/tau))`` no
longer hold — those were ``log_peak_sfr`` semantics where ``peak_sfr`` was
the amplitude knob, not the integrated mass. After the refactor the integral
*is* the parameter.
"""

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.conservation

from scipy.integrate import trapezoid

from tengri.components.stellar.sfh.mean_sfh import (
    constant,
    delayed_exponential,
    double_powerlaw,
    dpl,
    exponential,
)

# Age of the universe today [yr], from the default cosmology — never a
# literal. SFH formation anchor (age_gyr) for dpl/lnorm shape tests.
from tengri.cosmology import age_at_z0 as _age_at_z0

_AGE_UNIV_YR = float(_age_at_z0()) * 1e9

# Common time grid: 1 Myr to 13 Gyr in 5000 steps
T_MIN = 1e6
T_MAX = 13e9
N_GRID = 5000
T_GRID = np.linspace(T_MIN, T_MAX, N_GRID)


def _integrate(sfr_fn, t=T_GRID, **kwargs):
    """Numerically integrate SFR over time grid."""
    sfr = np.array(sfr_fn(jnp.array(t), **kwargs))
    return float(trapezoid(sfr, t))


# ── exponential ───────────────────────────────────────────────


class TestExponentialSFHMassConservation:
    """exponential(t, log_total_mass, tau) integrates to 10**log_total_mass."""

    @pytest.mark.parametrize(
        "log_total_mass, tau",
        [(10.0, 1e9), (9.5, 5e9), (11.0, 2e9), (8.5, 3e9)],
    )
    def test_integral_matches_log_total_mass(self, log_total_mass, tau):
        numerical = _integrate(exponential, log_total_mass=log_total_mass, tau=tau)
        expected = 10.0**log_total_mass
        assert abs(numerical - expected) / expected < 0.01

    def test_start_offset_preserves_total_mass(self):
        """start != 0 just truncates the grid; renormalization still hits target mass."""
        log_total_mass, tau, start = 10.0, 2e9, 1e9
        t = np.linspace(start, T_MAX, N_GRID)
        sfr = np.array(
            exponential(jnp.array(t), log_total_mass=log_total_mass, tau=tau, start=start)
        )
        numerical = float(trapezoid(sfr, t))
        assert abs(numerical - 10.0**log_total_mass) / 10.0**log_total_mass < 0.01


# ── delayed_exponential ───────────────────────────────────────


class TestDelayedExponentialSFHMassConservation:
    """delayed_exponential(t, log_total_mass, tau) integrates to 10**log_total_mass."""

    @pytest.mark.parametrize(
        "log_total_mass, tau",
        [(10.0, 1e9), (9.5, 5e9), (11.0, 3e9)],
    )
    def test_integral_matches_log_total_mass(self, log_total_mass, tau):
        numerical = _integrate(delayed_exponential, log_total_mass=log_total_mass, tau=tau)
        expected = 10.0**log_total_mass
        assert abs(numerical - expected) / expected < 0.01


# ── constant ──────────────────────────────────────────────────


class TestConstantSFHMassConservation:
    """constant(t, log_total_mass, start, end) integrates to 10**log_total_mass within window."""

    def test_full_range(self):
        log_total_mass = 10.0
        start, end = T_MIN, T_MAX
        numerical = _integrate(constant, log_total_mass=log_total_mass, start=start, end=end)
        expected = 10.0**log_total_mass
        assert abs(numerical - expected) / expected < 0.005

    def test_partial_range(self):
        log_total_mass = 9.5
        start, end = 2e9, 8e9
        t = np.linspace(T_MIN, T_MAX, N_GRID)
        sfr = np.array(constant(jnp.array(t), log_total_mass=log_total_mass, start=start, end=end))
        numerical = float(trapezoid(sfr, t))
        expected = 10.0**log_total_mass
        assert abs(numerical - expected) / expected < 0.005

    def test_flat_sfr_within_window(self):
        """SFR should be constant (zero variance) within support."""
        log_total_mass = 10.0
        start, end = 1e9, 10e9
        t_in = np.linspace(start + 1e7, end - 1e7, 100)
        sfr_in = np.array(
            constant(jnp.array(t_in), log_total_mass=log_total_mass, start=start, end=end)
        )
        # After renormalization the flat value is approximately
        # 10**log_total_mass / mass_norm, where mass_norm = trapezoid(mask, t)
        # over the FULL grid (including soft-edge bins). Small boundary effects
        # from the trapezoid quadrature shift this from the analytical
        # ``(end - start)`` by ~0.2% on coarse grids — fine for science.
        assert float(np.std(sfr_in)) == pytest.approx(0.0, abs=1e-3)
        expected_flat = 10.0**log_total_mass / (end - start)
        np.testing.assert_allclose(sfr_in, expected_flat, rtol=0.01)

    def test_zero_outside_support(self):
        """SFR must be zero outside [start, end]."""
        log_total_mass = 10.0
        start, end = 3e9, 7e9
        t_before = np.linspace(T_MIN, start - 1e8, 50)
        t_after = np.linspace(end + 1e8, T_MAX, 50)
        sfr_before = np.array(
            constant(jnp.array(t_before), log_total_mass=log_total_mass, start=start, end=end)
        )
        sfr_after = np.array(
            constant(jnp.array(t_after), log_total_mass=log_total_mass, start=start, end=end)
        )
        np.testing.assert_array_equal(sfr_before, 0.0)
        np.testing.assert_array_equal(sfr_after, 0.0)


# ── double_powerlaw ───────────────────────────────────────────────


class TestDoublePowerlawMassConservation:
    """Bare double_powerlaw (unnormalized) — verify numerical properties only.

    ``double_powerlaw`` is the bare shape function and does NOT use
    ``log_total_mass`` (the registry-facing ``dpl`` wrapper does).
    """

    def test_finite_positive(self):
        alpha, beta, tau, norm = 1.5, 2.0, 3e9, 5.0
        mass = _integrate(double_powerlaw, alpha=alpha, beta=beta, tau=tau, norm=norm)
        assert np.isfinite(mass)
        assert mass > 0.0

    def test_dpl_renormalizes_to_target(self):
        """The registry-facing ``dpl`` rescales the bare shape to 10**log_total_mass."""
        log_total_mass = 10.0
        mass = _integrate(
            dpl, alpha=1.5, beta=2.0, tau=3e9, age=_AGE_UNIV_YR, log_total_mass=log_total_mass
        )
        assert abs(mass - 10.0**log_total_mass) / 10.0**log_total_mass < 0.01
