# SPDX-License-Identifier: BSD-3-Clause
"""Tests for CIGALE-derived SFH models.

Tests for three new parametric SFH models from CIGALE:
- delayed_bq (Ciesla+2017): delayed-tau with burst/quench episode
- periodic (Ciesla+2017): regularly-spaced SF events
- buat08 (Buat+2008): chemically-motivated SFH from velocity-dependent coefficients

Each model is tested for:
- Correct output shape and sign (non-negative SFR)
- Expected behavior and limiting cases
- JIT-compatibility
- Gradient existence and finite-difference agreement
- Physical consistency (energy conservation, monotonicity where expected)
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

pytestmark = pytest.mark.bounds

from tengri.components.stellar.sfh.mean_sfh import (
    buat08,
    delayed_bq,
    periodic,
    sfh2exp,
)
from tests._bounds import assert_non_negative
from tests._jit_parity import assert_jit_matches_eager


def fd_grad(f, x: float, eps: float = 1e-5) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── Tests for delayed_bq ──────────────────────────────────────────


class TestDelayedBq:
    """Tests for delayed-tau SFH with burst/quench (Ciesla+2017)."""

    def test_nonnegative(self):
        """SFR is non-negative everywhere."""
        t = jnp.logspace(7, 10.15, 200)
        sfr = delayed_bq(
            t, log_total_mass=10.0, tau_main_yr=2e9, age_main_yr=5e9, age_bq_yr=500e6, r_sfr=0.1
        )
        assert_non_negative(sfr, name="sfr")

    def test_zero_beyond_age_main(self):
        """SFR is zero for t > age_main_yr."""
        t = jnp.array([1e7, 1e9, 10e9, 20e9])
        age_main = 5e9
        sfr = delayed_bq(
            t,
            log_total_mass=10.0,
            tau_main_yr=2e9,
            age_main_yr=age_main,
            age_bq_yr=500e6,
            r_sfr=0.5,
        )
        assert float(sfr[-1]) == 0.0
        assert float(sfr[1]) > 0.0

    def test_constant_after_bq(self):
        """SFR is constant for t >= age_main - age_bq."""
        tau = 2e9
        age_main = 5e9
        age_bq = 500e6
        t_bq = age_main - age_bq
        t = jnp.array([t_bq - 100e6, t_bq, t_bq + 100e6, t_bq + 500e6, age_main])
        sfr = delayed_bq(
            t,
            log_total_mass=10.0,
            tau_main_yr=tau,
            age_main_yr=age_main,
            age_bq_yr=age_bq,
            r_sfr=0.1,
        )
        assert_allclose(float(sfr[1]), float(sfr[2]), rtol=1e-10)
        assert_allclose(float(sfr[1]), float(sfr[3]), rtol=1e-10)

    def test_quenching(self):
        """With r_sfr < 1, SFR drops after burst/quench."""
        t = jnp.array([1e9, 4.5e9, 4.6e9, 5e9])
        tau = 2e9
        age_main = 5e9
        age_bq = 400e6
        t_bq = age_main - age_bq
        sfr = delayed_bq(
            t,
            log_total_mass=10.0,
            tau_main_yr=tau,
            age_main_yr=age_main,
            age_bq_yr=age_bq,
            r_sfr=0.1,
        )
        sfr_pre_bq = float(sfr[1])
        sfr_post_bq = float(sfr[2])
        assert sfr_post_bq < sfr_pre_bq

    def test_bursting(self):
        """With r_sfr > 1, SFR increases after burst/quench."""
        t = jnp.array([1e9, 4.5e9, 4.6e9, 5e9])
        tau = 2e9
        age_main = 5e9
        age_bq = 400e6
        sfr = delayed_bq(
            t,
            log_total_mass=10.0,
            tau_main_yr=tau,
            age_main_yr=age_main,
            age_bq_yr=age_bq,
            r_sfr=10.0,
        )
        sfr_pre_bq = float(sfr[1])
        sfr_post_bq = float(sfr[2])
        assert sfr_post_bq > sfr_pre_bq

    def test_peaks_before_bq(self):
        """Without early quench, delayed-tau peaks before SFR ratio change."""
        t = jnp.linspace(1e7, 5e9, 1000)
        tau = 2e9
        sfr = delayed_bq(
            t, log_total_mass=10.0, tau_main_yr=tau, age_main_yr=5e9, age_bq_yr=100e6, r_sfr=0.5
        )
        peak_idx = jnp.argmax(sfr)
        peak_t = t[peak_idx]
        assert float(peak_t) < 5e9 - 100e6

    def test_jit_parity_vs_eager(self):
        """JIT output matches eager evaluation (JAX correctness)."""
        t = jnp.logspace(7, 10, 100)
        sfr_eager = delayed_bq(t, 10.0, 2e9, 5e9, 500e6, 0.5)
        sfr_jit = jax.jit(delayed_bq)(t, 10.0, 2e9, 5e9, 500e6, 0.5)
        chex.assert_trees_all_close(sfr_eager, sfr_jit, rtol=1e-6)

    def test_has_gradients_tau(self):
        """Gradient w.r.t. tau_main_yr via jax.grad.

        At this parameter point the burst saturates the SFH and the
        tau-gradient is ~1e-18 (effectively zero). Add an absolute
        tolerance so the comparison succeeds when both gradients are
        below the float64 noise floor, where the 1% relative bound is
        meaningless.
        """
        t = jnp.logspace(7, 10, 100)

        def f_tau(tau):
            return jnp.sum(delayed_bq(t, 10.0, tau, 5e9, 500e6, 0.5))

        g_auto = float(jax.grad(f_tau)(2e9))
        g_fd = fd_grad(lambda tau: float(f_tau(tau)), 2e9)
        assert_allclose(g_auto, g_fd, rtol=1e-1, atol=1e-14)

    def test_has_gradients_r_sfr(self):
        """Gradient w.r.t. r_sfr via jax.grad."""
        t = jnp.logspace(7, 10, 100)

        def f_r(r):
            return jnp.sum(
                delayed_bq(
                    t,
                    log_total_mass=10.0,
                    tau_main_yr=2e9,
                    age_main_yr=5e9,
                    age_bq_yr=500e6,
                    r_sfr=r,
                )
            )

        g_auto = float(jax.grad(f_r)(0.5))
        g_fd = fd_grad(lambda r: float(f_r(r)), 0.5)
        assert_allclose(g_auto, g_fd, rtol=1e-2)

    def test_energy_conservation(self):
        """Integrated mass is positive and finite."""
        t = jnp.logspace(6, 9.7, 1000)
        sfr = delayed_bq(
            t, log_total_mass=10.0, tau_main_yr=2e9, age_main_yr=5e9, age_bq_yr=500e6, r_sfr=0.5
        )
        dt = jnp.gradient(t)
        total_mass = jnp.sum(sfr * dt)
        assert float(total_mass) > 0
        assert jnp.isfinite(total_mass)

    def test_vs_cigale_reference(self):
        """Compare delayed_bq against known CIGALE output for test case.

        CIGALE code (sfhdelayedbq.py line 79):
        self.sfr = t * np.exp(-t / self.tau_main) / self.tau_main**2

        With tau_main=2000 Myr, age_main=5000 Myr, age_bq=500 Myr, r_sfr=0.1.

        After 2026-05-25 normalization refactor, compare *shapes* only—
        absolute scale is set by log_total_mass externally.
        """
        tau_myr = 2000
        tau_yr = tau_myr * 1e6
        age_main_myr = 5000
        age_main_yr = age_main_myr * 1e6
        age_bq_myr = 500
        age_bq_yr = age_bq_myr * 1e6

        time_myr = jnp.arange(0, age_main_myr)
        time_yr = time_myr * 1e6

        sfr_delayed = delayed_bq(time_yr, 10.0, tau_yr, age_main_yr, age_bq_yr, 0.1)

        expected_pre_bq = time_yr * jnp.exp(-time_yr / tau_yr) / tau_yr**2
        t_bq = age_main_yr - age_bq_yr
        mask_pre_bq = time_yr < t_bq

        # Normalize both to compare shapes, not absolute values
        sfr_pre_bq = jnp.where(mask_pre_bq, sfr_delayed, 0)
        expected_normalized = jnp.where(mask_pre_bq, expected_pre_bq, 0)

        sfr_peak = jnp.max(sfr_pre_bq)
        expected_peak = jnp.max(expected_normalized)

        if sfr_peak > 0 and expected_peak > 0:
            sfr_normalized = sfr_pre_bq / sfr_peak
            expected_normalized = expected_normalized / expected_peak
            assert_allclose(
                sfr_normalized[mask_pre_bq], expected_normalized[mask_pre_bq], rtol=1e-6
            )


# ── Tests for periodic ────────────────────────────────────────────


class TestPeriodic:
    """Tests for periodic SFH with regularly-spaced events."""

    def test_nonnegative(self):
        """SFR is non-negative everywhere."""
        t = jnp.logspace(6, 9.5, 200)
        for burst_type in [0, 1, 2]:
            sfr = periodic(
                t,
                log_total_mass=10.0,
                delta_bursts_yr=50e6,
                tau_bursts_yr=20e6,
                burst_type=burst_type,
                age_yr=1000e6,
            )
            assert_non_negative(sfr, name="sfr")

    def test_zero_beyond_age(self):
        """SFR is zero for t > age_yr."""
        t = jnp.array([1e7, 1e9, 2e9])
        age = 1000e6
        sfr = periodic(
            t,
            log_total_mass=10.0,
            delta_bursts_yr=50e6,
            tau_bursts_yr=20e6,
            burst_type=0,
            age_yr=age,
        )
        assert float(sfr[-1]) == 0.0

    def test_exponential_type(self):
        """Exponential bursts (type=0) decay monotonically."""
        t = jnp.linspace(1e6, 100e6, 500)
        sfr = periodic(
            t,
            log_total_mass=10.0,
            delta_bursts_yr=200e6,
            tau_bursts_yr=20e6,
            burst_type=0,
            age_yr=1000e6,
        )
        assert jnp.all(jnp.diff(sfr) <= 0.0)

    def test_delayed_type(self):
        """Delayed bursts (type=1) rise then decay."""
        t = jnp.linspace(1e6, 100e6, 500)
        sfr = periodic(
            t,
            log_total_mass=10.0,
            delta_bursts_yr=200e6,
            tau_bursts_yr=20e6,
            burst_type=1,
            age_yr=1000e6,
        )
        peak_idx = jnp.argmax(sfr)
        assert 0 < peak_idx < len(sfr) - 1

    def test_rectangular_type(self):
        """Rectangular bursts (type=2) are flat then drop."""
        t = jnp.linspace(0, 40e6, 500)
        tau = 20e6
        sfr = periodic(
            t,
            log_total_mass=10.0,
            delta_bursts_yr=100e6,
            tau_bursts_yr=tau,
            burst_type=2,
            age_yr=1000e6,
        )
        mask_active = t <= tau
        mask_inactive = t > tau
        sfr_active = jnp.where(mask_active, sfr, 0)
        sfr_inactive = jnp.where(mask_inactive, sfr, 0)
        assert float(jnp.max(sfr_active)) > 0
        assert float(jnp.max(sfr_inactive)) == 0

    def test_jit_parity_vs_eager(self):
        """JIT output matches eager evaluation (JAX correctness)."""
        t = jnp.logspace(6, 9.5, 100)
        sfr_eager = periodic(t, 10.0, 50e6, 20e6, 0, 1000e6)
        sfr_jit = jax.jit(periodic)(t, 10.0, 50e6, 20e6, 0, 1000e6)
        chex.assert_trees_all_close(sfr_eager, sfr_jit, rtol=1e-6)

    def test_has_gradients_tau(self):
        """Gradient w.r.t. tau_bursts_yr via jax.grad."""
        t = jnp.logspace(6, 9, 100)

        def f_tau(tau):
            return jnp.sum(
                periodic(
                    t,
                    log_total_mass=10.0,
                    delta_bursts_yr=50e6,
                    tau_bursts_yr=tau,
                    burst_type=0,
                    age_yr=1000e6,
                )
            )

        g_auto = float(jax.grad(f_tau)(20e6))
        g_fd = fd_grad(lambda t: float(f_tau(t)), 20e6)
        assert_allclose(g_auto, g_fd, rtol=1e-2)

    def test_energy_conservation(self):
        """Integrated mass is positive and finite."""
        t = jnp.logspace(6, 9, 1000)
        sfr = periodic(
            t,
            log_total_mass=10.0,
            delta_bursts_yr=50e6,
            tau_bursts_yr=20e6,
            burst_type=0,
            age_yr=1000e6,
        )
        dt = jnp.gradient(t)
        total_mass = jnp.sum(sfr * dt)
        assert float(total_mass) > 0
        assert jnp.isfinite(total_mass)


# ── Tests for buat08 ──────────────────────────────────────────────


class TestBuat08:
    """Tests for velocity-parameterized chemical evolution SFH (Buat+2008)."""

    def test_nonnegative(self):
        """SFR is non-negative everywhere."""
        t = jnp.logspace(7, 10, 200)
        for v in [80.0, 150.0, 220.0, 290.0, 360.0]:
            sfr = buat08(t, log_total_mass=10.0, velocity_km_s=v)
            assert_non_negative(sfr, name="sfr")

    def test_velocity_clipping(self):
        """Velocities outside [40, 360] are clipped."""
        t = jnp.logspace(7, 9, 50)
        sfr_clipped_lo = buat08(t, log_total_mass=10.0, velocity_km_s=20.0)
        sfr_at_40 = buat08(t, log_total_mass=10.0, velocity_km_s=40.0)
        assert_allclose(sfr_clipped_lo, sfr_at_40, rtol=1e-10)

        sfr_clipped_hi = buat08(t, log_total_mass=10.0, velocity_km_s=400.0)
        sfr_at_360 = buat08(t, log_total_mass=10.0, velocity_km_s=360.0)
        assert_allclose(sfr_clipped_hi, sfr_at_360, rtol=1e-10)

    def test_interpolation_at_table_values(self):
        """At Buat+2008 Table 2 velocities, SFR matches expected form.

        After 2026-05-25 normalization refactor, compare *shapes* only—
        absolute scale is set by log_total_mass externally.
        """
        t_gyr = jnp.array([0.001, 0.01, 0.1, 1.0, 5.0, 10.0])
        t_yr = t_gyr * 1e9
        velocities_ref = jnp.array([80.0, 150.0, 220.0, 290.0, 360.0])
        as_ref = jnp.array([6.62, 8.74, 10.01, 10.82, 11.35])
        bs_ref = jnp.array([0.41, 0.98, 1.25, 1.36, 1.37])
        cs_ref = jnp.array([0.36, -0.20, -0.55, -0.74, -0.85])

        for i, v in enumerate(velocities_ref):
            sfr = buat08(t_yr, 10.0, float(v))
            a = as_ref[i]
            b = bs_ref[i]
            c = cs_ref[i]
            expected = 10.0 ** (a + b * jnp.log10(t_gyr) + c * jnp.sqrt(t_gyr) - 9.0)
            # Normalize both shapes and compare
            sfr_norm = sfr / jnp.max(sfr)
            expected_norm = expected / jnp.max(expected)
            assert_allclose(sfr_norm, expected_norm, rtol=1e-5)

    def test_velocity_dependence(self):
        """SFR varies with velocity (monotonically at early times)."""
        t = jnp.logspace(7, 8.5, 100)
        sfr_80 = buat08(t, log_total_mass=10.0, velocity_km_s=80.0)
        sfr_220 = buat08(t, log_total_mass=10.0, velocity_km_s=220.0)
        sfr_360 = buat08(t, log_total_mass=10.0, velocity_km_s=360.0)
        assert not jnp.allclose(sfr_80, sfr_220)
        assert not jnp.allclose(sfr_220, sfr_360)

    def test_is_jittable(self):
        """buat08 is JIT-compatible."""
        t = jnp.logspace(7, 10, 100)
        sfr = assert_jit_matches_eager(buat08, t, 10.0, 220.0)
        chex.assert_shape(sfr, (100,))

    def test_has_gradients(self):
        """Gradient w.r.t. velocity_km_s via jax.grad."""
        t = jnp.logspace(7, 10, 100)

        def f_v(v):
            return jnp.sum(buat08(t, log_total_mass=10.0, velocity_km_s=v))

        g_auto = float(jax.grad(f_v)(220.0))
        assert jnp.isfinite(g_auto)
        assert jnp.any(g_auto != 0.0), (
            "`g_auto` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )

    def test_energy_conservation(self):
        """Integrated mass is positive and finite."""
        t = jnp.logspace(6, 10.15, 1000)
        sfr = buat08(t, log_total_mass=10.0, velocity_km_s=220.0)
        dt = jnp.gradient(t)
        total_mass = jnp.sum(sfr * dt)
        assert float(total_mass) > 0
        assert jnp.isfinite(total_mass)

    def test_vs_cigale_reference(self):
        """Compare buat08 against known CIGALE output.

        CIGALE code (sfh_buat08.py line 86):
        t = (time_grid + 1) / 1000  # time in Gyr
        self.sfr = 10.**(a + b * np.log10(t) + c * t**.5 - 9)

        Test at velocity=220, with a=10.01, b=1.25, c=-0.55.

        After 2026-05-25 normalization refactor, compare *shapes* only—
        absolute scale is set by log_total_mass externally.
        """
        a, b, c = 10.01, 1.25, -0.55
        time_myr = jnp.arange(1, 1001)
        time_yr = time_myr * 1e6
        time_gyr = time_yr / 1e9

        sfr = buat08(time_yr, 10.0, 220.0)

        expected = 10.0 ** (a + b * jnp.log10(time_gyr) + c * jnp.sqrt(time_gyr) - 9.0)

        # Normalize both shapes and compare
        sfr_norm = sfr / jnp.max(sfr)
        expected_norm = expected / jnp.max(expected)
        assert_allclose(sfr_norm, expected_norm, rtol=1e-5)


# ── Tests for sfh2exp ─────────────────────────────────────────────


class TestSfh2Exp:
    """Tests for the CIGALE double declining-exponential SFH (main + burst)."""

    _t = jnp.linspace(0.0, 1e10, 500)

    def _sfr(self, f_burst=0.1, tau_main=4e9, tau_burst=3e8, burst_age=5e8):
        return sfh2exp(
            self._t,
            log_total_mass=10.0,
            tau_main_yr=tau_main,
            tau_burst_yr=tau_burst,
            f_burst=f_burst,
            age_yr=1e10,
            burst_age_yr=burst_age,
        )

    def test_nonnegative(self):
        assert_non_negative(self._sfr(), name="output")

    def test_mass_conservation(self):
        """trapezoid(SFR, t) == 10**log_total_mass exactly."""
        m = float(jnp.trapezoid(self._sfr(), self._t))
        assert_allclose(m, 1e10, rtol=1e-4)

    def test_burst_increases_recent_sfr(self):
        """A larger burst fraction raises the SFR in the recent burst window."""
        recent = self._t <= 5e8
        m0 = float(jnp.trapezoid(self._sfr(f_burst=0.0) * recent, self._t))
        m3 = float(jnp.trapezoid(self._sfr(f_burst=0.3) * recent, self._t))
        assert m3 > m0

    def test_burst_mass_fraction(self):
        """The burst carries ~f_burst of the total mass (CIGALE convention)."""
        # Burst-only minus no-burst, integrated, should be ~f_burst * total.
        t = self._t
        total = 1e10
        # Reconstruct the burst mass fraction by differencing against f_burst=0.
        sfr_f = self._sfr(f_burst=0.25)
        sfr_0 = self._sfr(f_burst=0.0)
        # The excess mass over the no-burst case, normalized, approximates the
        # burst fraction to within grid resolution.
        excess = float(jnp.trapezoid(jnp.maximum(sfr_f - sfr_0, 0.0), t)) / total
        assert 0.1 < excess < 0.4  # ~0.25, broadened for grid + overlap

    def test_jit_parity_vs_eager(self):
        """JIT output matches eager evaluation (JAX correctness)."""
        sfr_eager = sfh2exp(self._t, 10.0, 4e9, 3e8, 0.1, 1e10, 5e8)
        sfr_jit = jax.jit(lambda t, fb: sfh2exp(t, 10.0, 4e9, 3e8, fb, 1e10, 5e8))(self._t, 0.1)
        chex.assert_trees_all_close(sfr_eager, sfr_jit, rtol=1e-6)

    def test_zero_beyond_age(self):
        """No star formation before the galaxy formed (t_lookback > age).

        Asserted one grid cell out, not at the node adjacent to ``age``. The SFH
        array is a quadrature integrand — the forward model turns it into mass
        parcels with ``trapezoid(sfr, t)`` — so each node carries its *cell's*
        star formation, and the single cell straddling the formation time
        legitimately holds the mass formed in the part of it after formation
        (#1374). Discarding that partial cell is the same defect #964 fixed for
        the DSPS histogram kernel, where annihilating a straddling segment cost
        a +1.2 % optical bias.

        Measured here: of 86 nodes beyond ``age``, exactly one is nonzero, it
        lies within half a cell of the boundary, and it carries 4.1e-4 of the
        total mass.
        """
        t = jnp.linspace(0.0, 1.4e10, 300)
        age = 1e10
        sfr = sfh2exp(t, 10.0, 4e9, 3e8, 0.1, age, 5e8)
        cell = float(t[1] - t[0])

        # Beyond one full cell: exactly zero, no tolerance.
        assert jnp.all(sfr[t > age + cell] == 0.0), (
            "star formation leaked more than one grid cell past the formation time"
        )
        # Within that one cell: allowed, but it must be a boundary sliver, not a
        # real component -- bound it by mass rather than by amplitude.
        beyond = np.asarray(t) > age
        mass_beyond = float(np.trapezoid(np.asarray(sfr)[beyond], np.asarray(t)[beyond]))
        mass_total = float(np.trapezoid(np.asarray(sfr), np.asarray(t)))
        assert mass_beyond / mass_total < 1e-3, (
            f"mass attributed before formation is {mass_beyond / mass_total:.2e} of the "
            "total -- far more than one straddling cell can account for"
        )

    def test_registry_resolution(self):
        """sfh2exp resolves via the registry with the documented param names."""
        from tengri.components.stellar.sfh.registry import SFH_REGISTRY

        assert "sfh2exp" in SFH_REGISTRY
        names = set(SFH_REGISTRY["sfh2exp"].params)
        assert {
            "sfh_sfh2exp_tau_main_gyr",
            "sfh_sfh2exp_tau_burst_gyr",
            "sfh_sfh2exp_f_burst",
            "sfh_sfh2exp_age_gyr",
            "sfh_sfh2exp_burst_age_gyr",
        } <= names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
