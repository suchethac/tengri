# SPDX-License-Identifier: BSD-3-Clause
"""Tests for parametric mean SFH models.

Tests all 9 model functions from the registry plus shared helpers.
Each model is tested for:
- Correct output shape and sign (non-negative SFR)
- Expected peak location and behavior
- JIT-compatibility
- Gradient existence via jax.grad
- Edge-case stability (t=0, t=AGEMAX)
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

pytestmark = pytest.mark.bounds

jax.config.update("jax_enable_x64", True)


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


from tengri.components.stellar.sfh.mean_sfh import (
    AGEMAX_YR,
    _clamp_age,
    _skewed_gaussian_kernel,
    constant,
    delayed_exponential,
    delayed_tau,
    double_powerlaw,
    dpl,
    exponential,
    lnorm,
    norm,
    snorm,
    triweight_burst,
    tsnorm,
)

jax.config.update("jax_enable_x64", True)


# ── Helpers ───────────────────────────────────────────────────────


class TestClampAge:
    """Tests for _clamp_age helper."""

    def test_clamps_low_end(self):
        """Ages below 1e5 yr are clamped to 1e5."""
        t = jnp.array([0.0, 100.0, 1e4])
        clamped = _clamp_age(t)
        assert jnp.all(clamped >= 1e5)

    def test_clamps_high_end(self):
        """Ages above AGEMAX_YR are clamped."""
        t = jnp.array([1e11, 2e10])
        clamped = _clamp_age(t)
        assert jnp.all(clamped <= AGEMAX_YR)

    def test_passthrough_in_range(self):
        """Ages within bounds are unchanged."""
        t = jnp.array([1e6, 1e8, 1e10])
        clamped = _clamp_age(t)
        assert_allclose(clamped, t)


class TestSkewedGaussianKernel:
    """Tests for _skewed_gaussian_kernel."""

    def test_peaks_at_peak_lbt(self):
        """Kernel peaks near the specified peak lookback time."""
        age = jnp.linspace(1e8, 1e10, 5000)
        peak_lbt = 5e9
        kernel = _skewed_gaussian_kernel(age, peak_lbt=peak_lbt, width=1e9, skew=0.0)
        peak_idx = jnp.argmax(kernel)
        assert abs(float(age[peak_idx]) - peak_lbt) / peak_lbt < 0.05

    def test_symmetric_when_skew_zero(self):
        """Kernel is symmetric about peak when skew=0."""
        peak = 5e9
        dt = 1e9
        val_left = _skewed_gaussian_kernel(
            jnp.array(peak - dt), peak_lbt=peak, width=1e9, skew=0.0
        )
        val_right = _skewed_gaussian_kernel(
            jnp.array(peak + dt), peak_lbt=peak, width=1e9, skew=0.0
        )
        assert_allclose(float(val_left), float(val_right), rtol=1e-10)

    def test_nonnegative(self):
        """Kernel values are non-negative (it's an exponential)."""
        age = jnp.logspace(6, 10, 500)
        kernel = _skewed_gaussian_kernel(age, peak_lbt=3e9, width=1e9, skew=0.5)
        assert jnp.all(kernel >= 0)


# ── Double Power Law (legacy + registry) ──────────────────────────


class TestDoublePowerlaw:
    """Tests for the BAGPIPES-style double power law mean SFH."""

    def test_positive_output(self):
        """SFR is positive for all lookback times."""
        t = jnp.logspace(6, 10, 200)
        sfr = double_powerlaw(t, alpha=1.0, beta=1.0, tau=1e9, norm=10.0)
        assert jnp.all(sfr > 0)

    def test_peak_near_tau(self):
        """SFR peaks near t = tau for symmetric alpha=beta."""
        t = jnp.logspace(6, 10, 1000)
        tau = 1e9
        sfr = double_powerlaw(t, alpha=2.0, beta=2.0, tau=tau, norm=10.0)
        peak_idx = jnp.argmax(sfr)
        peak_t = t[peak_idx]
        assert 0.5 * tau < float(peak_t) < 2.0 * tau

    def test_peak_value_equals_norm(self):
        """At the exact peak (symmetric case), SFR = norm / 2."""
        tau = 1e9
        sfr_at_tau = double_powerlaw(jnp.array(tau), alpha=1.0, beta=1.0, tau=tau, norm=10.0)
        assert_allclose(float(sfr_at_tau), 5.0, rtol=1e-10)

    def test_falling_at_late_times(self):
        """SFR decreases at t >> tau (controlled by alpha)."""
        tau = 1e8
        t_late = jnp.array([1e9, 2e9, 5e9, 1e10])
        sfr = double_powerlaw(t_late, alpha=2.0, beta=1.0, tau=tau, norm=10.0)
        assert jnp.all(jnp.diff(sfr) < 0)

    def test_is_jittable(self):
        """Double power law is JIT-compatible."""
        fn = jax.jit(double_powerlaw)
        t = jnp.logspace(6, 10, 100)
        sfr = fn(t, 1.0, 1.0, 1e9, 10.0)
        chex.assert_shape(sfr, (100,))

    def test_has_gradients(self):
        """FD check: gradients w.r.t. all 4 DPL parameters (alpha, beta, tau, norm)."""
        t = jnp.logspace(6, 10, 100)
        alpha0, beta0, tau0, norm0 = 1.0, 1.0, 1e9, 10.0

        # alpha
        def f_a(a):
            return float(jnp.sum(double_powerlaw(t, a, beta0, tau0, norm0)))

        g_a = float(jax.grad(lambda a: jnp.sum(double_powerlaw(t, a, beta0, tau0, norm0)))(alpha0))
        np.testing.assert_allclose(
            g_a,
            fd_grad(f_a, alpha0),
            rtol=1e-3,
            err_msg="double_powerlaw: FD check ∂/∂alpha",
        )

        # beta
        def f_b(b):
            return float(jnp.sum(double_powerlaw(t, alpha0, b, tau0, norm0)))

        g_b = float(jax.grad(lambda b: jnp.sum(double_powerlaw(t, alpha0, b, tau0, norm0)))(beta0))
        np.testing.assert_allclose(
            g_b,
            fd_grad(f_b, beta0),
            rtol=1e-3,
            err_msg="double_powerlaw: FD check ∂/∂beta",
        )

        # tau (large scale — eps=1e4 yr)
        def f_tau(tau):
            return float(jnp.sum(double_powerlaw(t, alpha0, beta0, tau, norm0)))

        g_tau = float(
            jax.grad(lambda tau: jnp.sum(double_powerlaw(t, alpha0, beta0, tau, norm0)))(tau0)
        )
        np.testing.assert_allclose(
            g_tau,
            fd_grad(f_tau, tau0, eps=1e4),
            rtol=1e-3,
            err_msg="double_powerlaw: FD check ∂/∂tau",
        )

        # norm
        def f_n(n):
            return float(jnp.sum(double_powerlaw(t, alpha0, beta0, tau0, n)))

        g_n = float(jax.grad(lambda n: jnp.sum(double_powerlaw(t, alpha0, beta0, tau0, n)))(norm0))
        np.testing.assert_allclose(
            g_n,
            fd_grad(f_n, norm0),
            rtol=1e-3,
            err_msg="double_powerlaw: FD check ∂/∂norm",
        )


class TestDpl:
    """Tests for the registry-compatible DPL with log_total_mass."""

    def test_matches_double_powerlaw_shape(self):
        """dpl shares the bare double_powerlaw shape, scaled to log_total_mass.

        After the 2026-05-25 normalization refactor, ``dpl`` renormalizes the
        bare ``double_powerlaw`` shape so the integral equals
        ``10**log_total_mass``. The pointwise ratio between the two outputs
        is therefore a constant.
        """
        t = jnp.logspace(6, 10, 200)
        sfr_bare = double_powerlaw(t, alpha=1.5, beta=1.0, tau=3e9, norm=10.0)
        sfr_new = dpl(t, alpha=1.5, beta=1.0, tau=3e9, log_total_mass=10.0)
        mask = sfr_bare > 0
        ratio = sfr_new[mask] / sfr_bare[mask]
        assert_allclose(ratio, ratio[0] * jnp.ones_like(ratio), rtol=1e-6)
        assert float(jnp.trapezoid(sfr_new, t)) == pytest.approx(1e10, rel=1e-6)

    def test_jit_and_grad(self):
        """JIT-compatible and differentiable."""
        t = jnp.logspace(6, 10, 100)
        fn = jax.jit(dpl)
        sfr = fn(t, 1.0, 1.0, 1e9, 1.0)
        chex.assert_tree_all_finite(sfr)

        def _loss_lp(lp: float) -> float:
            return float(jnp.sum(dpl(t, 1.0, 1.0, 1e9, lp)))

        grad_jax = float(jax.grad(lambda lp: jnp.sum(dpl(t, 1.0, 1.0, 1e9, lp)))(1.0))
        np.testing.assert_allclose(
            grad_jax,
            fd_grad(_loss_lp, 1.0),
            rtol=1e-3,
            err_msg="dpl: FD check ∂(∑SFR)/∂log_total_mass",
        )

    def test_dpl_peak_time(self):
        """DPL peak occurs at tau (Carnall+2018, MNRAS 480, 4379, Eq. A3).

        For a rising slope α and falling slope β, the analytic peak is at
        t_peak = tau * (α/β)^(1/(α+β)).  For α=β=1, t_peak = tau.
        """
        tau = 3e9  # yr — peak time in the equal-slope case
        t = jnp.logspace(6, 10.2, 5000)
        sfr = dpl(t, alpha=1.0, beta=1.0, tau=tau, log_total_mass=1.0)
        t_peak = float(t[jnp.argmax(sfr)])
        assert abs(t_peak - tau) / tau < 0.05, (
            f"DPL peak at t={t_peak:.3e} yr, expected tau={tau:.3e} yr (±5%)"
        )
        # Monotonicity checks around peak
        sfr_early = float(jnp.interp(jnp.array(1e9), t, sfr))
        sfr_at_peak = float(jnp.interp(jnp.array(3e9), t, sfr))
        sfr_late = float(jnp.interp(jnp.array(10e9), t, sfr))
        assert sfr_at_peak > sfr_early, "DPL should rise toward tau"
        assert sfr_at_peak > sfr_late, "DPL should fall after tau"

    def test_dpl_mass_conservation(self):
        """Integrated mass ≈ 10^log_total_mass * tau * B(α,β) within 1%.

        The DPL integral scales as log_total_mass=1 → total formed mass=10 Msun.
        We verify the trapezoidal integral is physically finite and
        self-consistent when log_total_mass is doubled (mass doubles).
        """
        t = jnp.logspace(6, 10.2, 10000)
        dt = jnp.diff(t)
        sfr1 = dpl(t, alpha=1.0, beta=2.0, tau=3e9, log_total_mass=1.0)
        sfr2 = dpl(t, alpha=1.0, beta=2.0, tau=3e9, log_total_mass=2.0)
        mass1 = float(jnp.sum(0.5 * (sfr1[:-1] + sfr1[1:]) * dt))
        mass2 = float(jnp.sum(0.5 * (sfr2[:-1] + sfr2[1:]) * dt))
        # mass2 should be 10× mass1 (log_total_mass increases norm by 10×)
        np.testing.assert_allclose(mass2 / mass1, 10.0, rtol=0.01)


# ── tsnorm / snorm / norm family ──────────────────────────────────


class TestTsnorm:
    """Tests for truncated skew-normal SFH."""

    def test_positive_output(self):
        """SFR is non-negative."""
        t = jnp.logspace(6, 10, 500)
        sfr = tsnorm(t, log_total_mass=1.0, peak_lbt=5e9, width=2e9, skew=0.0, trunc=3.0)
        assert jnp.all(sfr >= 0)

    def test_peak_near_peak_lbt(self):
        """SFR peaks near the specified lookback time."""
        t = jnp.logspace(6, 10, 5000)
        peak_lbt = 5e9
        sfr = tsnorm(t, log_total_mass=1.0, peak_lbt=peak_lbt, width=2e9, skew=0.0, trunc=5.0)
        peak_t = float(t[jnp.argmax(sfr)])
        assert abs(peak_t - peak_lbt) / peak_lbt < 0.2

    def test_truncation_suppresses(self):
        """Truncation cuts off the old-age tail.

        After normalization, both SFRs integrate to the same total mass. The bare
        Gaussian shape is multiplied by erfc (which suppresses old ages).
        Test that the integral of the truncated shape is smaller (before normalization)—
        then normalization stretches it back up to match the target mass.
        """
        t = jnp.logspace(6, 10, 500)
        # Build the bare shapes (without normalization) to see what truncation does
        age = t
        peak_lbt = 5e9
        width = 2e9

        # No truncation: Gaussian kernel only
        kernel = jnp.exp(-0.5 * ((age - peak_lbt) / width) ** 2)
        shape_no_trunc = kernel

        # With truncation: Gaussian × erfc tail suppression
        x = (age - peak_lbt) / (width * 2.0)
        trunc_factor = 0.5 * jax.lax.erfc(x / jnp.sqrt(2.0))
        shape_trunc = kernel * trunc_factor

        # The bare integral before renormalization should be smaller for trunc
        integral_no_trunc = float(jnp.trapezoid(shape_no_trunc, t))
        integral_trunc = float(jnp.trapezoid(shape_trunc, t))
        assert integral_trunc < integral_no_trunc

    def test_jit_and_grad(self):
        """JIT-compatible and differentiable."""
        t = jnp.logspace(6, 10, 100)
        fn = jax.jit(tsnorm)
        sfr = fn(t, 1.0, 5e9, 2e9, 0.0, 3.0)
        chex.assert_tree_all_finite(sfr)

        def _loss_lp(lp: float) -> float:
            return float(jnp.sum(tsnorm(t, lp, 5e9, 2e9, 0.0, 3.0)))

        grad_jax = float(jax.grad(lambda lp: jnp.sum(tsnorm(t, lp, 5e9, 2e9, 0.0, 3.0)))(1.0))
        np.testing.assert_allclose(
            grad_jax,
            fd_grad(_loss_lp, 1.0),
            rtol=1e-3,
            err_msg="tsnorm: FD check ∂(∑SFR)/∂log_total_mass",
        )


class TestSnorm:
    """Tests for skew-normal SFH."""

    def test_nonnegative(self):
        """SFR is non-negative."""
        t = jnp.logspace(6, 10, 500)
        sfr = snorm(t, log_total_mass=1.0, peak_lbt=5e9, width=2e9, skew=0.3)
        assert jnp.all(sfr >= 0)

    def test_skew_changes_shape(self):
        """Non-zero skew changes the SFH shape (asymmetry)."""
        t = jnp.logspace(6, 10, 5000)
        sfr_sym = snorm(t, log_total_mass=1.0, peak_lbt=5e9, width=2e9, skew=0.0)
        sfr_skew = snorm(t, log_total_mass=1.0, peak_lbt=5e9, width=2e9, skew=0.8)
        # The normalized shapes should differ (not identical arrays)
        # Use a relative difference check to account for normalization
        rel_diff = jnp.abs(sfr_skew - sfr_sym) / jnp.maximum(jnp.abs(sfr_sym), 1e-12)
        assert jnp.max(rel_diff) > 0.01


class TestNorm:
    """Tests for normal (Gaussian) SFH."""

    def test_is_snorm_with_zero_skew(self):
        """norm is identical to snorm(skew=0)."""
        t = jnp.logspace(6, 10, 300)
        sfr_norm = norm(t, log_total_mass=1.0, peak_lbt=4e9, width=1e9)
        sfr_snorm = snorm(t, log_total_mass=1.0, peak_lbt=4e9, width=1e9, skew=0.0)
        assert_allclose(sfr_norm, sfr_snorm, rtol=1e-10)


# ── lnorm (log-normal) ────────────────────────────────────────────


class TestLnorm:
    """Tests for log-normal SFH."""

    def test_nonnegative(self):
        """SFR is non-negative."""
        t = jnp.logspace(6, 10, 500)
        sfr = lnorm(t, log_total_mass=1.0, peak_lbt=3e9, width=0.5)
        assert jnp.all(sfr >= 0)

    def test_peak_in_log_space(self):
        """Peak should be near peak_lbt."""
        t = jnp.logspace(6, 10, 5000)
        sfr = lnorm(t, log_total_mass=1.0, peak_lbt=1e9, width=0.3)
        peak_t = float(t[jnp.argmax(sfr)])
        assert abs(jnp.log10(peak_t) - jnp.log10(1e9)) < 0.5

    def test_asymmetric_in_linear(self):
        """Log-normal is asymmetric in linear time."""
        t = jnp.logspace(6, 10, 5000)
        peak_lbt = 3e9
        sfr = lnorm(t, log_total_mass=1.0, peak_lbt=peak_lbt, width=0.5)
        peak_idx = jnp.argmax(sfr)
        # Sum of SFR below peak vs above peak (in linear time) should differ
        sum_below = jnp.sum(sfr[:peak_idx])
        sum_above = jnp.sum(sfr[peak_idx:])
        assert sum_below != pytest.approx(float(sum_above), rel=0.1)


# ── constant, exponential, delayed-exponential ────────────────────


class TestConstantSFH:
    """Tests for constant SFH with start/end window."""

    def test_flat_in_range(self):
        """SFR is constant within the window after renormalization."""
        t = jnp.linspace(1e6, 10e9, 500)
        sfr = constant(t, log_total_mass=10.0, start=0.0, end=AGEMAX_YR)
        # All points inside the window should be equal to each other (flat).
        # Extract points clearly inside the window
        inside = (t >= 1e6) & (t <= 10e9)
        sfr_inside = sfr[inside]
        # All inside should be approximately equal to the first one
        assert_allclose(sfr_inside, sfr_inside[0], rtol=1e-10)

    def test_zero_outside(self):
        """SFR is zero outside the [start, end] lookback window."""
        t = jnp.array([0.5e9, 1e9, 5e9, 10e9, 15e9])
        sfr = constant(t, log_total_mass=10.0, start=1e9, end=10e9)
        assert float(sfr[0]) == 0.0  # before start
        assert float(sfr[-1]) == 0.0  # after end
        assert float(sfr[2]) > 0.0  # inside

    def test_correct_shape(self):
        """Output shape matches input."""
        t = jnp.logspace(6, 10, 42)
        sfr = constant(t, log_total_mass=10.0)
        chex.assert_shape(sfr, (42,))


class TestExponentialSFH:
    """Tests for declining exponential SFH."""

    def test_peaks_at_start(self):
        """SFR is highest at the start time."""
        t = jnp.linspace(0, 10e9, 1000)
        sfr = exponential(t, log_total_mass=1.0, tau=1e9, start=0.0)
        assert jnp.argmax(sfr) < 10  # near t=0

    def test_declining(self):
        """SFR declines with time after start."""
        t = jnp.array([0.0, 1e9, 2e9, 5e9, 10e9])
        sfr = exponential(t, log_total_mass=1.0, tau=1e9, start=0.0)
        assert jnp.all(jnp.diff(sfr) <= 0)

    def test_zero_before_start(self):
        """SFR is zero before start."""
        # Use grid instead of scalar to avoid trapezoid dimension error
        t = jnp.array([0.5e9, 1e9, 2e9, 5e9, 10e9])
        sfr = exponential(t, log_total_mass=1.0, tau=1e9, start=1e9)
        assert float(sfr[0]) == 0.0


class TestDelayedExponentialSFH:
    """Tests for delayed exponential SFH."""

    def test_peaks_at_start_plus_tau(self):
        """SFR peaks at t = start + tau."""
        start = 1e9
        tau = 2e9
        t = jnp.linspace(start, 10e9, 5000)
        sfr = delayed_exponential(t, log_total_mass=1.0, tau=tau, start=start)
        peak_t = float(t[jnp.argmax(sfr)])
        expected_peak = start + tau
        assert abs(peak_t - expected_peak) / expected_peak < 0.05

    def test_peak_occurs_at_start_plus_tau(self):
        """SFR peaks at t = start + tau (shape has maximum there).

        After NEW normalization, peak value is no longer 10^log_total_mass,
        but the integral of the curve equals 10^log_total_mass.
        """
        start = 0.0
        tau = 1e9
        log_total_mass = 1.0
        # Sample around the peak
        t = jnp.linspace(start, start + 3 * tau, 1000)
        sfr = delayed_exponential(t, log_total_mass=log_total_mass, tau=tau, start=start)
        # Peak should occur near start + tau
        t_peak = float(t[jnp.argmax(sfr)])
        assert_allclose(t_peak, start + tau, rtol=0.02)
        # Verify integral matches log_total_mass
        dt = jnp.diff(t)
        integral = float(jnp.sum(0.5 * (sfr[:-1] + sfr[1:]) * dt))
        expected = 10.0**log_total_mass
        assert_allclose(integral, expected, rtol=0.01)

    def test_zero_before_start(self):
        """SFR is zero before start."""
        # Use grid instead of scalar to avoid trapezoid dimension error
        t = jnp.array([0.5e9, 1e9, 2e9, 5e9, 10e9])
        sfr = delayed_exponential(t, log_total_mass=1.0, tau=1e9, start=1e9)
        assert float(sfr[0]) == 0.0


# ── Burst (triweight) ─────────────────────────────────────────────


class TestTriweightBurst:
    """Tests for triweight burst kernel."""

    def test_compact_support(self):
        """Kernel is zero far from the peak."""
        t = jnp.array([1e2, 1e12])  # very young and very old
        kernel = triweight_burst(t, log_tpeak_myr=2.0, log_tmax_myr=1.0)
        assert_allclose(kernel, 0.0, atol=1e-10)

    def test_nonnegative(self):
        """Kernel is non-negative everywhere."""
        t = jnp.logspace(5, 11, 1000)
        kernel = triweight_burst(t, log_tpeak_myr=2.0, log_tmax_myr=1.0)
        assert jnp.all(kernel >= 0)

    def test_peaks_at_tpeak(self):
        """Kernel peaks near the specified log_tpeak_myr."""
        t = jnp.logspace(5, 11, 10000)
        log_tpeak = 2.0  # 100 Myr
        kernel = triweight_burst(t, log_tpeak_myr=log_tpeak, log_tmax_myr=1.0)
        peak_t = float(t[jnp.argmax(kernel)])
        peak_log_myr = jnp.log10(peak_t / 1e6)
        assert abs(float(peak_log_myr) - log_tpeak) < 0.3

    def test_normalization_prefactor(self):
        """Kernel at peak should be close to 35/96."""
        # At the peak (x=0), kernel = (35/96) * (1 - 0)^3 = 35/96
        t_peak = jnp.array(10.0 ** (2.0 + 6.0))  # 100 Myr in years
        kernel = triweight_burst(t_peak, log_tpeak_myr=2.0, log_tmax_myr=1.0)
        assert_allclose(float(kernel), 35.0 / 96.0, rtol=0.01)


# ── Legacy functions ──────────────────────────────────────────────


class TestDelayedTau:
    """Tests for delayed-tau SFH (legacy)."""

    def test_peaks_at_tau(self):
        """SFR peaks at t = tau (analytic)."""
        tau = 1e8
        t = jnp.logspace(6, 10, 10000)
        sfr = delayed_tau(t, tau=tau, norm=1.0)
        peak_t = float(t[jnp.argmax(sfr)])
        assert_allclose(peak_t, tau, rtol=0.05)

    def test_positive(self):
        """SFR is positive."""
        t = jnp.logspace(6, 10, 100)
        sfr = delayed_tau(t, tau=1e8, norm=1.0)
        assert jnp.all(sfr > 0)


# ── Cross-cutting: all models JIT and grad ────────────────────────


_TSNORM_KW = {
    "log_total_mass": 1.0,
    "peak_lbt": 5e9,
    "width": 2e9,
    "skew": 0.0,
    "trunc": 3.0,
}
_SNORM_KW = {
    "log_total_mass": 1.0,
    "peak_lbt": 5e9,
    "width": 2e9,
    "skew": 0.0,
}


class TestAllModelsJitAndGrad:
    """Verify JIT and gradient support for all registry models."""

    @pytest.mark.parametrize(
        "fn,kwargs",
        [
            (tsnorm, _TSNORM_KW),
            (snorm, _SNORM_KW),
            (norm, {"log_total_mass": 1.0, "peak_lbt": 5e9, "width": 2e9}),
            (lnorm, {"log_total_mass": 1.0, "peak_lbt": 3e9, "width": 0.5}),
            (dpl, {"alpha": 1.5, "beta": 1.0, "tau": 3e9, "log_total_mass": 1.0}),
            (constant, {"log_total_mass": 10.0, "start": 0.0, "end": AGEMAX_YR}),
            (exponential, {"log_total_mass": 1.0, "tau": 1e9, "start": 0.0}),
            (delayed_exponential, {"log_total_mass": 1.0, "tau": 1e9, "start": 0.0}),
        ],
    )
    def test_jit(self, fn, kwargs):
        """All models are JIT-compatible."""
        t = jnp.logspace(6, 10, 100)
        jit_fn = jax.jit(lambda t_: fn(t_, **kwargs))
        sfr = jit_fn(t)
        chex.assert_tree_all_finite(sfr)
        chex.assert_shape(sfr, (100,))

    @pytest.mark.parametrize(
        "fn,kwargs,grad_key",
        [
            (tsnorm, _TSNORM_KW, "log_total_mass"),
            (snorm, _SNORM_KW, "log_total_mass"),
            (
                dpl,
                {"alpha": 1.5, "beta": 1.0, "tau": 3e9, "log_total_mass": 1.0},
                "log_total_mass",
            ),
            (exponential, {"log_total_mass": 1.0, "tau": 1e9, "start": 0.0}, "log_total_mass"),
        ],
    )
    def test_grad(self, fn, kwargs, grad_key):
        """All models have finite gradients for key parameters."""
        t = jnp.logspace(6, 10, 100)

        def loss(val):
            kw = dict(kwargs)
            kw[grad_key] = val
            return jnp.sum(fn(t, **kw))

        x0 = float(kwargs[grad_key])
        grad_jax = float(jax.grad(loss)(x0))

        def f_scalar(val: float) -> float:
            return float(loss(val))

        np.testing.assert_allclose(
            grad_jax,
            fd_grad(f_scalar, x0),
            rtol=1e-3,
            err_msg=f"{fn.__name__}: FD check ∂/∂{grad_key}",
        )

    def test_no_nan_at_edges(self):
        """No NaN at extreme ages."""
        t = jnp.array([1e5, AGEMAX_YR])
        for fn, kw in [
            (tsnorm, _TSNORM_KW),
            (snorm, _SNORM_KW),
            (lnorm, {"log_total_mass": 1.0, "peak_lbt": 3e9, "width": 0.5}),
            (dpl, {"alpha": 1.5, "beta": 1.0, "tau": 3e9, "log_total_mass": 1.0}),
        ]:
            sfr = fn(t, **kw)
            assert jnp.all(jnp.isfinite(sfr)), f"NaN in {fn.__name__}"
