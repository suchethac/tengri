# SPDX-License-Identifier: BSD-3-Clause
"""Tests for censored (upper/lower limit) likelihood.

Tests Feature 3 — the ``censored_neg_log_likelihood`` function in
``tengri.observation.noise`` and its integration into the standardized
loss framework.
"""

import pytest

pytestmark = pytest.mark.contract

import jax
import jax.numpy as jnp
import numpy.testing as npt

from tengri.observation.noise import (
    DETECTED,
    LOWER_LIMIT,
    UPPER_LIMIT,
    censored_neg_log_likelihood,
    variable_noise_hamiltonian,
)
from tests._jit_parity import assert_jit_matches_eager


def fd_grad(f, x: float, eps: float = 1e-5) -> float:
    """Central finite difference."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── Constants ─────────────────────────────────────────────────────


class TestMaskConstants:
    """Verify mask convention values."""

    def test_detected(self):
        assert DETECTED == 0

    def test_upper_limit(self):
        assert UPPER_LIMIT == 1

    def test_lower_limit(self):
        assert LOWER_LIMIT == -1


# ── All-detected: should match standard likelihood ────────────────


class TestCensoredAllDetected:
    """When all bands are detected, censored ≡ standard likelihood."""

    def test_matches_variable_noise_hamiltonian(self):
        """All mask=0 → same as variable_noise_hamiltonian (with logdet)."""
        data = jnp.array([1.0, 2.0, 3.0])
        noise = jnp.array([0.1, 0.2, 0.3])
        predicted = jnp.array([1.05, 2.1, 2.9])
        mask = jnp.array([DETECTED, DETECTED, DETECTED])

        censored = censored_neg_log_likelihood(data, noise, predicted, mask)
        standard = variable_noise_hamiltonian(data, noise, predicted, 0.0)
        npt.assert_allclose(float(censored), float(standard), rtol=1e-10)

    def test_matches_with_fcal(self):
        """All detected + f_cal: censored matches standard."""
        data = jnp.array([1.0, 2.0])
        noise = jnp.array([0.1, 0.2])
        predicted = jnp.array([0.95, 2.15])
        mask = jnp.array([DETECTED, DETECTED])
        f_cal = 0.05

        censored = censored_neg_log_likelihood(data, noise, predicted, mask, f_cal=f_cal)
        standard = variable_noise_hamiltonian(data, noise, predicted, f_cal)
        npt.assert_allclose(float(censored), float(standard), rtol=1e-10)


# ── Upper limits ──────────────────────────────────────────────────


class TestUpperLimit:
    """Tests for upper limit (mask=1) behavior."""

    def test_model_below_limit_low_energy(self):
        """SEDModel << limit → Φ ≈ 1 → energy ≈ 0."""
        data = jnp.array([10.0])  # upper limit at 10
        noise = jnp.array([1.0])
        predicted = jnp.array([1.0])  # well below limit
        mask = jnp.array([UPPER_LIMIT])

        energy = censored_neg_log_likelihood(data, noise, predicted, mask)
        assert float(energy) < 0.5

    def test_model_above_limit_high_energy(self):
        """SEDModel >> limit → Φ → 0 → energy large."""
        data = jnp.array([1.0])  # upper limit at 1
        noise = jnp.array([0.1])
        predicted = jnp.array([10.0])  # well above limit
        mask = jnp.array([UPPER_LIMIT])

        energy = censored_neg_log_likelihood(data, noise, predicted, mask)
        assert float(energy) > 10.0

    def test_model_at_limit_moderate_energy(self):
        """SEDModel = limit → Φ(0) = 0.5 → energy = -log(0.5) ≈ 0.693."""
        data = jnp.array([5.0])
        noise = jnp.array([1.0])
        predicted = jnp.array([5.0])
        mask = jnp.array([UPPER_LIMIT])

        energy = censored_neg_log_likelihood(data, noise, predicted, mask)
        expected = -jnp.log(0.5)
        npt.assert_allclose(float(energy), float(expected), rtol=1e-6)

    def test_energy_monotonic_with_model_flux(self):
        """As model flux increases past limit, energy increases."""
        data = jnp.array([5.0])
        noise = jnp.array([1.0])
        mask = jnp.array([UPPER_LIMIT])

        energies = []
        for m in [0.0, 2.0, 5.0, 8.0, 12.0]:
            e = censored_neg_log_likelihood(data, noise, jnp.array([m]), mask)
            energies.append(float(e))

        for i in range(len(energies) - 1):
            assert energies[i] <= energies[i + 1] + 1e-10, (
                f"Energy not monotonic: E({i})={energies[i]:.4f} "
                f"> E({i + 1})={energies[i + 1]:.4f}"
            )


# ── Lower limits ──────────────────────────────────────────────────


class TestLowerLimit:
    """Tests for lower limit (mask=-1) behavior."""

    def test_model_above_limit_low_energy(self):
        """SEDModel >> limit → Φ ≈ 1 → energy ≈ 0."""
        data = jnp.array([1.0])  # lower limit at 1
        noise = jnp.array([0.1])
        predicted = jnp.array([10.0])  # well above limit
        mask = jnp.array([LOWER_LIMIT])

        energy = censored_neg_log_likelihood(data, noise, predicted, mask)
        assert float(energy) < 0.5

    def test_model_below_limit_high_energy(self):
        """SEDModel << limit → energy large."""
        data = jnp.array([10.0])  # lower limit at 10
        noise = jnp.array([1.0])
        predicted = jnp.array([1.0])  # well below limit
        mask = jnp.array([LOWER_LIMIT])

        energy = censored_neg_log_likelihood(data, noise, predicted, mask)
        assert float(energy) > 10.0

    def test_model_at_limit_moderate_energy(self):
        """SEDModel = limit → energy = -log(0.5) ≈ 0.693."""
        data = jnp.array([5.0])
        noise = jnp.array([1.0])
        predicted = jnp.array([5.0])
        mask = jnp.array([LOWER_LIMIT])

        energy = censored_neg_log_likelihood(data, noise, predicted, mask)
        expected = -jnp.log(0.5)
        npt.assert_allclose(float(energy), float(expected), rtol=1e-6)

    def test_energy_monotonic_decreasing_with_model_flux(self):
        """As model flux increases past lower limit, energy decreases."""
        data = jnp.array([5.0])
        noise = jnp.array([1.0])
        mask = jnp.array([LOWER_LIMIT])

        energies = []
        for m in [0.0, 2.0, 5.0, 8.0, 12.0]:
            e = censored_neg_log_likelihood(data, noise, jnp.array([m]), mask)
            energies.append(float(e))

        for i in range(len(energies) - 1):
            assert energies[i] >= energies[i + 1] - 1e-10, (
                f"Energy not decreasing: E({i})={energies[i]:.4f} "
                f"< E({i + 1})={energies[i + 1]:.4f}"
            )

    def test_symmetry_with_upper_limit(self):
        """Upper and lower limits are symmetric: E_upper(m, f_up) = E_lower(2f - m, f_low=f)."""
        data = jnp.array([5.0])
        noise = jnp.array([1.0])

        e_upper = censored_neg_log_likelihood(
            data, noise, jnp.array([3.0]), jnp.array([UPPER_LIMIT])
        )
        e_lower = censored_neg_log_likelihood(
            data, noise, jnp.array([7.0]), jnp.array([LOWER_LIMIT])
        )
        npt.assert_allclose(float(e_upper), float(e_lower), rtol=1e-10)


# ── Mixed masks ───────────────────────────────────────────────────


class TestMixedMask:
    """Tests with a mixture of detected, upper, and lower limit bands."""

    def test_three_band_mixed(self):
        """One detected, one upper limit, one lower limit."""
        data = jnp.array([1.0, 5.0, 2.0])
        noise = jnp.array([0.1, 0.5, 0.2])
        predicted = jnp.array([1.05, 3.0, 5.0])
        mask = jnp.array([DETECTED, UPPER_LIMIT, LOWER_LIMIT])

        energy = censored_neg_log_likelihood(data, noise, predicted, mask)
        assert jnp.isfinite(energy)

    def test_sum_of_individual_bands(self):
        """Total energy = sum of per-band energies."""
        data = jnp.array([1.0, 5.0, 2.0])
        noise = jnp.array([0.1, 0.5, 0.2])
        predicted = jnp.array([1.05, 3.0, 5.0])
        mask = jnp.array([DETECTED, UPPER_LIMIT, LOWER_LIMIT])

        total = censored_neg_log_likelihood(data, noise, predicted, mask)

        e0 = censored_neg_log_likelihood(
            data[0:1], noise[0:1], predicted[0:1], jnp.array([DETECTED])
        )
        e1 = censored_neg_log_likelihood(
            data[1:2], noise[1:2], predicted[1:2], jnp.array([UPPER_LIMIT])
        )
        e2 = censored_neg_log_likelihood(
            data[2:3], noise[2:3], predicted[2:3], jnp.array([LOWER_LIMIT])
        )
        npt.assert_allclose(float(total), float(e0 + e1 + e2), rtol=1e-10)

    def test_all_upper_limits(self):
        """All bands upper limits — should still work."""
        data = jnp.array([5.0, 3.0])
        noise = jnp.array([1.0, 0.5])
        predicted = jnp.array([2.0, 1.0])
        mask = jnp.array([UPPER_LIMIT, UPPER_LIMIT])

        energy = censored_neg_log_likelihood(data, noise, predicted, mask)
        assert jnp.isfinite(energy)
        assert float(energy) > 0

    def test_all_lower_limits(self):
        """All bands lower limits."""
        data = jnp.array([1.0, 2.0])
        noise = jnp.array([0.1, 0.2])
        predicted = jnp.array([5.0, 10.0])
        mask = jnp.array([LOWER_LIMIT, LOWER_LIMIT])

        energy = censored_neg_log_likelihood(data, noise, predicted, mask)
        assert jnp.isfinite(energy)
        assert float(energy) >= 0


# ── Variable noise (f_cal) interaction ────────────────────────────


class TestCensoredWithFcal:
    """Tests for censored likelihood with calibration floor."""

    def test_fcal_increases_sigma_reduces_upper_limit_penalty(self):
        """Larger σ_eff → z-score lower → less penalty for upper limit."""
        data = jnp.array([5.0])
        noise = jnp.array([1.0])
        predicted = jnp.array([8.0])  # above limit
        mask = jnp.array([UPPER_LIMIT])

        e_no_cal = censored_neg_log_likelihood(data, noise, predicted, mask, f_cal=0.0)
        e_with_cal = censored_neg_log_likelihood(data, noise, predicted, mask, f_cal=0.5)
        assert float(e_with_cal) < float(e_no_cal)

    def test_fcal_zero_matches_no_fcal(self):
        data = jnp.array([5.0, 3.0])
        noise = jnp.array([1.0, 0.5])
        predicted = jnp.array([4.0, 2.0])
        mask = jnp.array([UPPER_LIMIT, LOWER_LIMIT])

        e0 = censored_neg_log_likelihood(data, noise, predicted, mask, f_cal=0.0)
        e_default = censored_neg_log_likelihood(data, noise, predicted, mask)
        npt.assert_allclose(float(e0), float(e_default), rtol=1e-12)


# ── Student-t interaction ─────────────────────────────────────────


class TestCensoredWithStudentT:
    """Tests for censored likelihood with Student-t detected bands."""

    def test_detected_band_uses_student_t(self):
        """Detected band with dof should differ from Gaussian."""
        data = jnp.array([1.0])
        noise = jnp.array([0.1])
        predicted = jnp.array([1.5])  # 5σ residual
        mask = jnp.array([DETECTED])

        e_gauss = censored_neg_log_likelihood(data, noise, predicted, mask, dof=None)
        e_t2 = censored_neg_log_likelihood(data, noise, predicted, mask, dof=2.0)
        assert float(e_t2) < float(e_gauss)

    def test_censored_bands_unaffected_by_dof(self):
        """Upper/lower limit CDF is always Gaussian, regardless of dof."""
        data = jnp.array([5.0])
        noise = jnp.array([1.0])
        predicted = jnp.array([3.0])

        for mask_val in [UPPER_LIMIT, LOWER_LIMIT]:
            mask = jnp.array([mask_val])
            e_none = censored_neg_log_likelihood(data, noise, predicted, mask, dof=None)
            e_dof2 = censored_neg_log_likelihood(data, noise, predicted, mask, dof=2.0)
            npt.assert_allclose(float(e_none), float(e_dof2), rtol=1e-10)

    def test_student_t_large_dof_converges_to_gaussian(self):
        """Student-t(ν→∞) → Gaussian for detected bands."""
        data = jnp.array([1.0])
        noise = jnp.array([0.1])
        predicted = jnp.array([1.05])
        mask = jnp.array([DETECTED])

        e_gauss = censored_neg_log_likelihood(data, noise, predicted, mask, dof=None)
        e_large = censored_neg_log_likelihood(data, noise, predicted, mask, dof=1e6)
        npt.assert_allclose(float(e_large), float(e_gauss), rtol=1e-4)


# ── JIT compatibility ─────────────────────────────────────────────


class TestCensoredJIT:
    """Tests for JIT compilation."""

    def test_jit_compiles(self):
        data = jnp.array([1.0, 5.0])
        noise = jnp.array([0.1, 1.0])
        predicted = jnp.array([0.95, 3.0])
        mask = jnp.array([DETECTED, UPPER_LIMIT])

        result = assert_jit_matches_eager(
            censored_neg_log_likelihood, data, noise, predicted, mask
        )
        assert jnp.isfinite(result)

    def test_jit_with_fcal(self):
        data = jnp.array([1.0, 5.0, 2.0])
        noise = jnp.array([0.1, 1.0, 0.2])
        predicted = jnp.array([0.95, 3.0, 5.0])
        mask = jnp.array([DETECTED, UPPER_LIMIT, LOWER_LIMIT])

        result = assert_jit_matches_eager(
            lambda d, n, p, m: censored_neg_log_likelihood(d, n, p, m, f_cal=0.05),
            data,
            noise,
            predicted,
            mask,
        )
        assert jnp.isfinite(result)


# ── Gradients ─────────────────────────────────────────────────────


class TestCensoredGradients:
    """Tests for gradient flow through censored likelihood."""

    def test_grad_through_predicted_detected(self):
        """Gradient w.r.t. predicted flux for detected band."""
        data = jnp.array([1.0])
        noise = jnp.array([0.1])
        mask = jnp.array([DETECTED])

        def loss(m):
            return censored_neg_log_likelihood(data, noise, jnp.array([m]), mask)

        grad_jax = float(jax.grad(loss)(1.05))
        grad_fd = fd_grad(loss, 1.05)
        npt.assert_allclose(grad_jax, grad_fd, rtol=1e-3)

    def test_grad_through_predicted_upper_limit(self):
        """Gradient for upper limit: increasing model past limit increases energy."""
        data = jnp.array([5.0])
        noise = jnp.array([1.0])
        mask = jnp.array([UPPER_LIMIT])

        def loss(m):
            return censored_neg_log_likelihood(data, noise, jnp.array([m]), mask)

        # SEDModel at 7.0 (above limit 5.0) → gradient should be positive
        # (increasing model further increases energy)
        grad = float(jax.grad(loss)(7.0))
        assert grad > 0, f"Expected positive gradient, got {grad}"

        # SEDModel at 2.0 (below limit 5.0) → gradient should be near zero
        # (model is safely below limit, almost no penalty change)
        grad_low = float(jax.grad(loss)(2.0))
        assert abs(grad_low) < abs(grad)

    def test_grad_through_predicted_lower_limit(self):
        """Gradient for lower limit: decreasing model below limit increases energy."""
        data = jnp.array([5.0])
        noise = jnp.array([1.0])
        mask = jnp.array([LOWER_LIMIT])

        def loss(m):
            return censored_neg_log_likelihood(data, noise, jnp.array([m]), mask)

        # SEDModel at 3.0 (below limit 5.0) → gradient should be negative
        # (increasing model reduces energy)
        grad = float(jax.grad(loss)(3.0))
        assert grad < 0, f"Expected negative gradient, got {grad}"

    def test_grad_agrees_with_finite_diff_upper(self):
        """Autodiff matches finite differences for upper limit."""
        data = jnp.array([5.0])
        noise = jnp.array([1.0])
        mask = jnp.array([UPPER_LIMIT])

        def loss(m):
            return censored_neg_log_likelihood(data, noise, jnp.array([m]), mask)

        for m_val in [2.0, 5.0, 8.0]:
            grad_jax = float(jax.grad(loss)(m_val))
            grad_fd = fd_grad(loss, m_val)
            npt.assert_allclose(
                grad_jax,
                grad_fd,
                rtol=1e-3,
                err_msg=f"m={m_val}: autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
            )

    def test_grad_agrees_with_finite_diff_lower(self):
        """Autodiff matches finite differences for lower limit."""
        data = jnp.array([5.0])
        noise = jnp.array([1.0])
        mask = jnp.array([LOWER_LIMIT])

        def loss(m):
            return censored_neg_log_likelihood(data, noise, jnp.array([m]), mask)

        for m_val in [2.0, 5.0, 8.0]:
            grad_jax = float(jax.grad(loss)(m_val))
            grad_fd = fd_grad(loss, m_val)
            npt.assert_allclose(
                grad_jax,
                grad_fd,
                rtol=1e-3,
                err_msg=f"m={m_val}: autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
            )

    def test_grad_through_fcal(self):
        """Gradient w.r.t. f_cal in mixed-mask setting."""
        data = jnp.array([1.0, 5.0, 2.0])
        noise = jnp.array([0.1, 1.0, 0.2])
        predicted = jnp.array([1.05, 7.0, 5.0])
        mask = jnp.array([DETECTED, UPPER_LIMIT, LOWER_LIMIT])

        def loss(f_cal):
            return censored_neg_log_likelihood(data, noise, predicted, mask, f_cal=f_cal)

        grad_jax = float(jax.grad(loss)(0.05))
        grad_fd = fd_grad(loss, 0.05)
        npt.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
        )


# ── Edge cases ────────────────────────────────────────────────────


class TestCensoredEdgeCases:
    """Edge cases for numerical stability."""

    def test_single_band_detected(self):
        data = jnp.array([1.0])
        noise = jnp.array([0.1])
        predicted = jnp.array([1.0])
        mask = jnp.array([DETECTED])
        energy = censored_neg_log_likelihood(data, noise, predicted, mask)
        assert jnp.isfinite(energy)

    def test_large_residual_upper_limit(self):
        """SEDModel 100σ above limit — energy should be finite, not inf."""
        data = jnp.array([1.0])
        noise = jnp.array([0.01])
        predicted = jnp.array([2.0])  # 100σ above
        mask = jnp.array([UPPER_LIMIT])

        energy = censored_neg_log_likelihood(data, noise, predicted, mask)
        assert jnp.isfinite(energy)
        assert float(energy) > 100.0

    def test_large_residual_lower_limit(self):
        """SEDModel 100σ below limit — energy should be finite, not inf."""
        data = jnp.array([2.0])
        noise = jnp.array([0.01])
        predicted = jnp.array([1.0])  # 100σ below
        mask = jnp.array([LOWER_LIMIT])

        energy = censored_neg_log_likelihood(data, noise, predicted, mask)
        assert jnp.isfinite(energy)
        assert float(energy) > 100.0

    def test_many_bands(self):
        """Works with realistic number of bands (20+)."""
        n = 25
        data = jnp.ones(n)
        noise = 0.1 * jnp.ones(n)
        predicted = 1.05 * jnp.ones(n)
        mask = jnp.zeros(n, dtype=int)
        mask = mask.at[5].set(UPPER_LIMIT)
        mask = mask.at[10].set(LOWER_LIMIT)

        energy = censored_neg_log_likelihood(data, noise, predicted, mask)
        assert jnp.isfinite(energy)
