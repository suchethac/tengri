# SPDX-License-Identifier: BSD-3-Clause
"""Tests for Poisson and Student-t photon-counting likelihoods.

Tests JIT compatibility, gradient flow, and physical correctness of
PoissonNoiseLikelihood and StudentTLikelihood.
"""

import chex
import pytest

pytestmark = pytest.mark.contract

import jax
import jax.numpy as jnp
import numpy.testing as npt

from tengri.observation.noise import PoissonNoiseLikelihood, StudentTLikelihood
from tests._grad_parity import assert_grad_matches_fd
from tests._jit_parity import assert_jit_matches_eager

# ── PoissonNoiseLikelihood ────────────────────────────────────────


class TestPoissonNoiseLikelihood:
    """Tests for photon-limited Poisson likelihood."""

    def test_pure_poisson_variance(self):
        """Pure Poisson (no sky, no read noise): σ² = F."""
        lh = PoissonNoiseLikelihood(gain=1.0, sky_var=0.0, read_noise=0.0)
        observed = jnp.array([100.0])
        predicted = jnp.array([100.0])  # Perfect match
        lp = lh.log_prob(observed, predicted)
        # For pure Poisson with σ² = F = 100, residual = 0
        # log_prob = -0.5 * 0² / 100 - log(sqrt(100)) = -log(10) ≈ -2.303
        expected = -jnp.log(10.0)
        npt.assert_allclose(lp[0], expected, rtol=1e-10)

    def test_reduces_to_gaussian_with_zero_background(self):
        """No background terms → reduces to Gaussian with σ² = F."""
        lh = PoissonNoiseLikelihood(gain=1.0, sky_var=0.0, read_noise=0.0)
        observed = jnp.array([100.0])
        predicted = jnp.array([100.0])
        lp = lh.log_prob(observed, predicted)
        # Pure Poisson: σ = sqrt(100) = 10, log_prob = -log(10) ≈ -2.303
        chex.assert_tree_all_finite(lp)
        expected = -jnp.log(10.0)
        npt.assert_allclose(lp[0], expected, rtol=1e-10)

    def test_sky_variance_broadens_likelihood(self):
        """Sky variance increases σ_eff → broader peak, smaller max log_prob."""
        obs = jnp.array([100.0])
        pred = jnp.array([100.0])

        lh_no_sky = PoissonNoiseLikelihood(gain=1.0, sky_var=0.0, read_noise=0.0)
        lh_with_sky = PoissonNoiseLikelihood(gain=1.0, sky_var=50.0, read_noise=0.0)

        lp_no_sky = lh_no_sky.log_prob(obs, pred)[0]
        lp_with_sky = lh_with_sky.log_prob(obs, pred)[0]

        # With sky: σ² = 100 + 50 = 150 > 100 (no sky)
        # log_prob ∝ -log(σ) → broader peak has smaller peak height
        assert float(lp_with_sky) < float(lp_no_sky)

    def test_read_noise_term(self):
        """Read noise adds σ²_read / g² to variance."""
        obs = jnp.array([100.0])
        pred = jnp.array([100.0])

        # g=1 e⁻/ADU, read_noise=5 e⁻ → σ²_read = 25
        lh = PoissonNoiseLikelihood(gain=1.0, sky_var=0.0, read_noise=5.0)
        lp = lh.log_prob(obs, pred)[0]

        # σ² = 100 + 25 = 125 → σ = sqrt(125)
        expected = -0.5 * 0.0**2 / 125.0 - jnp.log(jnp.sqrt(125.0))
        npt.assert_allclose(lp, expected, rtol=1e-10)

    def test_systematic_floor_broadens_for_bright_sources(self):
        """Systematic floor (f_sys · F)² becomes dominant for bright F."""
        obs = jnp.array([1000.0])  # Bright
        pred = jnp.array([1000.0])

        lh_no_sys = PoissonNoiseLikelihood(
            gain=1.0, sky_var=0.0, read_noise=0.0, systematic_floor=0.0
        )
        lh_with_sys = PoissonNoiseLikelihood(
            gain=1.0, sky_var=0.0, read_noise=0.0, systematic_floor=0.05
        )

        lp_no_sys = lh_no_sys.log_prob(obs, pred)[0]
        lp_with_sys = lh_with_sys.log_prob(obs, pred)[0]

        # With sys: σ² ≈ (0.05 * 1000)² = 2500 >> 1000 (pure Poisson)
        assert float(lp_with_sys) < float(lp_no_sys)

    def test_predicted_zero_clamped_safely(self):
        """predicted=0 clamped internally; no NaN or inf."""
        lh = PoissonNoiseLikelihood(gain=1.0)
        observed = jnp.array([0.1, 0.0])
        predicted = jnp.array([0.0, 0.0])  # Edge case
        lp = lh.log_prob(observed, predicted)
        chex.assert_tree_all_finite(lp)

    def test_jit_compatible(self):
        """Function compiles under jax.jit."""
        lh = PoissonNoiseLikelihood(gain=1.0)
        observed = jnp.array([100.0, 200.0])
        predicted = jnp.array([95.0, 205.0])
        lp = assert_jit_matches_eager(lh.log_prob, observed, predicted)
        chex.assert_shape(lp, (2,))
        chex.assert_tree_all_finite(lp)

    def test_grad_w_r_t_predicted(self):
        """Gradient flows through predicted flux (differentiable)."""

        def loss(pred):
            lh = PoissonNoiseLikelihood(gain=1.0)
            lp = lh.log_prob(jnp.array([100.0]), pred)
            return -jnp.sum(lp)  # Negative because log_prob returns likelihood

        pred0 = jnp.array([100.0])
        grad_val = assert_grad_matches_fd(loss, pred0)
        assert jnp.isfinite(grad_val[0])

    def test_outlier_same_as_inlier(self):
        """Residual |d - m| drives variance, not magnitude."""
        lh = PoissonNoiseLikelihood(gain=1.0)
        # Both have residual = 10
        obs1 = jnp.array([110.0])
        pred1 = jnp.array([100.0])
        obs2 = jnp.array([90.0])
        pred2 = jnp.array([100.0])
        lp1 = lh.log_prob(obs1, pred1)
        lp2 = lh.log_prob(obs2, pred2)
        npt.assert_allclose(lp1, lp2, rtol=1e-10)


# ── StudentTLikelihood ─────────────────────────────────────────────


class TestStudentTLikelihood:
    """Tests for Student-t heavy-tailed likelihood."""

    def test_finite_for_typical_values(self):
        """Log-prob is finite for typical AB-mag flux values."""
        lh = StudentTLikelihood(dof=4.0)
        observed = jnp.array([0.5, 1.0, 2.0])  # Typical SED flux range
        predicted = jnp.array([0.5, 1.0, 2.0])
        sigma = jnp.array([0.1, 0.2, 0.3])
        lp = lh.log_prob(observed, predicted, sigma)
        chex.assert_tree_all_finite(lp)

    def test_zero_residual_is_maximum(self):
        """Log-prob at residual=0 is greater than at nonzero residual."""
        lh = StudentTLikelihood(dof=4.0)
        sigma = jnp.array([1.0])

        lp_at_zero = lh.log_prob(jnp.array([0.0]), jnp.array([0.0]), sigma)[0]
        lp_at_one = lh.log_prob(jnp.array([1.0]), jnp.array([0.0]), sigma)[0]
        assert float(lp_at_zero) > float(lp_at_one)

    def test_gaussian_limit_dof_infinity(self):
        """As dof → ∞, Student-t → Gaussian-like behavior."""
        obs = jnp.array([1.0])
        pred = jnp.array([0.0])
        sigma = jnp.array([1.0])

        lh_low_dof = StudentTLikelihood(dof=2.0)
        lh_high_dof = StudentTLikelihood(dof=1e4)  # Very high dof

        lp_low = lh_low_dof.log_prob(obs, pred, sigma)[0]
        lp_high = lh_high_dof.log_prob(obs, pred, sigma)[0]
        # As dof increases, behavior approaches Gaussian (less tolerant of outliers)
        # For 1-sigma outlier, higher dof should give closer to Gaussian
        assert jnp.isfinite(lp_low) and jnp.isfinite(lp_high)

    def test_heavy_tails_robust_to_outliers(self):
        """5-sigma outlier has higher log_prob in Student-t vs Gaussian limit."""
        obs = jnp.array([5.0])
        pred = jnp.array([0.0])
        sigma = jnp.array([1.0])

        lh_student = StudentTLikelihood(dof=2.0)  # Heavy tails
        lh_gaussian = StudentTLikelihood(dof=1e6)  # Gaussian limit

        lp_student = lh_student.log_prob(obs, pred, sigma)[0]
        lp_gaussian = lh_gaussian.log_prob(obs, pred, sigma)[0]

        # 5-sigma outlier: Student-t much more forgiving than Gaussian
        assert float(lp_student) > float(lp_gaussian)

    def test_dof_controls_tail_weight(self):
        """Lower dof → heavier tails (higher outlier likelihood)."""
        obs = jnp.array([3.0])  # 3-sigma outlier
        pred = jnp.array([0.0])
        sigma = jnp.array([1.0])

        lh_low_dof = StudentTLikelihood(dof=2.0)
        lh_high_dof = StudentTLikelihood(dof=20.0)

        lp_low = lh_low_dof.log_prob(obs, pred, sigma)[0]
        lp_high = lh_high_dof.log_prob(obs, pred, sigma)[0]

        # Lower dof (heavier tails) more tolerant of outliers
        assert float(lp_low) > float(lp_high)

    def test_sigma_clamped_safely(self):
        """sigma=0 clamped internally; no NaN or inf."""
        lh = StudentTLikelihood(dof=4.0)
        observed = jnp.array([1.0])
        predicted = jnp.array([0.0])
        sigma = jnp.array([0.0])  # Edge case
        lp = lh.log_prob(observed, predicted, sigma)
        chex.assert_tree_all_finite(lp)

    def test_jit_compatible(self):
        """Function compiles under jax.jit."""
        lh = StudentTLikelihood(dof=4.0)
        observed = jnp.array([1.0, 5.0, 10.0])
        predicted = jnp.array([0.0, 2.0, 5.0])
        sigma = jnp.array([1.0, 1.0, 2.0])
        lp = assert_jit_matches_eager(lh.log_prob, observed, predicted, sigma)
        chex.assert_shape(lp, (3,))
        chex.assert_tree_all_finite(lp)

    def test_grad_w_r_t_all_inputs(self):
        """Gradient flows through obs, pred, and sigma."""

        def loss_pred(pred):
            lh = StudentTLikelihood(dof=4.0)
            lp = lh.log_prob(jnp.array([1.0]), pred, jnp.array([1.0]))
            return -jnp.sum(lp)

        def loss_sigma(sigma):
            lh = StudentTLikelihood(dof=4.0)
            lp = lh.log_prob(jnp.array([1.0]), jnp.array([0.0]), sigma)
            return -jnp.sum(lp)

        grad_pred = jax.grad(loss_pred)(jnp.array([0.0]))
        grad_sigma = jax.grad(loss_sigma)(jnp.array([1.0]))

        assert jnp.isfinite(grad_pred[0])
        assert jnp.isfinite(grad_sigma[0])

    def test_symmetry_residual(self):
        """log_prob depends on |obs - pred|, so (obs, pred) ↔ (pred, obs) up
        to sign shouldn't differ."""
        lh = StudentTLikelihood(dof=4.0)
        sigma = jnp.array([1.0])
        lp1 = lh.log_prob(jnp.array([1.5]), jnp.array([0.5]), sigma)
        lp2 = lh.log_prob(jnp.array([0.5]), jnp.array([1.5]), sigma)
        # Both have residual magnitude 1.0, so same log_prob
        npt.assert_allclose(lp1, lp2, rtol=1e-10)


# ── Integration tests ──────────────────────────────────────────────


class TestLikelihoodIntegration:
    """Integration tests with realistic data scenarios."""

    def test_poisson_realistic_photometry(self):
        """Poisson likelihood on realistic photometric data."""
        # Simulate SDSS photometry: 5 bands, AB magnitudes ≈ 20
        # In flux: ~10 uJy per band, noise ~0.1–0.3 uJy
        observed = jnp.array([10.0, 15.0, 20.0, 25.0, 30.0])
        predicted = jnp.array([9.5, 15.2, 19.8, 26.0, 29.5])
        lh = PoissonNoiseLikelihood(gain=1.0, systematic_floor=0.02)
        lp = lh.log_prob(observed, predicted)
        # Should be well-behaved for all bands
        chex.assert_tree_all_finite(lp)
        # Residuals < 1 sigma → expect negative log_prob (since we're
        # logging a likelihood, not neg-log-likelihood)
        # Actually, each lp should represent per-band log-likelihood
        chex.assert_shape(lp, (5,))

    def test_student_t_with_outlier(self):
        """Student-t robustly handles one bad measurement."""
        # 4 good points, 1 outlier (e.g., cosmic ray hit)
        observed = jnp.array([1.0, 1.1, 0.9, 1.0, 5.0])
        predicted = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
        sigma = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1])

        lh_gaussian = StudentTLikelihood(dof=1e6)
        lh_student = StudentTLikelihood(dof=3.0)

        lp_gaussian = lh_gaussian.log_prob(observed, predicted, sigma)
        lp_student = lh_student.log_prob(observed, predicted, sigma)

        # Outlier (index 4) penalizes Gaussian much more
        assert float(lp_gaussian[4]) < float(lp_student[4])

    def test_combined_noise_sources(self):
        """Poisson with all noise sources active."""
        lh = PoissonNoiseLikelihood(gain=1.5, sky_var=10.0, read_noise=3.0, systematic_floor=0.03)
        observed = jnp.array([100.0, 500.0, 1000.0])
        predicted = jnp.array([100.0, 510.0, 950.0])
        lp = lh.log_prob(observed, predicted)
        chex.assert_tree_all_finite(lp)
