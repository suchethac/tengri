# SPDX-License-Identifier: BSD-3-Clause
"""Tests for analytical emission-line marginalization.

Verifies:
1. Design matrix shape and normalization
2. Recovery of known line amplitudes from synthetic data
3. Marginalized likelihood > non-marginalized when lines present
4. Gradient through marginalization is finite
5. Prior shrinks amplitudes toward zero
6. JIT compilation works
7. Multiple lines don't interfere
8. Zero-amplitude lines don't affect likelihood
"""

import chex
import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from tengri.observation.eline_marginalization import (
    DEFAULT_LINE_WAVELENGTHS,
    build_eline_design_matrix,
    marginalize_emission_lines,
    predict_with_marginalized_lines,
)
from tests._bounds import assert_non_negative
from tests._grad_parity import assert_grad_matches_fd

pytestmark = pytest.mark.bounds


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def wave_grid():
    """Observed wavelength grid covering optical range."""
    return jnp.linspace(3500.0, 7500.0, 4000)


@pytest.fixture
def simple_setup(wave_grid):
    """Single H-alpha line at z=0, R=1000."""
    line_waves = jnp.array([6562.80])
    R = 1000.0
    z = 0.0
    G = build_eline_design_matrix(wave_grid, line_waves, R, z)
    return wave_grid, G, line_waves, R, z


@pytest.fixture
def multi_line_setup(wave_grid):
    """Three well-separated lines: H-beta, [OIII]5007, H-alpha at z=0."""
    line_waves = jnp.array([4861.33, 5006.84, 6562.80])
    R = 2000.0
    z = 0.0
    G = build_eline_design_matrix(wave_grid, line_waves, R, z)
    return wave_grid, G, line_waves, R, z


# ── 1. Design matrix shape and normalization ──────────────────────


class TestDesignMatrix:
    """Tests for build_eline_design_matrix."""

    def test_shape(self, wave_grid):
        """Design matrix should be (n_pix, n_lines)."""
        n_lines = 5
        line_waves = DEFAULT_LINE_WAVELENGTHS[:n_lines]
        G = build_eline_design_matrix(wave_grid, line_waves, 1000.0, 0.0)
        assert G.shape == (len(wave_grid), n_lines)

    def test_columns_normalized(self, wave_grid):
        """Each column should integrate to ~1 (normalized Gaussian)."""
        line_waves = jnp.array([5000.0])
        G = build_eline_design_matrix(wave_grid, line_waves, 1000.0, 0.0)
        dlam = wave_grid[1] - wave_grid[0]
        integral = jnp.sum(G[:, 0]) * dlam
        assert_allclose(
            float(integral), 1.0, rtol=0.01, err_msg="Gaussian profile should integrate to 1"
        )

    def test_peak_at_line_center(self, wave_grid):
        """Peak of each column should be at the redshifted line center."""
        z = 0.1
        line_waves = jnp.array([5000.0])
        G = build_eline_design_matrix(wave_grid, line_waves, 1000.0, z)
        peak_idx = jnp.argmax(G[:, 0])
        peak_wave = float(wave_grid[peak_idx])
        expected = 5000.0 * (1.0 + z)
        assert_allclose(peak_wave, expected, atol=2.0)

    def test_redshift_shifts_lines(self, wave_grid):
        """Redshift should shift the design matrix."""
        line_waves = jnp.array([5000.0])
        G_z0 = build_eline_design_matrix(wave_grid, line_waves, 1000.0, 0.0)
        G_z1 = build_eline_design_matrix(wave_grid, line_waves, 1000.0, 0.5)
        peak_z0 = float(wave_grid[jnp.argmax(G_z0[:, 0])])
        peak_z1 = float(wave_grid[jnp.argmax(G_z1[:, 0])])
        assert peak_z1 > peak_z0, "Higher z should shift peak redward"

    def test_non_negative(self, wave_grid):
        """Design matrix should be non-negative (Gaussian profiles)."""
        G = build_eline_design_matrix(wave_grid, DEFAULT_LINE_WAVELENGTHS[:3], 1000.0, 0.0)
        assert_non_negative(G, name="G")

    def test_default_line_list(self, wave_grid):
        """Should work with the full default line list."""
        G = build_eline_design_matrix(wave_grid, DEFAULT_LINE_WAVELENGTHS, 1000.0, 0.0)
        assert G.shape == (len(wave_grid), len(DEFAULT_LINE_WAVELENGTHS))


# ── 2. Recovery of known line amplitudes ──────────────────────────


class TestAmplitudeRecovery:
    """Test that known line amplitudes are recovered from synthetic data."""

    def test_single_line_recovery(self, simple_setup):
        """Should recover a single injected line amplitude."""
        wave, G, _, _, _ = simple_setup
        true_amp = 5.0
        noise = jnp.ones_like(wave) * 0.01
        continuum = jnp.ones_like(wave)
        data = continuum + true_amp * G[:, 0] + 0.0  # noiseless
        residual = data - continuum

        _, a_hat, _ = marginalize_emission_lines(residual, noise, G)
        assert_allclose(
            float(a_hat[0]), true_amp, rtol=1e-4, err_msg="Should recover injected amplitude"
        )

    def test_multi_line_recovery(self, multi_line_setup):
        """Should recover multiple injected line amplitudes."""
        wave, G, _, _, _ = multi_line_setup
        true_amps = jnp.array([2.0, 8.0, 5.0])
        noise = jnp.ones_like(wave) * 0.01
        continuum = jnp.zeros_like(wave)
        data = continuum + G @ true_amps
        residual = data - continuum

        _, a_hat, _ = marginalize_emission_lines(residual, noise, G)
        assert_allclose(
            a_hat, true_amps, rtol=1e-3, err_msg="Should recover all injected amplitudes"
        )

    def test_recovery_with_noise(self, simple_setup):
        """Recovery should be approximate with noise, but unbiased."""
        wave, G, _, _, _ = simple_setup
        true_amp = 10.0
        noise_level = 0.1
        noise = jnp.ones_like(wave) * noise_level
        key = jax.random.PRNGKey(42)
        noise_realization = jax.random.normal(key, wave.shape) * noise_level
        continuum = jnp.ones_like(wave)
        data = continuum + true_amp * G[:, 0] + noise_realization
        residual = data - continuum

        _, a_hat, a_cov = marginalize_emission_lines(residual, noise, G)
        # The recovered amplitude should be within ~3 sigma of truth
        sigma_a = jnp.sqrt(a_cov[0, 0])
        assert abs(float(a_hat[0]) - true_amp) < 3.0 * float(sigma_a), (
            f"Recovered {float(a_hat[0]):.3f} vs true {true_amp}, sigma={float(sigma_a):.3f}"
        )


# ── 3. Marginalized likelihood > non-marginalized when lines present


class TestLikelihoodImprovement:
    """Marginalized likelihood should improve when lines are present."""

    def test_marg_better_than_continuum_only(self, simple_setup):
        """Marginalized lnL should exceed continuum-only lnL."""
        wave, G, _, _, _ = simple_setup
        true_amp = 5.0
        noise = jnp.ones_like(wave) * 0.1
        continuum = jnp.ones_like(wave)
        data = continuum + true_amp * G[:, 0]
        residual = data - continuum

        ln_l_marg, _, _ = marginalize_emission_lines(residual, noise, G)

        # Continuum-only log-likelihood (residual includes the line signal)
        ln_l_cont = -0.5 * jnp.sum(residual**2 / noise**2)

        # Marginalized should be much better (less negative)
        assert float(ln_l_marg) > float(ln_l_cont), (
            f"Marginalized lnL ({float(ln_l_marg):.1f}) should exceed "
            f"continuum-only lnL ({float(ln_l_cont):.1f})"
        )


# ── 4. Gradient through marginalization is finite ─────────────────


class TestGradients:
    """Test differentiability through the marginalization."""

    def test_grad_wrt_continuum(self, simple_setup):
        """Gradient of lnL w.r.t. continuum model should be finite."""
        wave, G, _, _, _ = simple_setup
        noise = jnp.ones_like(wave) * 0.1
        data = jnp.ones_like(wave) * 1.5

        def loss(continuum):
            residual = data - continuum
            ln_l, _, _ = marginalize_emission_lines(residual, noise, G)
            return ln_l

        grad = assert_grad_matches_fd(loss, jnp.ones_like(wave))
        chex.assert_tree_all_finite(grad)

    def test_grad_wrt_noise(self, simple_setup):
        """Gradient of lnL w.r.t. noise should be finite."""
        wave, G, _, _, _ = simple_setup
        continuum = jnp.ones_like(wave)
        data = continuum + 2.0 * G[:, 0]
        residual = data - continuum

        def loss(noise):
            ln_l, _, _ = marginalize_emission_lines(residual, noise, G)
            return ln_l

        grad = assert_grad_matches_fd(loss, jnp.ones_like(wave) * 0.1)
        chex.assert_tree_all_finite(grad)

    def test_grad_through_predict(self, simple_setup):
        """Gradient should flow through predict_with_marginalized_lines."""
        wave, G, _, _, _ = simple_setup
        noise = jnp.ones_like(wave) * 0.1
        data = jnp.ones_like(wave) * 2.0

        def loss(continuum):
            residual = data - continuum
            _, a_hat, _ = marginalize_emission_lines(residual, noise, G)
            model_full = predict_with_marginalized_lines(continuum, G, a_hat)
            return jnp.sum((data - model_full) ** 2)

        grad = assert_grad_matches_fd(loss, jnp.ones_like(wave))
        chex.assert_tree_all_finite(grad)


# ── 5. Prior shrinks amplitudes toward zero ───────────────────────


class TestPriorEffect:
    """Test that the prior regularizes line amplitudes."""

    def test_tight_prior_shrinks_to_zero(self, simple_setup):
        """Very tight prior should shrink amplitudes toward zero."""
        wave, G, _, _, _ = simple_setup
        true_amp = 5.0
        noise = jnp.ones_like(wave) * 0.1
        continuum = jnp.ones_like(wave)
        data = continuum + true_amp * G[:, 0]
        residual = data - continuum

        # Very tight prior: variance = 1e-10
        _, a_hat_tight, _ = marginalize_emission_lines(
            residual, noise, G, prior_variance=jnp.array(1e-10)
        )
        # Wide prior: variance = 1e10
        _, a_hat_wide, _ = marginalize_emission_lines(
            residual, noise, G, prior_variance=jnp.array(1e10)
        )

        assert abs(float(a_hat_tight[0])) < abs(float(a_hat_wide[0])), (
            "Tight prior should shrink amplitude more than wide prior"
        )
        assert abs(float(a_hat_tight[0])) < 0.1, "Very tight prior should push amplitude near zero"

    def test_per_line_prior(self, multi_line_setup):
        """Per-line prior variances should independently regulate lines."""
        wave, G, _, _, _ = multi_line_setup
        true_amps = jnp.array([5.0, 5.0, 5.0])
        noise = jnp.ones_like(wave) * 0.01
        residual = G @ true_amps

        # Tight prior on first line only
        prior_var = jnp.array([1e-10, 1e10, 1e10])
        _, a_hat, _ = marginalize_emission_lines(residual, noise, G, prior_variance=prior_var)
        assert abs(float(a_hat[0])) < 0.1, "First line should be suppressed"
        assert abs(float(a_hat[1]) - 5.0) < 0.5, "Second line should be recovered"
        assert abs(float(a_hat[2]) - 5.0) < 0.5, "Third line should be recovered"


# ── 6. JIT compilation works ──────────────────────────────────────


class TestJIT:
    """Test that all functions compile under jax.jit."""

    def test_design_matrix_jit(self, wave_grid):
        """build_eline_design_matrix should JIT-compile."""
        line_waves = jnp.array([5000.0, 6000.0])
        G = build_eline_design_matrix(wave_grid, line_waves, 1000.0, 0.0)
        assert G.shape == (len(wave_grid), 2)
        chex.assert_tree_all_finite(G)

    def test_marginalize_jit(self, simple_setup):
        """marginalize_emission_lines should JIT-compile."""
        wave, G, _, _, _ = simple_setup
        noise = jnp.ones_like(wave) * 0.1
        residual = jnp.ones_like(wave) * 0.5

        ln_l, a_hat, a_cov = marginalize_emission_lines(residual, noise, G)
        assert jnp.isfinite(ln_l)
        chex.assert_tree_all_finite(a_hat)
        chex.assert_tree_all_finite(a_cov)

    def test_predict_jit(self, simple_setup):
        """predict_with_marginalized_lines should JIT-compile."""
        wave, G, _, _, _ = simple_setup
        continuum = jnp.ones_like(wave)
        a_hat = jnp.array([3.0])
        model = predict_with_marginalized_lines(continuum, G, a_hat)
        chex.assert_equal_shape([model, wave])
        chex.assert_tree_all_finite(model)

    def test_full_pipeline_jit(self, wave_grid):
        """End-to-end pipeline should JIT-compile."""
        line_waves = jnp.array([4861.33, 6562.80])
        R = 1000.0
        z = 0.0

        @jax.jit
        def pipeline(data, continuum, noise):
            G = build_eline_design_matrix(wave_grid, line_waves, R, z)
            residual = data - continuum
            ln_l, a_hat, _ = marginalize_emission_lines(residual, noise, G)
            model = predict_with_marginalized_lines(continuum, G, a_hat)
            return ln_l, model

        noise = jnp.ones_like(wave_grid) * 0.1
        continuum = jnp.ones_like(wave_grid)
        data = continuum + 0.5
        ln_l, model = pipeline(data, continuum, noise)
        assert jnp.isfinite(ln_l)
        chex.assert_tree_all_finite(model)


# ── 7. Multiple lines don't interfere ─────────────────────────────


class TestLineIndependence:
    """Test that well-separated lines are independently recovered."""

    def test_adding_distant_line_doesnt_change_others(self, wave_grid):
        """Adding a line far from existing ones shouldn't affect recovery."""
        # Setup with H-alpha only
        line_single = jnp.array([6562.80])
        line_pair = jnp.array([6562.80, 4861.33])
        R = 2000.0
        z = 0.0

        G1 = build_eline_design_matrix(wave_grid, line_single, R, z)
        G2 = build_eline_design_matrix(wave_grid, line_pair, R, z)

        true_amp = 5.0
        noise = jnp.ones_like(wave_grid) * 0.01
        # Data has only H-alpha
        data_signal = true_amp * G1[:, 0]

        _, a1, _ = marginalize_emission_lines(data_signal, noise, G1)
        _, a2, _ = marginalize_emission_lines(data_signal, noise, G2)

        # H-alpha amplitude should be nearly identical
        assert_allclose(
            float(a1[0]),
            float(a2[0]),
            rtol=1e-3,
            err_msg="Adding distant line shouldn't change H-alpha recovery",
        )
        # H-beta should be near zero
        assert abs(float(a2[1])) < 0.01, "Absent line should have near-zero amplitude"

    def test_covariance_off_diagonal_small_for_separated_lines(self, wave_grid):
        """Off-diagonal covariance should be small for well-separated lines."""
        line_waves = jnp.array([4861.33, 6562.80])  # H-beta, H-alpha
        G = build_eline_design_matrix(wave_grid, line_waves, 2000.0, 0.0)
        noise = jnp.ones_like(wave_grid) * 0.1
        residual = jnp.zeros_like(wave_grid)

        _, _, a_cov = marginalize_emission_lines(residual, noise, G)
        # Normalize off-diagonal by geometric mean of diagonals
        corr_01 = float(a_cov[0, 1]) / jnp.sqrt(float(a_cov[0, 0]) * float(a_cov[1, 1]))
        assert abs(corr_01) < 0.1, (
            f"Correlation between well-separated lines should be small, got {corr_01:.4f}"
        )


# ── 8. Zero-amplitude lines don't affect likelihood ───────────────


class TestZeroLines:
    """Test behavior when no emission lines are present in the data."""

    def test_zero_amplitude_lines(self, simple_setup):
        """When data has no lines, recovered amplitudes should be near zero."""
        wave, G, _, _, _ = simple_setup
        noise = jnp.ones_like(wave) * 0.1
        # Pure continuum, no lines
        residual = jnp.zeros_like(wave)

        _, a_hat, _ = marginalize_emission_lines(residual, noise, G)
        assert abs(float(a_hat[0])) < 0.01, (
            f"No-line data should give near-zero amplitude, got {float(a_hat[0]):.4e}"
        )

    def test_likelihood_similar_without_lines(self, wave_grid):
        """Marginalized lnL should be similar to plain lnL when no lines."""
        line_waves = jnp.array([6562.80])
        G = build_eline_design_matrix(wave_grid, line_waves, 1000.0, 0.0)
        noise = jnp.ones_like(wave_grid) * 0.1

        # Data = pure noise, no lines, no continuum offset
        key = jax.random.PRNGKey(123)
        residual = jax.random.normal(key, wave_grid.shape) * 0.1

        ln_l_marg, _, _ = marginalize_emission_lines(residual, noise, G)
        ln_l_plain = -0.5 * jnp.sum(residual**2 / noise**2)

        # Marginalized should not be much worse than plain (lines don't help noise)
        # The difference should be small (just the prior/det correction)
        diff = abs(float(ln_l_marg) - float(ln_l_plain))
        # With 1 line and uninformative prior, correction is O(1)
        assert diff < 50.0, (
            f"Marginalized and plain lnL should be similar for noise-only data, diff={diff:.1f}"
        )

    def test_predict_equals_continuum_for_zero_amps(self, simple_setup):
        """predict_with_marginalized_lines should return continuum when a=0."""
        wave, G, _, _, _ = simple_setup
        continuum = jnp.linspace(1.0, 2.0, len(wave))
        a_zero = jnp.zeros(G.shape[1])
        model = predict_with_marginalized_lines(continuum, G, a_zero)
        assert_allclose(model, continuum, atol=1e-15)


# ── Edge cases ────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_more_lines_than_pixels(self):
        """Should handle n_lines > n_pix via prior regularization."""
        n_pix = 5
        n_lines = 10
        wave = jnp.linspace(4000.0, 7000.0, n_pix)
        line_waves = jnp.linspace(4000.0, 7000.0, n_lines)
        G = build_eline_design_matrix(wave, line_waves, 100.0, 0.0)

        noise = jnp.ones(n_pix) * 0.1
        residual = jnp.zeros(n_pix)

        # Should not crash — prior regularizes the underdetermined system
        ln_l, a_hat, _a_cov = marginalize_emission_lines(residual, noise, G)
        assert jnp.isfinite(ln_l), "Should handle n_lines > n_pix"
        chex.assert_tree_all_finite(a_hat)

    def test_posterior_covariance_symmetric(self, simple_setup):
        """Posterior covariance should be symmetric."""
        wave, G, _, _, _ = simple_setup
        noise = jnp.ones_like(wave) * 0.1
        residual = jnp.ones_like(wave) * 0.5
        _, _, a_cov = marginalize_emission_lines(residual, noise, G)
        assert_allclose(
            a_cov, a_cov.T, atol=1e-10, err_msg="Posterior covariance should be symmetric"
        )

    def test_posterior_covariance_positive_definite(self, multi_line_setup):
        """Posterior covariance should be positive definite."""
        wave, G, _, _, _ = multi_line_setup
        noise = jnp.ones_like(wave) * 0.1
        residual = jnp.ones_like(wave) * 0.5
        _, _, a_cov = marginalize_emission_lines(residual, noise, G)
        eigenvalues = jnp.linalg.eigvalsh(a_cov)
        assert jnp.all(eigenvalues > 0), (
            f"Posterior covariance should be PD, min eigenvalue={float(jnp.min(eigenvalues)):.2e}"
        )
