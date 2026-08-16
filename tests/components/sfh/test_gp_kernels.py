# SPDX-License-Identifier: BSD-3-Clause
"""Tests for GP covariance kernels in spectral noise modeling."""

import chex
import jax
import jax.numpy as jnp
import pytest

from tengri.observation.noise import (
    exp_squared_kernel,
    gp_noise_covariance,
    matern32_kernel,
)

pytestmark = pytest.mark.bounds


class TestExpSquaredKernel:
    """Tests for exp_squared_kernel function.

    Tests anchor to the documented formula:
        K(x, x') = σ² exp(-(x - x')²/(2ℓ²))
    """

    def test_symmetric(self):
        """Test that square kernel matrix is symmetric."""
        x = jnp.linspace(4000.0, 8000.0, 20)
        K = exp_squared_kernel(x, amplitude=1.0, length_scale=500.0)
        assert jnp.allclose(K, K.T, atol=1e-10)

    def test_diagonal_equals_amplitude_squared(self):
        """Test that diagonal entries equal amplitude^2.

        At x=x', the squared exponential formula gives K(x,x) = σ².
        """
        x = jnp.linspace(4000.0, 8000.0, 20)
        K = exp_squared_kernel(x, amplitude=2.0, length_scale=500.0)
        assert jnp.allclose(jnp.diag(K), 4.0, atol=1e-10)

    def test_positive_semidefinite(self):
        """Test that kernel matrix is positive semi-definite.

        All eigenvalues of the covariance matrix must be non-negative.
        """
        x = jnp.linspace(4000.0, 8000.0, 10)
        K = exp_squared_kernel(x, amplitude=1.0, length_scale=500.0)
        eigenvalues = jnp.linalg.eigvalsh(K)
        assert jnp.all(eigenvalues >= -1e-10)

    def test_amplitude_scaling(self):
        """Test that amplitude scales kernel correctly.

        K(σ₂, ...) = (σ₂/σ₁)² · K(σ₁, ...).
        """
        x = jnp.linspace(4000.0, 8000.0, 20)
        K1 = exp_squared_kernel(x, amplitude=1.0, length_scale=500.0)
        K2 = exp_squared_kernel(x, amplitude=2.0, length_scale=500.0)
        assert jnp.allclose(K2, 4.0 * K1, atol=1e-10)

    def test_exponential_decay_with_distance(self):
        """Test that covariance decays exponentially with distance.

        Verifies K(Δx) = σ² exp(-Δx²/(2ℓ²)) at specific distances.
        """
        amp = 1.5
        ell = 500.0
        # Two points at specific distances
        x1 = jnp.array([4000.0, 5000.0])
        K = exp_squared_kernel(x1, amplitude=amp, length_scale=ell)

        # Off-diagonal distance is 1000
        delta = 1000.0
        expected = amp**2 * jnp.exp(-0.5 * (delta / ell) ** 2)
        assert jnp.allclose(K[0, 1], expected, rtol=1e-6)

    def test_jit_parity(self):
        """Test that jitted and eager evaluation match.

        Verifies JIT compilation doesn't alter semantics.
        """
        x = jnp.linspace(4000.0, 8000.0, 10)
        K_eager = exp_squared_kernel(x, amplitude=1.0, length_scale=500.0)
        K_jit = jax.jit(exp_squared_kernel)(x, amplitude=1.0, length_scale=500.0)
        chex.assert_trees_all_close(K_eager, K_jit, rtol=1e-6)


class TestMatern32Kernel:
    """Tests for matern32_kernel function.

    Tests anchor to the documented formula:
        K(r) = σ² (1 + √3·r/ℓ) exp(-√3·r/ℓ), where r = |x - x'|
    """

    def test_symmetric(self):
        """Test that square kernel matrix is symmetric."""
        x = jnp.linspace(4000.0, 8000.0, 20)
        K = matern32_kernel(x, amplitude=1.0, length_scale=500.0)
        assert jnp.allclose(K, K.T, atol=1e-10)

    def test_diagonal_equals_amplitude_squared(self):
        """Test that diagonal entries equal amplitude^2.

        At r=0, the Matérn 3/2 formula gives K(0) = σ² (1 + 0) exp(0) = σ².
        """
        x = jnp.linspace(4000.0, 8000.0, 20)
        K = matern32_kernel(x, amplitude=3.0, length_scale=500.0)
        assert jnp.allclose(jnp.diag(K), 9.0, atol=1e-10)

    def test_positive_semidefinite(self):
        """Test that kernel matrix is positive semi-definite.

        All eigenvalues of the covariance matrix must be non-negative.
        """
        x = jnp.linspace(4000.0, 8000.0, 10)
        K = matern32_kernel(x, amplitude=1.0, length_scale=500.0)
        eigenvalues = jnp.linalg.eigvalsh(K)
        assert jnp.all(eigenvalues >= -1e-10)

    def test_amplitude_scaling(self):
        """Test that amplitude scales kernel correctly.

        K(σ₂, ...) = (σ₂/σ₁)² · K(σ₁, ...).
        """
        x = jnp.linspace(4000.0, 8000.0, 10)
        K1 = matern32_kernel(x, amplitude=1.0, length_scale=500.0)
        K2 = matern32_kernel(x, amplitude=2.0, length_scale=500.0)
        assert jnp.allclose(K2, 4.0 * K1, atol=1e-10)

    def test_matern32_formula_at_distance(self):
        """Test Matérn 3/2 formula at specific distance.

        Verifies K(r) = σ² (1 + √3·r/ℓ) exp(-√3·r/ℓ).
        """
        amp = 2.0
        ell = 500.0
        # Two points separated by distance 1000
        x = jnp.array([4000.0, 5000.0])
        K = matern32_kernel(x, amplitude=amp, length_scale=ell)

        r = 1000.0
        sqrt3 = jnp.sqrt(3.0)
        arg = sqrt3 * r / ell
        expected = amp**2 * (1.0 + arg) * jnp.exp(-arg)
        assert jnp.allclose(K[0, 1], expected, rtol=1e-6)

    def test_jit_parity(self):
        """Test that jitted and eager evaluation match.

        Verifies JIT compilation doesn't alter semantics.
        """
        x = jnp.linspace(4000.0, 8000.0, 10)
        K_eager = matern32_kernel(x, amplitude=1.0, length_scale=500.0)
        K_jit = jax.jit(matern32_kernel)(x, amplitude=1.0, length_scale=500.0)
        chex.assert_trees_all_close(K_eager, K_jit, rtol=1e-6)


class TestGPNoiseCovariance:
    """Tests for gp_noise_covariance function.

    Tests anchor to: N = diag(σ²_obs) + K_gp, where K_gp is the chosen
    GP kernel (exp_squared or matern32).
    """

    def test_diagonal_has_obs_noise_when_gp_zero(self):
        """Test that diagonal reduces to observation noise when gp_amplitude=0.

        When gp_amplitude=0, the GP kernel is zero, so N = diag(σ²_obs).
        """
        wave = jnp.linspace(4000.0, 8000.0, 10)
        sigma = 0.1
        noise = jnp.ones(10) * sigma
        N = gp_noise_covariance(wave, noise, gp_amplitude=0.0, gp_length_scale=500.0)
        assert jnp.allclose(jnp.diag(N), sigma**2, atol=1e-10)

    def test_invalid_kernel_raises(self):
        """Test that invalid kernel name raises ValueError."""
        wave = jnp.linspace(4000.0, 8000.0, 10)
        noise = jnp.ones(10) * 0.1
        with pytest.raises(ValueError, match="Unknown kernel"):
            gp_noise_covariance(
                wave, noise, gp_amplitude=0.5, gp_length_scale=300.0, kernel="invalid"
            )

    def test_symmetric(self):
        """Test that covariance matrix is symmetric.

        Sum of symmetric matrices is symmetric.
        """
        wave = jnp.linspace(4000.0, 8000.0, 15)
        noise = jnp.ones(15) * 0.1
        N = gp_noise_covariance(wave, noise, gp_amplitude=0.5, gp_length_scale=300.0)
        assert jnp.allclose(N, N.T, atol=1e-10)

    def test_positive_semidefinite(self):
        """Test that covariance matrix is positive semi-definite.

        Sum of positive semi-definite matrices is positive semi-definite.
        """
        wave = jnp.linspace(4000.0, 8000.0, 10)
        noise = jnp.ones(10) * 0.1
        N = gp_noise_covariance(wave, noise, gp_amplitude=0.5, gp_length_scale=300.0)
        eigenvalues = jnp.linalg.eigvalsh(N)
        assert jnp.all(eigenvalues >= -1e-10)

    def test_diagonal_is_variance_sum(self):
        """Test that diagonal is sum of obs noise and GP variance.

        N_ii = σ²_obs,i + K_gp(x_i, x_i) = σ²_obs,i + amp².
        """
        wave = jnp.linspace(4000.0, 8000.0, 10)
        sigma = 0.1
        noise = jnp.ones(10) * sigma
        amp = 0.5
        N = gp_noise_covariance(wave, noise, gp_amplitude=amp, gp_length_scale=300.0)
        # Diagonal = sigma^2 + amp^2 (from GP kernel diagonal)
        expected_diag = sigma**2 + amp**2
        assert jnp.allclose(jnp.diag(N), expected_diag, atol=1e-10)

    def test_matern32_diagonal_equals_exp_squared_sum(self):
        """Test with variable noise levels and both kernels.

        Verifies that the diagonal formula holds for both kernel types.
        """
        wave = jnp.linspace(4000.0, 8000.0, 20)
        noise = jnp.linspace(0.05, 0.2, 20)
        amp = 0.5

        # Test with both kernels
        for kernel in ["exp_squared", "matern32"]:
            N = gp_noise_covariance(
                wave, noise, gp_amplitude=amp, gp_length_scale=300.0, kernel=kernel
            )
            # Diagonal should be noise^2 + amp^2 for any valid kernel
            assert jnp.allclose(jnp.diag(N), noise**2 + amp**2, atol=1e-10)
