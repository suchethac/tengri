"""Tests for GP covariance kernels in spectral noise modeling."""

import jax
import jax.numpy as jnp
import pytest

from tengri.observation.noise import (
    exp_squared_kernel,
    gp_noise_covariance,
    matern32_kernel,
)

jax.config.update("jax_enable_x64", True)


class TestExpSquaredKernel:
    """Tests for exp_squared_kernel function."""

    def test_shape_square(self):
        """Test square kernel matrix shape when x2 is None."""
        x = jnp.linspace(4000.0, 8000.0, 20)
        K = exp_squared_kernel(x, amplitude=1.0, length_scale=500.0)
        assert K.shape == (20, 20)

    def test_shape_cross(self):
        """Test cross-covariance kernel matrix shape."""
        x = jnp.linspace(4000.0, 8000.0, 20)
        x2 = jnp.linspace(4000.0, 8000.0, 15)
        K = exp_squared_kernel(x, amplitude=1.0, length_scale=500.0, x2=x2)
        assert K.shape == (20, 15)

    def test_symmetric(self):
        """Test that square kernel matrix is symmetric."""
        x = jnp.linspace(4000.0, 8000.0, 20)
        K = exp_squared_kernel(x, amplitude=1.0, length_scale=500.0)
        assert jnp.allclose(K, K.T, atol=1e-10)

    def test_diagonal_equals_amplitude_squared(self):
        """Test that diagonal entries equal amplitude^2."""
        x = jnp.linspace(4000.0, 8000.0, 20)
        K = exp_squared_kernel(x, amplitude=2.0, length_scale=500.0)
        assert jnp.allclose(jnp.diag(K), 4.0, atol=1e-10)

    def test_positive_semidefinite(self):
        """Test that kernel matrix is positive semi-definite."""
        x = jnp.linspace(4000.0, 8000.0, 10)
        K = exp_squared_kernel(x, amplitude=1.0, length_scale=500.0)
        eigenvalues = jnp.linalg.eigvalsh(K)
        assert jnp.all(eigenvalues >= -1e-10)

    def test_jit_compatible(self):
        """Test that kernel function is JIT-compatible."""
        x = jnp.linspace(4000.0, 8000.0, 20)
        fn = jax.jit(exp_squared_kernel)
        K = fn(x, amplitude=1.0, length_scale=500.0)
        assert K.shape == (20, 20)

    def test_amplitude_scaling(self):
        """Test that amplitude scales kernel correctly."""
        x = jnp.linspace(4000.0, 8000.0, 20)
        K1 = exp_squared_kernel(x, amplitude=1.0, length_scale=500.0)
        K2 = exp_squared_kernel(x, amplitude=2.0, length_scale=500.0)
        assert jnp.allclose(K2, 4.0 * K1, atol=1e-10)

    def test_length_scale_effect(self):
        """Test that shorter length scales give faster decay."""
        x = jnp.array([4000.0, 5000.0])
        K_short = exp_squared_kernel(x, amplitude=1.0, length_scale=100.0)
        K_long = exp_squared_kernel(x, amplitude=1.0, length_scale=2000.0)
        # Off-diagonal entry should be smaller with shorter length scale
        assert K_short[0, 1] < K_long[0, 1]

    def test_differentiable_wrt_amplitude(self):
        """Test that kernel is differentiable w.r.t. amplitude."""
        x = jnp.linspace(4000.0, 8000.0, 10)

        def f(amp):
            K = exp_squared_kernel(x, amplitude=amp, length_scale=500.0)
            return jnp.sum(K)

        grad_f = jax.grad(f)
        grad_val = grad_f(1.0)
        assert isinstance(grad_val, (float, jnp.ndarray))
        assert not jnp.isnan(grad_val)

    def test_differentiable_wrt_length_scale(self):
        """Test that kernel is differentiable w.r.t. length scale."""
        x = jnp.linspace(4000.0, 8000.0, 10)

        def f(ls):
            K = exp_squared_kernel(x, amplitude=1.0, length_scale=ls)
            return jnp.sum(K)

        grad_f = jax.grad(f)
        grad_val = grad_f(500.0)
        assert isinstance(grad_val, (float, jnp.ndarray))
        assert not jnp.isnan(grad_val)


class TestMatern32Kernel:
    """Tests for matern32_kernel function."""

    def test_shape(self):
        """Test kernel matrix shape."""
        x = jnp.linspace(4000.0, 8000.0, 20)
        K = matern32_kernel(x, amplitude=1.0, length_scale=500.0)
        assert K.shape == (20, 20)

    def test_shape_cross(self):
        """Test cross-covariance kernel matrix shape."""
        x = jnp.linspace(4000.0, 8000.0, 20)
        x2 = jnp.linspace(4000.0, 8000.0, 15)
        K = matern32_kernel(x, amplitude=1.0, length_scale=500.0, x2=x2)
        assert K.shape == (20, 15)

    def test_symmetric(self):
        """Test that square kernel matrix is symmetric."""
        x = jnp.linspace(4000.0, 8000.0, 20)
        K = matern32_kernel(x, amplitude=1.0, length_scale=500.0)
        assert jnp.allclose(K, K.T, atol=1e-10)

    def test_diagonal_equals_amplitude_squared(self):
        """Test that diagonal entries equal amplitude^2."""
        x = jnp.linspace(4000.0, 8000.0, 20)
        K = matern32_kernel(x, amplitude=3.0, length_scale=500.0)
        assert jnp.allclose(jnp.diag(K), 9.0, atol=1e-10)

    def test_positive_semidefinite(self):
        """Test that kernel matrix is positive semi-definite."""
        x = jnp.linspace(4000.0, 8000.0, 10)
        K = matern32_kernel(x, amplitude=1.0, length_scale=500.0)
        eigenvalues = jnp.linalg.eigvalsh(K)
        assert jnp.all(eigenvalues >= -1e-10)

    def test_jit_compatible(self):
        """Test that kernel function is JIT-compatible."""
        x = jnp.linspace(4000.0, 8000.0, 20)
        fn = jax.jit(matern32_kernel)
        K = fn(x, amplitude=1.0, length_scale=500.0)
        assert K.shape == (20, 20)

    def test_shorter_length_scale_falls_off_faster(self):
        """Test that shorter length scales give faster decay."""
        x = jnp.array([4000.0, 5000.0])
        K_short = matern32_kernel(x, amplitude=1.0, length_scale=100.0)
        K_long = matern32_kernel(x, amplitude=1.0, length_scale=2000.0)
        # Off-diagonal entry should be smaller with shorter length scale
        assert K_short[0, 1] < K_long[0, 1]

    def test_amplitude_scaling(self):
        """Test that amplitude scales kernel correctly."""
        x = jnp.linspace(4000.0, 8000.0, 10)
        K1 = matern32_kernel(x, amplitude=1.0, length_scale=500.0)
        K2 = matern32_kernel(x, amplitude=2.0, length_scale=500.0)
        assert jnp.allclose(K2, 4.0 * K1, atol=1e-10)

    def test_differentiable_wrt_amplitude(self):
        """Test that kernel is differentiable w.r.t. amplitude."""
        x = jnp.linspace(4000.0, 8000.0, 10)

        def f(amp):
            K = matern32_kernel(x, amplitude=amp, length_scale=500.0)
            return jnp.sum(K)

        grad_f = jax.grad(f)
        grad_val = grad_f(1.0)
        assert isinstance(grad_val, (float, jnp.ndarray))
        assert not jnp.isnan(grad_val)

    def test_differentiable_wrt_length_scale(self):
        """Test that kernel is differentiable w.r.t. length scale."""
        x = jnp.linspace(4000.0, 8000.0, 10)

        def f(ls):
            K = matern32_kernel(x, amplitude=1.0, length_scale=ls)
            return jnp.sum(K)

        grad_f = jax.grad(f)
        grad_val = grad_f(500.0)
        assert isinstance(grad_val, (float, jnp.ndarray))
        assert not jnp.isnan(grad_val)


class TestGPNoiseCovariance:
    """Tests for gp_noise_covariance function."""

    def test_shape(self):
        """Test covariance matrix shape."""
        wave = jnp.linspace(4000.0, 8000.0, 30)
        noise = jnp.ones(30) * 0.1
        N = gp_noise_covariance(wave, noise, gp_amplitude=0.5, gp_length_scale=300.0)
        assert N.shape == (30, 30)

    def test_diagonal_has_obs_noise(self):
        """Test that diagonal is dominated by observation noise when GP is small."""
        wave = jnp.linspace(4000.0, 8000.0, 10)
        sigma = 0.1
        noise = jnp.ones(10) * sigma
        N = gp_noise_covariance(wave, noise, gp_amplitude=0.0, gp_length_scale=500.0)
        assert jnp.allclose(jnp.diag(N), sigma**2, atol=1e-10)

    def test_matern32_option(self):
        """Test using Matérn 3/2 kernel."""
        wave = jnp.linspace(4000.0, 8000.0, 20)
        noise = jnp.ones(20) * 0.1
        N = gp_noise_covariance(
            wave,
            noise,
            gp_amplitude=0.5,
            gp_length_scale=300.0,
            kernel="matern32",
        )
        assert N.shape == (20, 20)

    def test_exp_squared_option(self):
        """Test using squared exponential kernel (default)."""
        wave = jnp.linspace(4000.0, 8000.0, 20)
        noise = jnp.ones(20) * 0.1
        N = gp_noise_covariance(
            wave,
            noise,
            gp_amplitude=0.5,
            gp_length_scale=300.0,
            kernel="exp_squared",
        )
        assert N.shape == (20, 20)

    def test_invalid_kernel_raises(self):
        """Test that invalid kernel name raises ValueError."""
        wave = jnp.linspace(4000.0, 8000.0, 10)
        noise = jnp.ones(10) * 0.1
        with pytest.raises(ValueError, match="Unknown kernel"):
            gp_noise_covariance(
                wave, noise, gp_amplitude=0.5, gp_length_scale=300.0, kernel="invalid"
            )

    def test_symmetric(self):
        """Test that covariance matrix is symmetric."""
        wave = jnp.linspace(4000.0, 8000.0, 15)
        noise = jnp.ones(15) * 0.1
        N = gp_noise_covariance(wave, noise, gp_amplitude=0.5, gp_length_scale=300.0)
        assert jnp.allclose(N, N.T, atol=1e-10)

    def test_positive_semidefinite(self):
        """Test that covariance matrix is positive semi-definite."""
        wave = jnp.linspace(4000.0, 8000.0, 10)
        noise = jnp.ones(10) * 0.1
        N = gp_noise_covariance(wave, noise, gp_amplitude=0.5, gp_length_scale=300.0)
        eigenvalues = jnp.linalg.eigvalsh(N)
        assert jnp.all(eigenvalues >= -1e-10)

    def test_diagonal_is_variance_sum(self):
        """Test that diagonal is sum of obs noise and GP variance."""
        wave = jnp.linspace(4000.0, 8000.0, 10)
        sigma = 0.1
        noise = jnp.ones(10) * sigma
        amp = 0.5
        N = gp_noise_covariance(wave, noise, gp_amplitude=amp, gp_length_scale=300.0)
        # Diagonal = sigma^2 + amp^2 (from GP kernel diagonal)
        expected_diag = sigma**2 + amp**2
        assert jnp.allclose(jnp.diag(N), expected_diag, atol=1e-10)

    def test_variable_noise_shape(self):
        """Test with variable noise levels."""
        wave = jnp.linspace(4000.0, 8000.0, 20)
        noise = jnp.linspace(0.05, 0.2, 20)
        N = gp_noise_covariance(wave, noise, gp_amplitude=0.5, gp_length_scale=300.0)
        assert N.shape == (20, 20)
        assert jnp.allclose(jnp.diag(N), noise**2 + 0.5**2, atol=1e-10)
