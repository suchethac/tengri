# SPDX-License-Identifier: BSD-3-Clause
"""Tests for GP generation from PSD (T2: PSD recovery, T3: ACF recovery)."""

import chex
import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from tengri.components.stellar.sfh.gp_sfh import (
    compute_sqrt_power_drw,
    generate_gp_batch,
    gp_from_xi,
)
from tengri.components.stellar.sfh.psd_models import psd_to_sqrt_power
from tests._bounds import assert_non_negative
from tests._jit_parity import assert_jit_matches_eager

N_GRID = 256


class TestGPFromXi:
    """Tests for deterministic GP generation from latent vector."""

    @pytest.fixture
    def sqrt_power(self):
        d = (10.14 - 6.0) / (N_GRID - 1)
        return compute_sqrt_power_drw(N_GRID, d, 1.0, 50e6)

    def test_output_shape(self, sqrt_power):
        """GP realization has shape (n_points,)."""
        xi = jnp.zeros(N_GRID)
        x = gp_from_xi(xi, sqrt_power, N_GRID)
        chex.assert_shape(x, (N_GRID,))

    def test_zero_xi_gives_zero_gp(self, sqrt_power):
        """xi = 0 produces x(t) = 0 everywhere."""
        xi = jnp.zeros(N_GRID)
        x = gp_from_xi(xi, sqrt_power, N_GRID)
        assert_allclose(x, 0.0, atol=1e-15)

    def test_deterministic(self, sqrt_power):
        """Same xi always gives same GP realization."""
        xi = jax.random.normal(jax.random.PRNGKey(0), shape=(N_GRID,))
        x1 = gp_from_xi(xi, sqrt_power, N_GRID)
        x2 = gp_from_xi(xi, sqrt_power, N_GRID)
        assert_allclose(x1, x2, atol=1e-15)

    def test_is_jittable(self, sqrt_power):
        """gp_from_xi can be JIT-compiled."""
        xi = jax.random.normal(jax.random.PRNGKey(0), shape=(N_GRID,))
        x = assert_jit_matches_eager(lambda xi: gp_from_xi(xi, sqrt_power, N_GRID), xi)
        chex.assert_shape(x, (N_GRID,))

    def test_has_gradients(self, sqrt_power):
        """Gradients w.r.t. xi exist and are finite."""
        grad_fn = jax.grad(lambda xi: jnp.sum(gp_from_xi(xi, sqrt_power, N_GRID)))
        xi = jax.random.normal(jax.random.PRNGKey(0), shape=(N_GRID,))
        g = grad_fn(xi)
        chex.assert_tree_all_finite(g)


pytestmark = pytest.mark.bounds


class TestGPStatisticalProperties:
    """Test that GP realizations have correct statistical properties.

    Uses a simple known PSD (flat / white noise) to verify the FFT machinery
    independently from the DRW Jacobian correction.
    """

    def test_white_noise_variance_scales(self):
        """For flat PSD, variance scales linearly with PSD level.

        We don't test the exact normalization constant (which depends on
        FFT conventions and Hermitian symmetry counting), just that the
        scaling is correct.
        """
        n = N_GRID
        d = 0.016
        n_freq = n // 2 + 1
        key = jax.random.PRNGKey(42)
        n_real = 3000

        vars_by_level = []
        for level in [0.5, 1.0, 2.0]:
            p_k = jnp.ones(n_freq) * level
            sqrt_power = psd_to_sqrt_power(p_k, d)
            batch = generate_gp_batch(key, sqrt_power, n, n_real)
            vars_by_level.append(float(jnp.var(batch)))

        # Variance should scale linearly with PSD level
        ratio_1 = vars_by_level[1] / vars_by_level[0]
        ratio_2 = vars_by_level[2] / vars_by_level[0]
        assert_allclose(ratio_1, 2.0, rtol=0.15)
        assert_allclose(ratio_2, 4.0, rtol=0.15)

    def test_different_psd_levels(self):
        """GP variance scales linearly with PSD level."""
        n = N_GRID
        d = 0.016
        key = jax.random.PRNGKey(99)
        n_real = 3000
        n_freq = n // 2 + 1

        vars_at_levels = []
        for level in [0.5, 1.0, 2.0]:
            p_k = jnp.ones(n_freq) * level
            sqrt_power = psd_to_sqrt_power(p_k, d)
            batch = generate_gp_batch(key, sqrt_power, n, n_real)
            vars_at_levels.append(float(jnp.var(batch)))

        # Variance should scale linearly with PSD level
        ratio_1 = vars_at_levels[1] / vars_at_levels[0]
        ratio_2 = vars_at_levels[2] / vars_at_levels[0]
        assert_allclose(ratio_1, 2.0, rtol=0.15)
        assert_allclose(ratio_2, 4.0, rtol=0.15)

    def test_gp_mean_near_zero(self):
        """GP realizations have approximately zero mean."""
        n = N_GRID
        d = 0.016
        n_freq = n // 2 + 1
        # Use DRW-like PSD with reasonable amplitudes
        p_k = jnp.ones(n_freq) * 0.01
        sqrt_power = psd_to_sqrt_power(p_k, d)

        key = jax.random.PRNGKey(77)
        n_real = 3000
        batch = generate_gp_batch(key, sqrt_power, n, n_real)

        # Mean over all realizations and grid points should be ~0
        global_mean = float(jnp.mean(batch))
        assert abs(global_mean) < 0.1, f"GP mean = {global_mean}, expected ~0"


class TestGPPSDRecoverySimple:
    """T2: Verify empirical PSD from GP realizations matches input.

    Tests with a simple power-law PSD in the grid's native frequency
    domain (no physical-unit conversion needed).
    """

    def test_psd_shape_recovery(self):
        """Empirical periodogram SHAPE matches input PSD shape.

        The absolute normalization depends on FFT conventions, but the
        relative shape (ratio of power at different frequencies) should
        match the input PSD.
        """
        n = N_GRID
        d = 0.016
        n_freq = n // 2 + 1

        # Create a Lorentzian PSD in grid-native frequency
        freqs = jnp.fft.rfftfreq(n, d=d)
        omega = 2.0 * jnp.pi * freqs
        k0 = 10.0
        input_psd = 1.0 / (1.0 + (omega / k0) ** 2)
        sqrt_power = psd_to_sqrt_power(input_psd, d)

        key = jax.random.PRNGKey(42)
        n_real = 3000
        batch = generate_gp_batch(key, sqrt_power, n, n_real)

        # Compute empirical periodogram
        fft_coeffs = jnp.fft.rfft(batch, axis=1)
        empirical_psd = jnp.mean(jnp.abs(fft_coeffs) ** 2, axis=0)

        # Compare SHAPES by normalizing both to their value at k=5
        idx_ref = 5
        emp_norm = empirical_psd[1:30] / empirical_psd[idx_ref]
        inp_norm = input_psd[1:30] / input_psd[idx_ref]

        # Shape ratio should be ~1 everywhere
        shape_ratio = emp_norm / inp_norm
        assert_allclose(
            float(jnp.median(shape_ratio)), 1.0, rtol=0.2, err_msg="PSD shape recovery failed"
        )


class TestDRWJacobianCorrection:
    """Test the DRW Jacobian correction for log-age grids."""

    def test_sqrt_power_finite(self):
        """Amplitude operator is finite for physical DRW parameters."""
        d = (10.14 - 6.0) / (N_GRID - 1)
        sqrt_power = compute_sqrt_power_drw(N_GRID, d, 1.0, 50e6)
        chex.assert_tree_all_finite(sqrt_power)
        assert_non_negative(sqrt_power, name="sqrt_power")

    def test_sqrt_power_reasonable_gp_variance(self):
        """GP variance from Jacobian-corrected DRW is finite and positive."""
        d = (10.14 - 6.0) / (N_GRID - 1)
        key = jax.random.PRNGKey(42)

        for psd_sigma in [0.5, 1.0, 2.0]:
            sqrt_power = compute_sqrt_power_drw(N_GRID, d, psd_sigma, 50e6)
            batch = generate_gp_batch(key, sqrt_power, N_GRID, 1000)
            var = float(jnp.var(batch))

            # Variance should be finite, positive, and bounded
            assert var > 0, f"GP variance = {var} for psd_sigma={psd_sigma}"
            assert var < 1e6, f"GP variance = {var} too large"

    def test_larger_sigma_larger_variance(self):
        """GP variance increases with psd_sigma."""
        d = (10.14 - 6.0) / (N_GRID - 1)
        key = jax.random.PRNGKey(42)
        n_real = 2000

        vars = []
        for psd_sigma in [0.5, 1.0, 2.0]:
            sqrt_power = compute_sqrt_power_drw(N_GRID, d, psd_sigma, 50e6)
            batch = generate_gp_batch(key, sqrt_power, N_GRID, n_real)
            vars.append(float(jnp.var(batch)))

        # Variance should increase monotonically with psd_sigma
        assert vars[0] < vars[1] < vars[2]
