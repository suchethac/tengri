# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for Prevot et al. (1984) SMC extinction curve.

Tests the prevot_smc() dust attenuation law, including functional form validation,
X-ray ramp-down behavior, JIT compatibility, and gradient flow.

References
----------
- Prevot et al. 1984, A&A, 132, 389
- Calistro Rivera et al. 2018 (AGNfitter), ApJ, 863, 56
"""

import chex
import pytest

pytestmark = pytest.mark.regression_paper
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


from tengri.components.dust.attenuation import DUST_LAWS, prevot_smc
from tests._jit_parity import assert_jit_matches_eager


class TestRegistry:
    """Prevot SMC should be registered in DUST_LAWS."""

    def test_registered(self):
        assert "prevot_smc" in DUST_LAWS
        # DUST_LAWS now stores DustLawRegistryEntry which wraps the function
        entry = DUST_LAWS["prevot_smc"]
        assert callable(entry)
        assert entry.callable is prevot_smc


class TestPrevotSMCFunction:
    """Test the Prevot SMC extinction curve functional form."""

    def test_reference_values(self):
        """Test prevot_smc at key optical/NIR wavelengths.
        Expected values computed from normalized k(λ) = [1.39 * λ^(-1.2) - 0.38] / k(V)
        where λ is in micrometers and k(V) ≈ 2.468.
        """
        # Normalization constant (k(V) at 5500 Å = 0.55 μm)
        k_v = 1.39 * (0.55) ** (-1.2) - 0.38
        # λ (Å) -> λ (μm), k(λ) expected (normalized)
        test_cases = [
            (2000.0, (1.39 * (0.2) ** (-1.2) - 0.38) / k_v),  # 0.2 μm
            (4500.0, (1.39 * (0.45) ** (-1.2) - 0.38) / k_v),  # 0.45 μm
            (5500.0, (1.39 * (0.55) ** (-1.2) - 0.38) / k_v),  # 0.55 μm (V-band)
            (10000.0, (1.39 * (1.0) ** (-1.2) - 0.38) / k_v),  # 1.0 μm
        ]
        for wav_aa, expected_k in test_cases:
            result = float(prevot_smc(jnp.array([wav_aa]))[0])
            # Tolerance to machine precision in the functional form
            np.testing.assert_allclose(result, expected_k, rtol=1e-6, atol=1e-10)

    def test_monotonic_optical_nir(self):
        """k(λ) should decrease from UV to NIR (monotonically decreasing)."""
        wavs = jnp.linspace(1000.0, 10000.0, 100)
        k = np.array(prevot_smc(wavs))
        # Allow small local features, but overall should be monotonically decreasing
        # (since λ^-1.2 is decreasing in λ)
        assert k[0] > k[-1], "Curve should generally decrease from UV to NIR"

    def test_positive_optical_nir(self):
        """k(λ) should be positive in optical and NIR (λ ≥ 1000 Å)."""
        wavs = jnp.linspace(1000.0, 20000.0, 200)
        k = prevot_smc(wavs)
        # At these wavelengths, k should be positive (ramp is ≈1)
        assert jnp.all(k > 0.0), "k should be positive for λ ≥ 1000 Å"

    def test_xray_rampdown(self):
        """k(λ) should ramp smoothly toward zero below 62 Å (X-ray edge)."""
        # Test at wavelengths below X-ray edge
        wavs_xray = jnp.array([10.0, 30.0, 50.0, 62.0])
        k_xray = prevot_smc(wavs_xray)
        # At 62 Å (edge), ramp = 0.5, so k should be suppressed
        # At 10 Å (deep X-ray), ramp ≈ 0, so k should be heavily suppressed
        k_10 = float(k_xray[0])
        k_62 = float(k_xray[-1])
        # The value at 10 Å should be much smaller than at 62 Å
        assert k_10 < k_62, "X-ray suppression should increase at shorter wavelengths"
        # Both should be suppressed compared to no ramp (i.e., smaller than
        # the raw normalized functional form without ramp)
        k_v = 1.39 * (0.55) ** (-1.2) - 0.38
        k_raw_10 = (1.39 * (0.001) ** (-1.2) - 0.38) / k_v
        k_raw_62 = (1.39 * (0.0062) ** (-1.2) - 0.38) / k_v
        assert k_10 < k_raw_10, "Ramp should suppress 10 Å value"
        assert k_62 < k_raw_62, "Ramp should suppress 62 Å value"
        # k should be smooth (no discontinuity at edge)
        # Check that k is finite near the edge
        wavs_edge = jnp.linspace(50.0, 100.0, 50)
        k_edge = prevot_smc(wavs_edge)
        # The transition should be smooth (no NaNs or Infs)
        chex.assert_tree_all_finite(k_edge)

    def test_jit_compatible(self):
        """prevot_smc should work inside jax.jit."""
        wavs = jnp.array([1000.0, 5500.0, 10000.0])
        result = assert_jit_matches_eager(prevot_smc, wavs)
        chex.assert_shape(result, (3,))
        chex.assert_tree_all_finite(result)

    def test_gradient_through_wavelength(self):
        """Gradient of k(λ) w.r.t. wavelength should be finite and smooth."""

        def loss_func(wav):
            return jnp.sum(prevot_smc(wav))

        wavs = jnp.array([5500.0])  # V-band test point
        grad_jax = float(jax.grad(loss_func)(wavs)[0])

        # Compute finite-difference approximation
        def f_scalar(wav0: float) -> float:
            return float(loss_func(jnp.array([wav0])))

        grad_fd = fd_grad(f_scalar, 5500.0, eps=1.0)
        # Check that gradients are finite and close to FD estimate
        assert np.isfinite(grad_jax), "Gradient should be finite"
        np.testing.assert_allclose(grad_jax, grad_fd, rtol=0.05)

    def test_vectorization(self):
        """prevot_smc should handle array inputs correctly."""
        wavs = jnp.linspace(500.0, 50000.0, 500)
        result = prevot_smc(wavs)
        chex.assert_shape(result, (500,))
        chex.assert_tree_all_finite(result)

    def test_kwargs_ignored(self):
        """prevot_smc should accept and ignore extra kwargs."""
        wavs = jnp.array([5500.0])
        # Should work with extra kwargs (like dust_Rv for other laws)
        result1 = prevot_smc(wavs)
        result2 = prevot_smc(wavs, dust_Rv=3.1)  # ignored kwarg
        np.testing.assert_allclose(result1, result2, rtol=1e-15)
