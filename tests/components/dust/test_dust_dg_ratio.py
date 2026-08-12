# SPDX-License-Identifier: BSD-3-Clause
"""Tests for metallicity-dependent dust-to-gas ratio (Rémy-Ruyer+2014).

Reference: Rémy-Ruyer et al. 2014, A&A, 563, A31, Table 1.
"""

import pytest

pytestmark = pytest.mark.bounds
import jax
import jax.numpy as jnp
import numpy as np

from tests._jit_parity import assert_jit_matches_eager

jax.config.update("jax_enable_x64", True)


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


class TestDustToGasScaling:
    """Verify broken power-law D/G scaling."""

    def test_solar_returns_one(self):
        """At Z_sun (logzsol=0), scaling = 1.0."""
        from tengri.components.dust.attenuation import dust_to_gas_scaling_remy_ruyer

        result = dust_to_gas_scaling_remy_ruyer(0.0)
        assert jnp.isclose(result, 1.0, atol=1e-6)

    def test_supersolar(self):
        """At 2 Z_sun (logzsol=0.3), scaling > 1.0."""
        from tengri.components.dust.attenuation import dust_to_gas_scaling_remy_ruyer

        result = dust_to_gas_scaling_remy_ruyer(0.3)
        assert float(result) > 1.0

    def test_subsolar_lower(self):
        """At 0.01 Z_sun (logzsol=-2), scaling << 1.0."""
        from tengri.components.dust.attenuation import dust_to_gas_scaling_remy_ruyer

        result = dust_to_gas_scaling_remy_ruyer(-2.0)
        assert float(result) < 0.01

    def test_steep_below_threshold(self):
        """Below 0.1 Z_sun, scaling drops faster than linear.
        Rémy-Ruyer+2014: slope steepens from ~1.0 to ~2.0-3.0 below
        12 + log(O/H) ≈ 7.96 (≈ 0.1 Z_sun).
        """
        from tengri.components.dust.attenuation import dust_to_gas_scaling_remy_ruyer

        # At 0.01 Z_sun: linear would give 0.01, quadratic gives 0.001
        result_01 = dust_to_gas_scaling_remy_ruyer(-2.0)
        linear_01 = 0.01  # 10^(-2)
        assert float(result_01) < linear_01

    def test_continuity_at_break(self):
        """No discontinuity at 0.1 Z_sun breakpoint."""
        from tengri.components.dust.attenuation import dust_to_gas_scaling_remy_ruyer

        eps = 1e-4
        above = dust_to_gas_scaling_remy_ruyer(jnp.log10(0.1 + eps))
        below = dust_to_gas_scaling_remy_ruyer(jnp.log10(0.1 - eps))
        assert jnp.isclose(above, below, atol=1e-2)

    def test_monotonic(self):
        """Higher metallicity → higher D/G ratio."""
        from tengri.components.dust.attenuation import dust_to_gas_scaling_remy_ruyer

        logzsols = [-3.0, -2.0, -1.0, 0.0, 0.3]
        scalings = [float(dust_to_gas_scaling_remy_ruyer(z)) for z in logzsols]
        for i in range(len(scalings) - 1):
            assert scalings[i] < scalings[i + 1]

    def test_jit_compatible(self):
        """Function can be JIT-compiled."""
        from tengri.components.dust.attenuation import dust_to_gas_scaling_remy_ruyer

        result = assert_jit_matches_eager(dust_to_gas_scaling_remy_ruyer, 0.0)
        assert jnp.isfinite(result)

    def test_differentiable(self):
        """Gradient w.r.t. logzsol is finite."""
        from tengri.components.dust.attenuation import dust_to_gas_scaling_remy_ruyer

        grad_jax = float(jax.grad(dust_to_gas_scaling_remy_ruyer)(0.0))
        grad_fd = fd_grad(lambda x: float(dust_to_gas_scaling_remy_ruyer(x)), 0.0)
        np.testing.assert_allclose(
            grad_jax, grad_fd, rtol=1e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
        )
