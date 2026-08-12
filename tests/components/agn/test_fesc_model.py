# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the Chisholm+2022 LyC escape fraction model.

Reference: Chisholm et al. 2022, ApJ, 931, 37 (LzLCS).
"""

import pytest

pytestmark = pytest.mark.bounds
import jax
import jax.numpy as jnp
import numpy as np

from tests._jit_parity import assert_jit_matches_eager


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


class TestFescChisholm:
    """Verify fesc_chisholm2022 calibration and properties."""

    def test_calibration_at_beta_minus2(self):
        """At β=-2, default params → f_esc = fesc_0 = 0.15."""
        from tengri.components.nebular.fesc_model import fesc_chisholm2022

        result = fesc_chisholm2022(-2.0)
        assert jnp.isclose(result, 0.15, atol=1e-6)

    def test_blue_slope_high_fesc(self):
        """β = -2.5 (very blue) → higher f_esc than at β = -2."""
        from tengri.components.nebular.fesc_model import fesc_chisholm2022

        fesc_blue = fesc_chisholm2022(-2.5)
        fesc_ref = fesc_chisholm2022(-2.0)
        assert fesc_blue > fesc_ref

    def test_red_slope_low_fesc(self):
        """β = 0 (red) → lower f_esc than at β = -2."""
        from tengri.components.nebular.fesc_model import fesc_chisholm2022

        fesc_red = fesc_chisholm2022(0.0)
        fesc_ref = fesc_chisholm2022(-2.0)
        assert fesc_red < fesc_ref

    def test_bounded_0_1(self):
        """f_esc always in [0, 1] for a wide range of β values."""
        from tengri.components.nebular.fesc_model import fesc_chisholm2022

        betas = jnp.linspace(-4.0, 2.0, 100)
        for beta in betas:
            fesc = fesc_chisholm2022(float(beta))
            assert 0.0 <= float(fesc) <= 1.0

    def test_monotonic_with_beta(self):
        """More negative β → higher f_esc (bluer = more escape)."""
        from tengri.components.nebular.fesc_model import fesc_chisholm2022

        betas = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0]
        fescs = [float(fesc_chisholm2022(b)) for b in betas]
        for i in range(len(fescs) - 1):
            assert fescs[i] >= fescs[i + 1], f"Not monotonic at β={betas[i]}"

    def test_ssfr_dependence_disabled_by_default(self):
        """With a2=0 (default), sSFR has no effect."""
        from tengri.components.nebular.fesc_model import fesc_chisholm2022

        f1 = fesc_chisholm2022(-2.0, log_ssfr=-8.0)
        f2 = fesc_chisholm2022(-2.0, log_ssfr=-12.0)
        assert jnp.isclose(f1, f2, atol=1e-10)

    def test_ssfr_dependence_when_enabled(self):
        """With a2 > 0, higher sSFR → higher f_esc."""
        from tengri.components.nebular.fesc_model import fesc_chisholm2022

        f_high = fesc_chisholm2022(-2.0, log_ssfr=-8.0, a2=0.5)
        f_low = fesc_chisholm2022(-2.0, log_ssfr=-11.0, a2=0.5)
        assert f_high > f_low

    def test_jit_compatible(self):
        """Function can be JIT-compiled."""
        from tengri.components.nebular.fesc_model import fesc_chisholm2022

        result = assert_jit_matches_eager(fesc_chisholm2022, -2.0)
        assert jnp.isfinite(result)

    def test_differentiable(self):
        """Gradient w.r.t. β agrees with FD (Chisholm+2022)."""
        from tengri.components.nebular.fesc_model import fesc_chisholm2022

        def f(beta: float) -> float:
            return float(fesc_chisholm2022(beta))

        grad_jax = float(jax.grad(fesc_chisholm2022)(-2.0))
        np.testing.assert_allclose(
            grad_jax,
            fd_grad(f, -2.0),
            rtol=1e-3,
            err_msg="fesc_chisholm2022: FD check ∂f_esc/∂β",
        )


class TestComputeUVSlope:
    """Verify UV slope computation."""

    def test_power_law_recovery(self):
        """Recover β from a known power-law spectrum f_λ ∝ λ^β."""
        from tengri.components.nebular.fesc_model import compute_uv_slope

        beta_true = -2.0
        wave = jnp.linspace(1200.0, 3000.0, 500)
        c_aa = 2.99792458e18
        l_lambda = wave**beta_true
        l_nu = l_lambda * wave**2 / c_aa
        beta_fit = compute_uv_slope(wave, l_nu)
        assert jnp.isclose(beta_fit, beta_true, atol=0.05)

    def test_steep_slope(self):
        """Recover steep UV slope β = -3."""
        from tengri.components.nebular.fesc_model import compute_uv_slope

        beta_true = -3.0
        wave = jnp.linspace(1200.0, 3000.0, 500)
        c_aa = 2.99792458e18
        l_lambda = wave**beta_true
        l_nu = l_lambda * wave**2 / c_aa
        beta_fit = compute_uv_slope(wave, l_nu)
        assert jnp.isclose(beta_fit, beta_true, atol=0.05)

    def test_jit_compatible(self):
        """UV slope computation can be JIT-compiled."""
        from tengri.components.nebular.fesc_model import compute_uv_slope

        wave = jnp.linspace(1200.0, 3000.0, 500)
        l_nu = jnp.ones_like(wave) * 1e28
        result = assert_jit_matches_eager(compute_uv_slope, wave, l_nu)
        assert jnp.isfinite(result)
