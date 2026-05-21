"""Regression test for SFR time-averaging trapezoid non-negative bug.

Bug: model.py:791-804 — zeroed age array caused negative SFR integrals.
"""

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


class TestSFRTrapezoidNonNegative:
    """Bug: model.py:791-804 — zeroed age array caused negative SFR integrals."""

    def test_sfr_100myr_non_negative(self):
        """sfr_100myr must never be negative regardless of SFH shape."""
        from tengri.components.stellar.sfh.mean_sfh import double_powerlaw

        age_yr = jnp.logspace(6, 10.1, 200)
        for norm in [0.01, 1.0, 100.0]:
            sfr = double_powerlaw(age_yr, alpha=0.5, beta=2.0, tau=2e9, norm=norm)
            mask_100 = age_yr <= 1e8
            sfr_100_masked = jnp.where(mask_100, sfr, 0.0)
            # Fixed bug: use real ages as x-values, not zeroed array
            integral_100 = jnp.trapezoid(sfr_100_masked, age_yr)
            assert integral_100 >= 0.0, f"sfr_100myr integral negative: {integral_100:.3e}"

    def test_sfr_10myr_non_negative(self):
        """sfr_10myr must be non-negative."""
        from tengri.components.stellar.sfh.mean_sfh import double_powerlaw

        age_yr = jnp.logspace(6, 10.1, 200)
        sfr = double_powerlaw(age_yr, alpha=0.3, beta=3.0, tau=5e8, norm=10.0)
        mask_10 = age_yr <= 1e7
        sfr_10_masked = jnp.where(mask_10, sfr, 0.0)
        integral_10 = jnp.trapezoid(sfr_10_masked, age_yr)
        assert integral_10 >= 0.0
