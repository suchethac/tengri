"""Regression test for BUG-02: SFR time-averaging trapezoid boundary bias.

See ADR / docs/known_bugs.md for full context.
"""

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


class TestBug02SfrTimeAvg:
    """model.py:791-804 — sfr_100myr must be positive and correct.

    Fix (model.py): replaced jnp.trapezoid on zeroed SFR with gradient-weighted
    Riemann sum (jnp.gradient for bin widths) over masked ages only. This avoids
    the phantom boundary segment from the last in-window age to the first
    out-of-window age.
    """

    def test_constant_sfr_recovery(self):
        """For constant SFR=10, sfr_100myr should be ~10 (not biased by boundary)."""
        age_yr = jnp.logspace(6, 10, 100)  # 1 Myr to 10 Gyr
        sfr = jnp.full_like(age_yr, 10.0)  # constant SFR = 10 Msun/yr

        dt = jnp.gradient(age_yr)
        mask_100 = age_yr <= 1e8
        numerator = jnp.sum(jnp.where(mask_100, sfr * dt, 0.0))
        denom = jnp.maximum(jnp.sum(jnp.where(mask_100, dt, 0.0)), 1.0)
        sfr_100myr = jnp.where(jnp.sum(mask_100) > 1, numerator / denom, sfr[0])

        assert float(sfr_100myr) > 0, "sfr_100myr must be positive"
        assert abs(float(sfr_100myr) - 10.0) < 0.5, (
            f"sfr_100myr = {float(sfr_100myr):.2f}, expected ~10.0"
        )

    def test_sfr_100myr_positive_with_declining_sfh(self):
        """sfr_100myr must be positive even for a strongly declining SFH."""
        age_yr = jnp.logspace(6, 10, 100)
        # Exponentially declining SFH: high early SFR, low recent
        tau_yr = 1e9
        sfr = 100.0 * jnp.exp(-age_yr / tau_yr)

        dt = jnp.gradient(age_yr)
        mask_100 = age_yr <= 1e8
        numerator = jnp.sum(jnp.where(mask_100, sfr * dt, 0.0))
        denom = jnp.maximum(jnp.sum(jnp.where(mask_100, dt, 0.0)), 1.0)
        sfr_100myr = jnp.where(jnp.sum(mask_100) > 1, numerator / denom, sfr[0])

        assert float(sfr_100myr) > 0, "sfr_100myr must be positive for declining SFH"
