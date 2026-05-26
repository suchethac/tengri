# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for precompute_dust_age_mask dtype preservation bug.

Bug: attenuation.py:836 — hardcoded jnp.float64 defeats mixed-precision support.
"""

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


class TestDustAgeMaskDtype:
    """Bug: attenuation.py:836 — hardcoded jnp.float64."""

    def test_float32_input_gives_float32_mask(self):
        """float32 age grid should produce float32 masks, not float64."""
        from tengri.components.dust.attenuation import precompute_dust_age_mask

        age_grid_f32 = jnp.linspace(0.0, 1e10, 100, dtype=jnp.float32)
        young, old = precompute_dust_age_mask(age_grid_f32, t_birth=3e8)
        assert young.dtype == jnp.float32, f"young mask is {young.dtype}, expected float32"
        assert old.dtype == jnp.float32, f"old mask is {old.dtype}, expected float32"

    def test_float64_input_gives_float64_mask(self):
        """float64 age grid should produce float64 masks."""
        from tengri.components.dust.attenuation import precompute_dust_age_mask

        age_grid_f64 = jnp.linspace(0.0, 1e10, 100, dtype=jnp.float64)
        young, _old = precompute_dust_age_mask(age_grid_f64, t_birth=3e8)
        assert young.dtype == jnp.float64, f"young mask is {young.dtype}, expected float64"
