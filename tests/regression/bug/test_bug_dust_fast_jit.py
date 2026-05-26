# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for single_component_dust_fast JIT-safe bug.

Bug: attenuation.py:1077 — len(wavelengths) is not JIT-safe.
"""

import chex
import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


class TestSingleComponentDustFastJITSafe:
    """Bug: attenuation.py:1077 — len(wavelengths) not JIT-safe."""

    def test_jit_compilable(self):
        """single_component_dust_fast should JIT-compile without errors."""
        from tengri.components.dust.attenuation import single_component_dust_fast

        wave = jnp.linspace(1000.0, 10000.0, 50)
        n_ages = 10

        @jax.jit
        def _eval():
            return single_component_dust_fast(wave, tau_v=1.0, n_ages=n_ages)

        result = _eval()
        chex.assert_shape(result, (n_ages, wave.shape[0]))
        chex.assert_tree_all_finite(result)
