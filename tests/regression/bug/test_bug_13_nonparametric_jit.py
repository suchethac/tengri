# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for BUG-13: nonparametric len() under JIT.

See ADR / docs/known_bugs.md for full context.
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from tests._jit_parity import assert_jit_matches_eager

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.regression_bug


class TestBug13NonparametricJit:
    """nonparametric.py:74 — Must not use len() on JAX arrays.

    len() is not JIT-compatible; must use .shape[0] or jnp.size() instead.
    """

    @pytest.mark.xfail(reason="BUG-13: len() on JAX array under JIT", strict=True)
    def test_continuity_sfh_jit(self):
        """continuity must JIT-compile without ConcretizationTypeError."""
        try:
            from tengri.components.stellar.sfh.nonparametric import continuity
        except ImportError:
            pytest.skip("nonparametric module not available")

        bin_edges = jnp.array([0.0, 0.1, 0.5, 1.0, 3.0, 6.0, 10.0, 13.0])
        log_ratios = jnp.zeros(6)
        age_grid = jnp.linspace(0.01, 13.0, 100)

        # This should work but currently raises ConcretizationTypeError
        result = assert_jit_matches_eager(continuity, log_ratios, age_grid, bin_edges)
        chex.assert_tree_all_finite(result)
