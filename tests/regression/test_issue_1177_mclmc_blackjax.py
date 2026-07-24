# SPDX-License-Identifier: BSD-3-Clause
"""Smoke tests for MCLMC backends: blackjax >= 1.5 adaptation interface (issue #1177).

blackjax >= 1.5 added logdensity_fn as a required parameter to mclmc adaptation
functions, and shifted its position in adjusted_mclmc_find_L_and_step_size.
This smoke test directly invokes tengri's feature-detecting helpers to verify
they correctly pass logdensity_fn to the adaptation functions, working on both
blackjax 1.3 (no param) and >= 1.5 (required param).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.regression_bug]

try:
    import blackjax

    HAS_BLACKJAX = True
except ImportError:
    HAS_BLACKJAX = False


@pytest.mark.skipif(not HAS_BLACKJAX, reason="blackjax not available")
def test_mclmc_adaptation_helper_feature_detects():
    """Test _call_mclmc_find_L_and_step_size correctly handles both versions."""
    from tengri.inference.backends.mcmc.mclmc import _call_mclmc_find_L_and_step_size

    def logdensity_fn(x):
        """Log-density of 2-D standard normal."""
        return -0.5 * jnp.sum(x**2)

    init_pos = jnp.array([0.0, 0.0])
    key = jax.random.PRNGKey(0)

    # Initialize state
    key, init_key = jax.random.split(key)
    state = blackjax.mcmc.mclmc.init(init_pos, logdensity_fn, init_key)

    # Build kernel factory
    def kernel_factory(inv_mass):
        return blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=logdensity_fn,
            inverse_mass_matrix=inv_mass,
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        )

    # Call via the helper (this is what run_mclmc uses)
    key, tune_key = jax.random.split(key)
    state, params, _ = _call_mclmc_find_L_and_step_size(
        logdensity_fn=logdensity_fn,
        mclmc_kernel=kernel_factory,
        num_steps=50,
        state=state,
        rng_key=tune_key,
        diagonal_preconditioning=True,
    )

    # Verify adaptation completed successfully
    assert hasattr(params, "L"), "params should have L"
    assert hasattr(params, "step_size"), "params should have step_size"
    assert np.isfinite(float(params.L)), "L should be finite"
    assert np.isfinite(float(params.step_size)), "step_size should be finite"


@pytest.mark.skipif(not HAS_BLACKJAX, reason="blackjax not available")
def test_adjusted_mclmc_adaptation_helper_feature_detects():
    """Test _call_adjusted_mclmc_find_L_and_step_size correctly handles both versions."""
    from tengri.inference.backends.mcmc.mclmc import (
        _call_adjusted_mclmc_find_L_and_step_size,
    )

    def logdensity_fn(x):
        """Log-density of 2-D standard normal."""
        return -0.5 * jnp.sum(x**2)

    init_pos = jnp.array([0.0, 0.0])
    key = jax.random.PRNGKey(0)

    # Initialize state
    state = blackjax.mcmc.adjusted_mclmc.init(init_pos, logdensity_fn)

    # Build kernel wrapper matching blackjax 1.3 adaptation interface
    def kernel_for_adaptation(
        rng_key, state, avg_num_integration_steps, step_size, inverse_mass_matrix
    ):
        """Build adapted kernel for blackjax 1.3 compatibility."""
        k = blackjax.mcmc.adjusted_mclmc.build_kernel(
            logdensity_fn=logdensity_fn,
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
            inverse_mass_matrix=inverse_mass_matrix,
        )
        n_steps = jnp.ceil(avg_num_integration_steps).astype(int)
        return k(rng_key, state, step_size, n_steps)

    # Call via the helper (this is what run_adjusted_mclmc uses)
    key, tune_key = jax.random.split(key)
    state, params, _ = _call_adjusted_mclmc_find_L_and_step_size(
        logdensity_fn=logdensity_fn,
        mclmc_kernel=kernel_for_adaptation,
        num_steps=50,
        state=state,
        rng_key=tune_key,
        target=0.65,
        diagonal_preconditioning=True,
    )

    # Verify adaptation completed successfully
    assert hasattr(params, "L"), "params should have L"
    assert hasattr(params, "step_size"), "params should have step_size"
    assert np.isfinite(float(params.L)), "L should be finite"
    assert np.isfinite(float(params.step_size)), "step_size should be finite"
