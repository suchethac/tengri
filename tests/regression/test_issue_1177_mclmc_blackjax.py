# SPDX-License-Identifier: BSD-3-Clause
"""Smoke guard for MCLMC backends: blackjax >= 1.5 adaptation interface.

Issue #1177: blackjax >= 1.5 added logdensity_fn as a required parameter
to mclmc adaptation functions, and shifted the position of logdensity_fn in
adjusted_mclmc_find_L_and_step_size (now position 2 instead of implicit).
This smoke test verifies both MCLMC and Adjusted MCLMC work on the installed
blackjax (1.3 has no param, >= 1.5 requires it) with version-aware calling.
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
def test_mclmc_smoke_standard_normal():
    """Test MCLMC sampler on a simple 2-D standard normal."""
    import inspect
    import time

    def logdensity_fn(x):
        """Log-density of a 2-D standard normal."""
        return -0.5 * jnp.sum(x**2)

    key = jax.random.PRNGKey(0)
    init_pos = jnp.array([0.0, 0.0])

    # Initialize the chain
    key, init_key = jax.random.split(key)
    state = blackjax.mcmc.mclmc.init(init_pos, logdensity_fn, init_key)

    # Build kernel factory
    def kernel_factory(inv_mass):
        return blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=logdensity_fn,
            inverse_mass_matrix=inv_mass,
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        )

    # Warmup: find L and step size, with version-aware calling
    key, tune_key = jax.random.split(key)
    n_warmup = 100
    t0 = time.time()

    # Detect if logdensity_fn is a required parameter (blackjax >= 1.5)
    sig = inspect.signature(blackjax.mclmc_find_L_and_step_size)
    adapt_kwargs = {
        "mclmc_kernel": kernel_factory,
        "num_steps": n_warmup,
        "state": state,
        "rng_key": tune_key,
        "diagonal_preconditioning": True,
    }
    if "logdensity_fn" in sig.parameters:
        adapt_kwargs["logdensity_fn"] = logdensity_fn

    state, params, _ = blackjax.mclmc_find_L_and_step_size(**adapt_kwargs)
    warmup_time = time.time() - t0

    # Sample
    kernel = kernel_factory(params.inverse_mass_matrix)
    key, sample_key = jax.random.split(key)
    sample_keys = jax.random.split(sample_key, 100)

    def _step(s, k):
        s, _ = kernel(k, s, params.L, params.step_size)
        return s, s.position

    _, positions = jax.lax.scan(_step, state, sample_keys)

    # Verify samples are finite and roughly centered at zero
    assert np.all(np.isfinite(positions)), "non-finite MCLMC samples"
    sample_mean = np.mean(positions, axis=0)
    assert np.all(np.abs(sample_mean) < 0.5), f"sample mean {sample_mean} not close to zero"

    print(
        f"MCLMC warmup: {warmup_time:.2f}s, L={float(params.L):.4f}, "
        f"step_size={float(params.step_size):.6f}"
    )
    print(f"Sample mean: {sample_mean}")


@pytest.mark.skipif(not HAS_BLACKJAX, reason="blackjax not available")
def test_adjusted_mclmc_smoke_standard_normal():
    """Test Adjusted MCLMC sampler on a simple 2-D standard normal."""
    import inspect
    import time

    def logdensity_fn(x):
        """Log-density of a 2-D standard normal."""
        return -0.5 * jnp.sum(x**2)

    key = jax.random.PRNGKey(0)
    init_pos = jnp.array([0.0, 0.0])

    # Initialize the chain
    state = blackjax.mcmc.adjusted_mclmc.init(init_pos, logdensity_fn)

    # Build kernel wrapper for adaptation
    def _kernel_for_adaptation(
        rng_key, state, avg_num_integration_steps, step_size, inverse_mass_matrix
    ):
        """Build adapted kernel for adjusted MCLMC tuning."""
        k = blackjax.mcmc.adjusted_mclmc.build_kernel(
            logdensity_fn=logdensity_fn,
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
            inverse_mass_matrix=inverse_mass_matrix,
        )
        n_steps = jnp.ceil(avg_num_integration_steps).astype(int)
        return k(rng_key, state, step_size, n_steps)

    # Warmup with version-aware calling
    key, tune_key = jax.random.split(key)
    n_warmup = 100
    target_accept_rate = 0.65
    t0 = time.time()

    # In blackjax >= 1.5, logdensity_fn is parameter 2 (after mclmc_kernel)
    sig = inspect.signature(blackjax.adjusted_mclmc_find_L_and_step_size)
    adapt_kwargs = {
        "mclmc_kernel": _kernel_for_adaptation,
        "num_steps": n_warmup,
        "state": state,
        "rng_key": tune_key,
        "target": target_accept_rate,
        "diagonal_preconditioning": True,
    }
    if "logdensity_fn" in sig.parameters:
        adapt_kwargs["logdensity_fn"] = logdensity_fn

    state, params, _ = blackjax.adjusted_mclmc_find_L_and_step_size(**adapt_kwargs)
    warmup_time = time.time() - t0

    # Sample
    kernel = blackjax.mcmc.adjusted_mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        inverse_mass_matrix=params.inverse_mass_matrix,
    )
    key, sample_key = jax.random.split(key)
    sample_keys = jax.random.split(sample_key, 100)
    L = params.L
    step_size = params.step_size
    n_integration_steps = jnp.ceil(L / step_size).astype(int)

    def _step(s, k):
        s, _ = kernel(k, s, step_size, n_integration_steps)
        return s, s.position

    _, positions = jax.lax.scan(_step, state, sample_keys)

    # Verify samples are finite and roughly centered at zero
    assert np.all(np.isfinite(positions)), "non-finite Adjusted MCLMC samples"
    sample_mean = np.mean(positions, axis=0)
    assert np.all(np.abs(sample_mean) < 0.5), f"sample mean {sample_mean} not close to zero"

    print(
        f"Adjusted MCLMC warmup: {warmup_time:.2f}s, L={float(params.L):.4f}, "
        f"step_size={float(params.step_size):.6f}"
    )
    print(f"Sample mean: {sample_mean}")
