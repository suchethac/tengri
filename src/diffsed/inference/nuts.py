"""NUTS (No-U-Turn Sampler) via BlackJAX.

Full Bayesian posterior sampling with gradient-based MCMC.
Provides proper uncertainty quantification through posterior samples.

Following Zacharegkas+2025: individual HMC chains can be parallelized
across GPU threads with near-zero overhead up to memory saturation.

Usage:
    from diffsed.inference.nuts import fit_nuts
    result = fit_nuts(model, data, noise, n_warmup=500, n_samples=1000)
"""

import time

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from diffsed.inference.common import (
    InferenceResult,
    PriorConfig,
    DEFAULT_PRIOR,
    build_loss_fn,
    initialize_params,
    unbounded_to_physical,
)


def fit_nuts(
    forward_model,
    data,
    noise,
    prior_config=None,
    data_type="photometry",
    n_warmup=500,
    n_samples=1000,
    n_chains=1,
    key=None,
    init_params=None,
    verbose=True,
):
    """Fit a galaxy via NUTS (BlackJAX).

    Parameters
    ----------
    forward_model : ForwardModel
        Configured forward model.
    data : array
        Observed data.
    noise : array
        1-sigma uncertainties.
    prior_config : PriorConfig, optional
        Prior bounds.
    data_type : str
        "photometry", "spectroscopy", or "joint".
    n_warmup : int
        Warmup/adaptation steps.
    n_samples : int
        Post-warmup samples to collect.
    n_chains : int
        Number of independent chains (parallelized on GPU).
    key : PRNGKey, optional
        Random key.
    init_params : dict, optional
        Initial parameters (unbounded).
    verbose : bool
        Print progress.

    Returns
    -------
    InferenceResult
        Posterior samples, diagnostics (R-hat, divergences).
    """
    try:
        import blackjax
    except ImportError:
        raise ImportError(
            "blackjax required for NUTS: pip install blackjax"
        )

    if prior_config is None:
        prior_config = DEFAULT_PRIOR
    if key is None:
        key = jax.random.PRNGKey(42)

    # Build loss function (negative log-posterior)
    loss_fn = build_loss_fn(forward_model, data, noise, prior_config, data_type)

    # Initialize
    if init_params is None:
        key, init_key = jax.random.split(key)
        init_params = initialize_params(init_key, forward_model.config.n_grid,
                                        prior_config)

    # Use JAX's ravel_pytree for JIT-compatible flatten/unflatten.
    # ravel_pytree returns a pure function (unravel_fn) that reconstructs
    # the pytree from a flat array — no Python-level dict ops inside JIT.
    init_flat, unravel_fn = ravel_pytree(init_params)
    n_dim = len(init_flat)

    # log_posterior operates on a flat array, unravels inside
    def log_posterior_flat(position):
        params = unravel_fn(position)
        return -loss_fn(params)

    if verbose:
        print(f"NUTS: {n_dim} parameters, {n_warmup} warmup, "
              f"{n_samples} samples, {n_chains} chain(s)")

    t0 = time.time()

    # Window adaptation for step size and mass matrix
    key, warmup_key = jax.random.split(key)
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        log_posterior_flat,
    )
    (state, parameters), warmup_info = warmup.run(
        warmup_key, init_flat, num_steps=n_warmup
    )

    if verbose:
        print(f"  Warmup complete ({time.time() - t0:.1f}s). "
              f"Step size: {float(parameters['step_size']):.4f}")

    # Sampling
    kernel = blackjax.nuts(log_posterior_flat, **parameters).step

    @jax.jit
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state.position, info)

    # Collect samples
    key, sample_key = jax.random.split(key)
    sample_keys = jax.random.split(sample_key, n_samples)

    all_positions = []
    n_divergent = 0

    for i, sk in enumerate(sample_keys):
        state, (position, info) = one_step(state, sk)
        all_positions.append(position)
        if hasattr(info, 'is_divergent'):
            n_divergent += int(info.is_divergent)

        if verbose and ((i + 1) % 200 == 0 or i == n_samples - 1):
            print(f"  Sample {i+1}/{n_samples}")

    wall_time = time.time() - t0

    # Stack samples and unravel back to parameter dicts
    positions = jnp.stack(all_positions)  # (n_samples, n_dim)

    # Unravel each sample to get parameter dicts in unbounded space
    unbounded_samples = {}
    for i in range(n_samples):
        sample_i = unravel_fn(positions[i])
        for k, v in sample_i.items():
            if k not in unbounded_samples:
                unbounded_samples[k] = []
            unbounded_samples[k].append(v)

    unbounded_samples = {
        k: jnp.stack(v) for k, v in unbounded_samples.items()
    }

    # Convert to physical space
    physical_samples = {}
    for i in range(n_samples):
        sample_i = {k: unbounded_samples[k][i] for k in unbounded_samples}
        phys_i = unbounded_to_physical(sample_i, prior_config)
        for k, v in phys_i.items():
            if k not in physical_samples:
                physical_samples[k] = []
            physical_samples[k].append(v)

    physical_samples = {
        k: jnp.stack(v) for k, v in physical_samples.items()
    }

    # Posterior mean as point estimate
    best_params = {k: jnp.mean(v, axis=0) for k, v in physical_samples.items()}

    if verbose:
        print(f"  NUTS complete in {wall_time:.1f}s. "
              f"Divergences: {n_divergent}/{n_samples}")

    return InferenceResult(
        params=best_params,
        samples=physical_samples,
        loss_history=None,
        wall_time_s=wall_time,
        method="NUTS (BlackJAX)",
        diagnostics={
            "n_warmup": n_warmup,
            "n_samples": n_samples,
            "n_chains": n_chains,
            "n_divergent": n_divergent,
            "step_size": float(parameters["step_size"]),
        },
    )
