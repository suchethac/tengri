"""MCMC runners: Ray Tracing, NUTS, and Elliptical Slice Sampling.

Extracted from fitter.py.
"""

from __future__ import annotations

import time
import warnings

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def run_raytrace(
    fitter,
    *,
    key,
    init_from=None,
    n_burnin=100,
    n_steps=500,
    n_leapfrog_steps=10,
    step_size=None,
    refresh_rate=0.0,
    verbose=True,
):
    """Ray Tracing Sampler (Behroozi 2025).

    Propagates light rays through a medium where the refractive
    index n(x) = L(x)^{1/(D-1)}, using Snell's law to bend rays
    toward high-likelihood regions.

    The sampling proceeds in two phases:
    1. **Burn-in**: initial samples are discarded to let the chain
       forget its starting position and reach the typical set.
    2. **Sampling**: posterior samples are collected.

    Parameters
    ----------
    n_burnin : int
        Burn-in steps (discarded).
    n_steps : int
        Post-burn-in samples to collect.
    n_leapfrog_steps : int
        Leapfrog integration steps per trajectory.
    step_size : float, optional
        Integration step size. Default: 0.03 * sqrt(D).
    refresh_rate : float
        Partial momentum refresh rate. 0 = no refresh (pure ray tracing).
    verbose : bool
        Print progress.
    """
    from tengri.inference.backends.mcmc.raytrace import sample_raytrace
    from tengri.inference.posterior import Posterior

    loss_fn = fitter._get_or_build_loss_fn()
    data_args = fitter._data_args

    if init_from is not None:
        init_params = fitter._unbounded_from_posterior(init_from)
    else:
        init_params = fitter._initialize_unbounded(key)

    # Flatten for the sampler (expects a flat 1D array)
    init_flat, unravel_fn = ravel_pytree(init_params)
    D = len(init_flat)

    if step_size is None:
        # Behroozi (2025) recommends 0.03 * sqrt(D), but for
        # stochastic SFH models the psd_xi variables create a
        # tighter curvature. Use a smaller default for D > 10.
        if D <= 10:
            step_size = 0.03 * jnp.sqrt(float(D))
        else:
            step_size = 0.01

    def log_prob_flat(position):
        params = unravel_fn(position)
        return -loss_fn(params, data_args)

    total_steps = n_burnin + n_steps

    if verbose:
        print(
            f"Ray Tracing: {D} params, {n_burnin} burn-in + "
            f"{n_steps} samples, {n_leapfrog_steps} leapfrog/step, "
            f"step_size={float(step_size):.4f}"
        )

    t0 = time.time()

    key, sample_key = jax.random.split(key)
    chain, log_likelihood, accept_prob = sample_raytrace(
        key=sample_key,
        params_init=init_flat,
        log_prob_fn=log_prob_flat,
        n_steps=total_steps,
        n_leapfrog_steps=n_leapfrog_steps,
        step_size=float(step_size),
        refresh_rate=float(refresh_rate),
        metro_check=1,
        sample_hmc=False,
    )

    wall_time = time.time() - t0

    # Discard burn-in
    chain = chain[n_burnin:]
    log_likelihood = log_likelihood[n_burnin:]
    accept_prob_post = accept_prob[n_burnin:]
    n_samples_out = chain.shape[0]

    mean_accept = float(jnp.mean(accept_prob))
    mean_accept_post = float(jnp.mean(accept_prob_post))

    # Convert to physical parameter space (vectorized)
    def _convert_one(flat_sample):
        return fitter._to_physical(unravel_fn(flat_sample))

    samples_phys = jax.vmap(_convert_one)(chain)
    best_params = {k: jnp.mean(v, axis=0) for k, v in samples_phys.items()}

    if verbose:
        print(
            f"  Ray Tracing complete in {wall_time:.1f}s. "
            f"Acceptance: {mean_accept:.1%} (overall), "
            f"{mean_accept_post:.1%} (post burn-in). "
            f"Samples: {n_samples_out}"
        )

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="Ray Tracing (Behroozi 2025)",
        wall_time_s=wall_time,
        diagnostics={
            "n_burnin": n_burnin,
            "n_steps": n_steps,
            "n_samples": n_samples_out,
            "n_leapfrog_steps": n_leapfrog_steps,
            "step_size": float(step_size),
            "refresh_rate": float(refresh_rate),
            "accept_rate": mean_accept,
            "accept_rate_post_burnin": mean_accept_post,
        },
        loss_history=None,
        _model=fitter.model,
    )


def run_nuts(
    fitter,
    *,
    key,
    init_from=None,
    n_warmup=300,
    n_burnin=100,
    n_samples=1000,
    target_accept_rate=0.85,
    max_num_doublings=10,
    dense_mass_matrix=True,
    verbose=True,
):
    """NUTS sampling via BlackJAX.

    Default strategy designed for SED fitting posteriors with strong
    degeneracies (age-dust-metallicity, SFR-mass), bounded parameters,
    and D=6-20 free parameters.

    The sampling proceeds in four phases:

    1. **MAP initialization** (automatic if ``init_from`` is None):
       200-step MAP optimization to find the posterior mode. Starts
       warmup near the mode instead of the prior, reducing warmup
       time by ~5x and improving mass matrix quality.
    2. **Warmup** (300 steps): BlackJAX window adaptation tunes
       step size and mass matrix.  Dense mass matrix (default)
       captures parameter correlations — critical for the
       age-dust-metallicity degeneracy.
    3. **Burn-in** (100 steps): post-warmup samples discarded.
       Lets the chain diffuse away from the MAP point estimate
       to the typical set of the posterior.
    4. **Sampling** (1000 steps): posterior samples collected.

    Convergence criterion: N > 5τ for all parameters (Sokal/Behroozi).
    Check with ``result.check_convergence()``.

    Parameters
    ----------
    init_from : str, Posterior, or None
        Initialization strategy. None (default) runs a quick MAP
        for warm-starting the chain. Pass a Posterior to start from
        a previous result.
    n_warmup : int
        Warmup/adaptation steps (tunes step size and mass matrix).
        300 is sufficient for D≤15 with MAP init. Increase for
        high-D or difficult geometries.
    n_burnin : int
        Post-warmup burn-in steps (discarded). Lets the chain forget
        the MAP initialization and reach the typical set. Set to 0
        if init_from is already a Posterior from a converged chain.
    n_samples : int
        Posterior samples to collect. 1000 gives convergence for
        most SED fitting scenarios at D≤10. Increase if
        ``check_convergence()`` reports unconverged parameters.
    target_accept_rate : float
        Target acceptance rate for step size adaptation. 0.85 is
        slightly more conservative than the Stan default (0.8),
        reducing divergences in the SED degeneracy banana. Range
        0.7-0.95; higher = smaller steps = fewer divergences but
        slower mixing.
    max_num_doublings : int
        Maximum tree depth for NUTS trajectory (2^max_num_doublings
        leapfrog steps per sample).
    dense_mass_matrix : bool
        Use a dense (full) mass matrix instead of diagonal. Captures
        parameter correlations (e.g. age-dust-metallicity) and
        dramatically reduces divergences. Default True. Set False
        for D>20 where the dense matrix becomes expensive.
    verbose : bool
        Print progress.
    """
    try:
        import blackjax
    except ImportError:
        raise ImportError("blackjax required for NUTS: pip install blackjax") from None

    from tengri.inference.posterior import Posterior

    # Warn about high dimensionality
    if fitter.spec.stochastic:
        n_total = fitter.spec.n_free + fitter.spec.n_grid
        warnings.warn(
            f"Stochastic SFH with NUTS: sampling {n_total} dimensions "
            f"({fitter.spec.n_grid} psd_xi + {fitter.spec.n_free} physical). "
            f"This is computationally expensive. "
            f"Recommended: method='vi' (10-100x faster).",
            stacklevel=3,
        )

    loss_fn = fitter._get_or_build_loss_fn()
    data_args = fitter._data_args

    if init_from is not None:
        init_params = fitter._unbounded_from_posterior(init_from)
    else:
        # Auto-MAP initialization: find the posterior mode first,
        # then start NUTS warmup from there.  Improves warmup
        # convergence (~5x faster) and mass matrix quality.
        if verbose:
            print("  MAP initialization (200 steps)...")
        key, map_key = jax.random.split(key)
        map_result = fitter._run_map(key=map_key, n_steps=200, verbose=False)
        init_params = fitter._unbounded_from_posterior(map_result)
        if verbose:
            print(f"  MAP init done (loss={map_result.diagnostics['final_loss']:.2f})")

    # Flatten for BlackJAX
    init_flat, unravel_fn = ravel_pytree(init_params)

    def log_posterior_flat(position):
        params = unravel_fn(position)
        return -loss_fn(params, data_args)

    if verbose:
        n_dim = len(init_flat)
        # Auto-adjust warnings based on dimensionality
        if n_dim > 20 and not fitter.spec.stochastic:
            warnings.warn(
                f"NUTS with {n_dim} dimensions may be slow. "
                f"Consider method='vi' or method='mcmc_raytrace' for D>{20}.",
                stacklevel=3,
            )
        burnin_msg = f", {n_burnin} burn-in" if n_burnin > 0 else ""
        print(
            f"NUTS: {n_dim} parameters, {n_warmup} warmup{burnin_msg}, "
            f"{n_samples} samples, target_accept={target_accept_rate}"
        )

    t0 = time.time()

    # Auto-select mass matrix type based on dimensionality.
    # Dense: O(D²) per step, captures correlations. Good for D≤30.
    # Diagonal: O(D) per step, sufficient when init_from=Posterior
    #   (VI already decorrelated) or D>30.
    n_dim = len(init_flat)
    use_dense = dense_mass_matrix
    if dense_mass_matrix and n_dim > 30:
        use_dense = False
        if verbose:
            print(
                f"  Auto-switching to diagonal mass matrix (D={n_dim}>30). "
                f"Dense would be O({n_dim}²)={n_dim**2} per step."
            )

    key, warmup_key = jax.random.split(key)
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        log_posterior_flat,
        is_mass_matrix_diagonal=not use_dense,
        target_acceptance_rate=target_accept_rate,
    )
    (state, parameters), _ = warmup.run(warmup_key, init_flat, num_steps=n_warmup)

    if verbose:
        print(
            f"  Warmup complete ({time.time() - t0:.1f}s). "
            f"Step size: {float(parameters['step_size']):.4f}"
        )

    # Sampling — use jax.lax.scan for zero Python dispatch overhead
    kernel = blackjax.nuts(log_posterior_flat, **parameters).step

    @jax.jit
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state.position, info.is_divergent)

    # Burn-in via scan (discarded)
    if n_burnin > 0:
        key, burnin_key = jax.random.split(key)
        burnin_keys = jax.random.split(burnin_key, n_burnin)

        @jax.jit
        def burnin_scan(state, keys):
            def _step(s, k):
                s, _ = one_step(s, k)
                return s, None

            s, _ = jax.lax.scan(_step, state, keys)
            return s

        state = burnin_scan(state, burnin_keys)
        if verbose:
            print(f"  Burn-in complete ({n_burnin} steps discarded)")

    key, sample_key = jax.random.split(key)
    sample_keys = jax.random.split(sample_key, n_samples)

    # Sampling via scan — single JIT call for all samples
    @jax.jit
    def sample_scan(state, keys):
        def _step(s, k):
            s, (pos, div) = one_step(s, k)
            return s, (pos, div)

        return jax.lax.scan(_step, state, keys)

    _, (positions, divergent) = sample_scan(state, sample_keys)
    n_divergent = int(jnp.sum(divergent))

    wall_time = time.time() - t0

    if verbose:
        print(f"  Sampling complete ({n_samples} samples)")

    # Vectorized post-processing: unravel + convert to physical
    def _convert_one(flat_pos):
        return fitter._to_physical(unravel_fn(flat_pos))

    samples_phys = jax.vmap(_convert_one)(positions)
    best_params = {k: jnp.mean(v, axis=0) for k, v in samples_phys.items()}

    if verbose:
        print(f"  NUTS complete in {wall_time:.1f}s. Divergences: {n_divergent}/{n_samples}")

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="NUTS (BlackJAX)",
        wall_time_s=wall_time,
        diagnostics={
            "n_warmup": n_warmup,
            "n_burnin": n_burnin,
            "n_samples": n_samples,
            "n_divergent": n_divergent,
            "step_size": float(parameters["step_size"]),
        },
        loss_history=None,
        _model=fitter.model,
    )


# -------------------------------------------------------------------
# Laplace, Pathfinder, Elliptical Slice Sampling
# -------------------------------------------------------------------


def run_elliptical_slice(fitter, *, key, init_from=None, **kwargs):
    """Elliptical Slice Sampling for Gaussian-prior latent models."""
    from tengri.inference.backends.mcmc.elliptical_slice import run_elliptical_slice

    loglik_unbounded_fn = fitter._build_loglikelihood_unbounded_fn()
    data_args = fitter._data_args

    if init_from is not None:
        init_params = fitter._unbounded_from_posterior(init_from)
    else:
        init_params = fitter._initialize_unbounded(key)

    return run_elliptical_slice(
        key=key,
        loglikelihood_unbounded_fn=loglik_unbounded_fn,
        data_args=data_args,
        init_params=init_params,
        to_physical_fn=fitter._to_physical,
        model=fitter.model,
        **kwargs,
    )
