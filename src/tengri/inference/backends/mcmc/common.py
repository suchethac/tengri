"""MCMC runners: HMC family, MCLMC, Ray Tracing, and Elliptical Slice Sampling.

Extracted from fitter.py. All BlackJAX-based samplers use module-level
cached scan functions with stable JIT identity to avoid recompilation.
"""

from __future__ import annotations

import functools
import logging
import time
import warnings

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from tengri.inference._model_cache import get_model_cache
from tengri.inference._sample_utils import _maybe_map_init, _mean_params, _vmap_samples_to_physical

logger = logging.getLogger(__name__)

# ── Cached NUTS scan functions (module-level for stable JIT identity)
# Using blackjax.mcmc.nuts.build_kernel() returns a kernel that takes
# logdensity_fn, step_size, inverse_mass_matrix as ARGUMENTS instead of
# closing over them.  This lets us define module-level @jax.jit functions
# whose trace cache key depends only on the logdensity_fn identity (stable
# via _get_flat_logdensity) — not on warmup parameters.


@functools.cache
def _get_nuts_kernel():
    import blackjax.mcmc.nuts

    return blackjax.mcmc.nuts.build_kernel()


@functools.cache
def _get_hmc_kernel():
    import blackjax.mcmc.hmc

    return blackjax.mcmc.hmc.build_kernel()


@functools.cache
def _get_dynamic_hmc_kernel():
    import blackjax.mcmc.dynamic_hmc

    return blackjax.mcmc.dynamic_hmc.build_kernel()


@functools.cache
def _get_ghmc_kernel():
    import blackjax.mcmc.ghmc

    return blackjax.mcmc.ghmc.build_kernel()


# --- NUTS scans ---
# All scan functions take a 2-arg logdensity_fn(position, data_args) as
# static_argnums so its identity is the cache key.  data_args flows in
# as a regular traced argument — changing galaxy data does NOT trigger
# recompilation.


@functools.partial(jax.jit, static_argnums=(2, 5))
def _nuts_sample_scan(
    state,
    keys,
    logdensity_fn_2arg,
    step_size,
    inv_mass_matrix,
    max_doublings,
    data_args,
):
    kernel = _get_nuts_kernel()

    def ld(pos):
        return logdensity_fn_2arg(pos, data_args)

    def _step(s, k):
        s, info = kernel(k, s, ld, step_size, inv_mass_matrix, max_doublings)
        return s, (s.position, info.is_divergent)

    return jax.lax.scan(_step, state, keys)


@functools.partial(jax.jit, static_argnums=(2, 5))
def _nuts_burnin_scan(
    state,
    keys,
    logdensity_fn_2arg,
    step_size,
    inv_mass_matrix,
    max_doublings,
    data_args,
):
    kernel = _get_nuts_kernel()

    def ld(pos):
        return logdensity_fn_2arg(pos, data_args)

    def _step(s, k):
        s, _info = kernel(k, s, ld, step_size, inv_mass_matrix, max_doublings)
        return s, None

    s, _ = jax.lax.scan(_step, state, keys)
    return s


# --- HMC scans ---


@functools.partial(jax.jit, static_argnums=(2, 5))
def _hmc_sample_scan(
    state,
    keys,
    logdensity_fn_2arg,
    step_size,
    inv_mass_matrix,
    n_leapfrog,
    data_args,
):
    kernel = _get_hmc_kernel()

    def ld(pos):
        return logdensity_fn_2arg(pos, data_args)

    def _step(s, k):
        s, info = kernel(k, s, ld, step_size, inv_mass_matrix, n_leapfrog)
        return s, (s.position, info.is_divergent)

    return jax.lax.scan(_step, state, keys)


@functools.partial(jax.jit, static_argnums=(2, 5))
def _hmc_burnin_scan(
    state,
    keys,
    logdensity_fn_2arg,
    step_size,
    inv_mass_matrix,
    n_leapfrog,
    data_args,
):
    kernel = _get_hmc_kernel()

    def ld(pos):
        return logdensity_fn_2arg(pos, data_args)

    def _step(s, k):
        s, _info = kernel(k, s, ld, step_size, inv_mass_matrix, n_leapfrog)
        return s, None

    s, _ = jax.lax.scan(_step, state, keys)
    return s


# --- Dynamic HMC scans ---


@functools.partial(jax.jit, static_argnums=(2,))
def _dynamic_hmc_sample_scan(
    state,
    keys,
    logdensity_fn_2arg,
    step_size,
    inv_mass_matrix,
    data_args,
):
    kernel = _get_dynamic_hmc_kernel()

    def ld(pos):
        return logdensity_fn_2arg(pos, data_args)

    def _step(s, k):
        s, info = kernel(k, s, ld, step_size, inv_mass_matrix)
        return s, (s.position, info.is_divergent)

    return jax.lax.scan(_step, state, keys)


@functools.partial(jax.jit, static_argnums=(2,))
def _dynamic_hmc_burnin_scan(
    state,
    keys,
    logdensity_fn_2arg,
    step_size,
    inv_mass_matrix,
    data_args,
):
    kernel = _get_dynamic_hmc_kernel()

    def ld(pos):
        return logdensity_fn_2arg(pos, data_args)

    def _step(s, k):
        s, _info = kernel(k, s, ld, step_size, inv_mass_matrix)
        return s, None

    s, _ = jax.lax.scan(_step, state, keys)
    return s


# --- GHMC scans ---


@functools.partial(jax.jit, static_argnums=(2,))
def _ghmc_sample_scan(
    state,
    keys,
    logdensity_fn_2arg,
    step_size,
    momentum_inv_scale,
    alpha,
    delta,
    data_args,
):
    kernel = _get_ghmc_kernel()

    def ld(pos):
        return logdensity_fn_2arg(pos, data_args)

    def _step(s, k):
        s, info = kernel(k, s, ld, step_size, momentum_inv_scale, alpha, delta)
        return s, (s.position, info.is_divergent)

    return jax.lax.scan(_step, state, keys)


@functools.partial(jax.jit, static_argnums=(2,))
def _ghmc_burnin_scan(
    state,
    keys,
    logdensity_fn_2arg,
    step_size,
    momentum_inv_scale,
    alpha,
    delta,
    data_args,
):
    kernel = _get_ghmc_kernel()

    def ld(pos):
        return logdensity_fn_2arg(pos, data_args)

    def _step(s, k):
        s, _info = kernel(k, s, ld, step_size, momentum_inv_scale, alpha, delta)
        return s, None

    s, _ = jax.lax.scan(_step, state, keys)
    return s


# --- MCLMC scans ---
# MCLMC/adjusted_mclmc kernels bake in logdensity_fn and inverse_mass_matrix
# at build time.  The kernel itself is the static identity for caching.
# Adaptation params are cached on the Model; the kernel is rebuilt per-fitter
# (cheap ~ms) using the fitter's data_args.


@functools.partial(jax.jit, static_argnums=(2,))
def _mclmc_sample_scan(state, keys, kernel, L, step_size):
    def _step(s, k):
        s, _info = kernel(k, s, L, step_size)
        return s, s.position

    return jax.lax.scan(_step, state, keys)


@functools.partial(jax.jit, static_argnums=(2,))
def _adjusted_mclmc_sample_scan(state, keys, kernel, step_size, n_integration_steps):
    def _step(s, k):
        s, info = kernel(k, s, step_size, n_integration_steps)
        return s, (s.position, info.is_divergent)

    return jax.lax.scan(_step, state, keys)


def _get_flat_logdensity(fitter, init_params):
    """Return (log_posterior_flat_2arg, unravel_fn, init_flat, data_args).

    The returned ``log_posterior_flat_2arg(position, data_args)`` takes
    ``data_args`` as a **traced** JAX argument (not closed over), so
    the compiled XLA program is reused across galaxies sharing the
    same model structure.  The function is cached on the **Model**
    (not fitter) so that multiple Fitters with different data share
    the same compiled code.
    """
    cache_key = fitter._engine_cache_key()
    model = fitter.model
    cache = get_model_cache(model).setdefault("flat_logdensity", {})

    if cache_key not in cache:
        logdensity_2arg = fitter._get_or_build_logdensity_fn()
        _, unravel_fn = ravel_pytree(init_params)

        def log_posterior_flat_2arg(position, data_args):
            return logdensity_2arg(unravel_fn(position), data_args)

        cache[cache_key] = (
            log_posterior_flat_2arg,
            unravel_fn,
        )

    logdensity_flat, unravel_fn = cache[cache_key]
    init_flat, _ = ravel_pytree(init_params)
    return logdensity_flat, unravel_fn, init_flat, fitter._data_args


def _get_cached_adaptation(fitter, method_key):
    """Return cached adaptation params for *method_key*, or None.

    Cached on the **Model** (not fitter) keyed by
    ``(engine_cache_key, method_key)`` so that adaptation results
    persist across Fitters sharing the same model structure.
    """
    mc = get_model_cache(fitter.model)
    cache = mc.get("adaptation")
    if cache is None:
        return None
    engine_key = fitter._engine_cache_key()
    return cache.get((engine_key, method_key))


def _set_cached_adaptation(fitter, method_key, params):
    """Store adaptation params on the **Model** for cross-fitter reuse."""
    cache = get_model_cache(fitter.model).setdefault("adaptation", {})
    engine_key = fitter._engine_cache_key()
    cache[(engine_key, method_key)] = params


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

    if init_from is not None:
        init_params = fitter._unbounded_from_posterior(init_from)
    else:
        init_params = fitter._initialize_unbounded(key)

    log_prob_flat_2arg, unravel_fn, init_flat, data_args = _get_flat_logdensity(
        fitter,
        init_params,
    )

    def log_prob_flat(pos):
        return log_prob_flat_2arg(pos, data_args)

    D = len(init_flat)

    if step_size is None:
        # Behroozi (2025) recommends 0.03 * sqrt(D), but for
        # stochastic SFH models the psd_xi variables create a
        # tighter curvature. Use a smaller default for D > 10.
        if D <= 10:
            step_size = 0.03 * jnp.sqrt(float(D))
        else:
            step_size = 0.01

    total_steps = n_burnin + n_steps

    if verbose:
        logger.info(
            "Ray Tracing: %d params, %d burn-in + %d samples, %d leapfrog/step, step_size=%.4f",
            D, n_burnin, n_steps, n_leapfrog_steps, float(step_size)
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

    samples_phys = _vmap_samples_to_physical(chain, unravel_fn, fitter._to_physical)
    best_params = _mean_params(samples_phys)

    if verbose:
        logger.info(
            "  Ray Tracing complete in %.1fs. Acceptance: %.1f%% (overall), "
            "%.1f%% (post burn-in). Samples: %d",
            wall_time, mean_accept * 100, mean_accept_post * 100, n_samples_out
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

    init_params, key = _maybe_map_init(fitter, key, init_from, verbose)

    log_posterior_flat_2arg, unravel_fn, init_flat, data_args = _get_flat_logdensity(
        fitter,
        init_params,
    )

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
        logger.info(
            "NUTS: %d parameters, %d warmup%s, %d samples, target_accept=%.2f",
            n_dim, n_warmup, burnin_msg, n_samples, target_accept_rate
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
            logger.info(
                "  Auto-switching to diagonal mass matrix (D=%d>30). "
                "Dense would be O(D²)=%d per step.",
                n_dim, n_dim**2
            )

    adapt_key = ("nuts", not use_dense)
    cached = _get_cached_adaptation(fitter, adapt_key)
    if cached is not None:
        parameters = cached

        def ld_1arg(pos):
            return log_posterior_flat_2arg(pos, data_args)

        state = blackjax.mcmc.nuts.init(init_flat, ld_1arg)
        if verbose:
            logger.info(
                "  Reusing cached warmup (%.1fs). Step size: %.4f",
                time.time() - t0, float(parameters['step_size'])
            )
    else:
        key, warmup_key = jax.random.split(key)

        def ld_1arg(pos):
            return log_posterior_flat_2arg(pos, data_args)

        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            ld_1arg,
            is_mass_matrix_diagonal=not use_dense,
            target_acceptance_rate=target_accept_rate,
        )
        (state, parameters), _ = warmup.run(warmup_key, init_flat, num_steps=n_warmup)
        _set_cached_adaptation(fitter, adapt_key, parameters)
        if verbose:
            logger.info(
                "  Warmup complete (%.1fs). Step size: %.4f",
                time.time() - t0, float(parameters['step_size'])
            )

    step_size = parameters["step_size"]
    inv_mass_matrix = parameters["inverse_mass_matrix"]

    # Burn-in via cached scan (discarded)
    if n_burnin > 0:
        key, burnin_key = jax.random.split(key)
        burnin_keys = jax.random.split(burnin_key, n_burnin)
        state = _nuts_burnin_scan(
            state,
            burnin_keys,
            log_posterior_flat_2arg,
            step_size,
            inv_mass_matrix,
            max_num_doublings,
            data_args,
        )
        if verbose:
            logger.info("  Burn-in complete (%d steps discarded)", n_burnin)

    key, sample_key = jax.random.split(key)
    sample_keys = jax.random.split(sample_key, n_samples)

    _, (positions, divergent) = _nuts_sample_scan(
        state,
        sample_keys,
        log_posterior_flat_2arg,
        step_size,
        inv_mass_matrix,
        max_num_doublings,
        data_args,
    )
    n_divergent = int(jnp.sum(divergent))

    wall_time = time.time() - t0

    if verbose:
        logger.info("  Sampling complete (%d samples)", n_samples)

    samples_phys = _vmap_samples_to_physical(positions, unravel_fn, fitter._to_physical)
    best_params = _mean_params(samples_phys)

    if verbose:
        logger.info(
            "  NUTS complete in %.1fs. Divergences: %d/%d",
            wall_time, n_divergent, n_samples
        )

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


def run_hmc(
    fitter,
    *,
    key,
    init_from=None,
    n_warmup=300,
    n_burnin=100,
    n_samples=1000,
    n_leapfrog_steps=10,
    target_accept_rate=0.85,
    dense_mass_matrix=True,
    verbose=True,
):
    """HMC sampling via BlackJAX.

    Standard Hamiltonian Monte Carlo with fixed trajectory length.
    Predictable cost per step (no tree building), making it faster
    than NUTS per sample when the geometry is well-conditioned.

    Parameters
    ----------
    n_warmup : int
        Warmup/adaptation steps (tunes step size and mass matrix).
    n_burnin : int
        Post-warmup burn-in steps (discarded).
    n_samples : int
        Posterior samples to collect.
    n_leapfrog_steps : int
        Number of leapfrog integration steps per proposal.
    target_accept_rate : float
        Target acceptance rate for step size adaptation.
    dense_mass_matrix : bool
        Use dense mass matrix. Set False for D>30.
    verbose : bool
        Print progress.
    """
    try:
        import blackjax
    except ImportError:
        raise ImportError("blackjax required for HMC: pip install blackjax") from None

    from tengri.inference.posterior import Posterior

    init_params, key = _maybe_map_init(fitter, key, init_from, verbose)

    log_posterior_flat_2arg, unravel_fn, init_flat, data_args = _get_flat_logdensity(
        fitter,
        init_params,
    )
    n_dim = len(init_flat)

    use_dense = dense_mass_matrix and n_dim <= 30

    if verbose:
        burnin_msg = f", {n_burnin} burn-in" if n_burnin > 0 else ""
        logger.info(
            "HMC: %d parameters, %d warmup%s, %d samples, %d leapfrog/step",
            n_dim, n_warmup, burnin_msg, n_samples, n_leapfrog_steps
        )

    t0 = time.time()

    adapt_key = ("hmc", not use_dense)
    cached = _get_cached_adaptation(fitter, adapt_key)

    def ld_1arg(pos):
        return log_posterior_flat_2arg(pos, data_args)

    if cached is not None:
        parameters = cached
        state = blackjax.mcmc.hmc.init(init_flat, ld_1arg)
        if verbose:
            logger.info(
                "  Reusing cached warmup (%.1fs). Step size: %.4f",
                time.time() - t0, float(parameters['step_size'])
            )
    else:
        key, warmup_key = jax.random.split(key)
        warmup = blackjax.window_adaptation(
            blackjax.hmc,
            ld_1arg,
            is_mass_matrix_diagonal=not use_dense,
            target_acceptance_rate=target_accept_rate,
            num_integration_steps=n_leapfrog_steps,
        )
        (state, parameters), _ = warmup.run(warmup_key, init_flat, num_steps=n_warmup)
        _set_cached_adaptation(fitter, adapt_key, parameters)
        if verbose:
            logger.info(
                "  Warmup complete (%.1fs). Step size: %.4f",
                time.time() - t0, float(parameters['step_size'])
            )

    step_size = parameters["step_size"]
    inv_mass_matrix = parameters["inverse_mass_matrix"]

    if n_burnin > 0:
        key, burnin_key = jax.random.split(key)
        burnin_keys = jax.random.split(burnin_key, n_burnin)
        state = _hmc_burnin_scan(
            state,
            burnin_keys,
            log_posterior_flat_2arg,
            step_size,
            inv_mass_matrix,
            n_leapfrog_steps,
            data_args,
        )
        if verbose:
            logger.info("  Burn-in complete (%d steps discarded)", n_burnin)

    key, sample_key = jax.random.split(key)
    sample_keys = jax.random.split(sample_key, n_samples)

    _, (positions, divergent) = _hmc_sample_scan(
        state,
        sample_keys,
        log_posterior_flat_2arg,
        step_size,
        inv_mass_matrix,
        n_leapfrog_steps,
        data_args,
    )
    n_divergent = int(jnp.sum(divergent))

    wall_time = time.time() - t0

    samples_phys = _vmap_samples_to_physical(positions, unravel_fn, fitter._to_physical)
    best_params = _mean_params(samples_phys)

    if verbose:
        logger.info(
            "  HMC complete in %.1fs. Divergences: %d/%d",
            wall_time, n_divergent, n_samples
        )

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="HMC (BlackJAX)",
        wall_time_s=wall_time,
        diagnostics={
            "n_warmup": n_warmup,
            "n_burnin": n_burnin,
            "n_samples": n_samples,
            "n_leapfrog_steps": n_leapfrog_steps,
            "n_divergent": n_divergent,
            "step_size": float(parameters["step_size"]),
        },
        loss_history=None,
        _model=fitter.model,
    )


def run_dynamic_hmc(
    fitter,
    *,
    key,
    init_from=None,
    n_warmup=300,
    n_burnin=100,
    n_samples=1000,
    target_accept_rate=0.85,
    dense_mass_matrix=True,
    verbose=True,
):
    """Dynamic HMC sampling via BlackJAX.

    HMC with dynamic trajectory length selection — adapts the number
    of leapfrog steps per proposal based on a heuristic that balances
    exploration vs cost. Similar to NUTS but without the binary tree.

    Parameters
    ----------
    n_warmup : int
        Warmup/adaptation steps.
    n_burnin : int
        Post-warmup burn-in steps (discarded).
    n_samples : int
        Posterior samples to collect.
    target_accept_rate : float
        Target acceptance rate for step size adaptation.
    dense_mass_matrix : bool
        Use dense mass matrix. Set False for D>30.
    verbose : bool
        Print progress.
    """
    try:
        import blackjax
    except ImportError:
        raise ImportError("blackjax required: pip install blackjax") from None

    from tengri.inference.posterior import Posterior

    init_params, key = _maybe_map_init(fitter, key, init_from, verbose)

    log_posterior_flat_2arg, unravel_fn, init_flat, data_args = _get_flat_logdensity(
        fitter,
        init_params,
    )
    n_dim = len(init_flat)

    use_dense = dense_mass_matrix and n_dim <= 30

    if verbose:
        burnin_msg = f", {n_burnin} burn-in" if n_burnin > 0 else ""
        logger.info(
            "Dynamic HMC: %d parameters, %d warmup%s, %d samples",
            n_dim, n_warmup, burnin_msg, n_samples
        )

    t0 = time.time()

    # dynamic_hmc.init needs random_generator_arg, incompatible with
    # window_adaptation. Use HMC warmup to tune step_size/mass matrix,
    # then initialize dynamic_hmc state separately.
    adapt_key = ("hmc", not use_dense)
    cached = _get_cached_adaptation(fitter, adapt_key)

    def ld_1arg(pos):
        return log_posterior_flat_2arg(pos, data_args)

    if cached is not None:
        parameters = cached
        if verbose:
            logger.info(
                "  Reusing cached warmup (%.1fs). Step size: %.4f",
                time.time() - t0, float(parameters['step_size'])
            )
    else:
        key, warmup_key = jax.random.split(key)
        warmup = blackjax.window_adaptation(
            blackjax.hmc,
            ld_1arg,
            is_mass_matrix_diagonal=not use_dense,
            target_acceptance_rate=target_accept_rate,
            num_integration_steps=10,
        )
        (_, parameters), _ = warmup.run(warmup_key, init_flat, num_steps=n_warmup)
        _set_cached_adaptation(fitter, adapt_key, parameters)
        if verbose:
            logger.info(
                "  Warmup complete (%.1fs). Step size: %.4f",
                time.time() - t0, float(parameters['step_size'])
            )

    step_size = parameters["step_size"]
    inv_mass_matrix = parameters["inverse_mass_matrix"]

    key, init_key = jax.random.split(key)
    state = blackjax.mcmc.dynamic_hmc.init(init_flat, ld_1arg, init_key)

    if n_burnin > 0:
        key, burnin_key = jax.random.split(key)
        burnin_keys = jax.random.split(burnin_key, n_burnin)
        state = _dynamic_hmc_burnin_scan(
            state,
            burnin_keys,
            log_posterior_flat_2arg,
            step_size,
            inv_mass_matrix,
            data_args,
        )
        if verbose:
            logger.info("  Burn-in complete (%d steps discarded)", n_burnin)

    key, sample_key = jax.random.split(key)
    sample_keys = jax.random.split(sample_key, n_samples)

    _, (positions, divergent) = _dynamic_hmc_sample_scan(
        state,
        sample_keys,
        log_posterior_flat_2arg,
        step_size,
        inv_mass_matrix,
        data_args,
    )
    n_divergent = int(jnp.sum(divergent))

    wall_time = time.time() - t0

    samples_phys = _vmap_samples_to_physical(positions, unravel_fn, fitter._to_physical)
    best_params = _mean_params(samples_phys)

    if verbose:
        logger.info(
            "  Dynamic HMC complete in %.1fs. Divergences: %d/%d",
            wall_time, n_divergent, n_samples
        )

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="Dynamic HMC (BlackJAX)",
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


def run_ghmc(
    fitter,
    *,
    key,
    init_from=None,
    n_warmup=300,
    n_burnin=100,
    n_samples=1000,
    alpha=0.8,
    delta=0.65,
    target_accept_rate=0.85,
    dense_mass_matrix=True,
    verbose=True,
):
    """Generalized HMC (GHMC) sampling via BlackJAX.

    GHMC uses partial momentum refreshment controlled by ``alpha``:
    at each step, the momentum is mixed with fresh noise as
    ``p_new = alpha * p_old + sqrt(1-alpha²) * noise``. This creates
    persistent chains that remember their direction, improving mixing
    in elongated posteriors like the age-dust-metallicity banana.

    Parameters
    ----------
    n_warmup : int
        Warmup/adaptation steps.
    n_burnin : int
        Post-warmup burn-in steps (discarded).
    n_samples : int
        Posterior samples to collect.
    alpha : float
        Momentum persistence parameter (0-1). 0 = full refresh (standard HMC),
        1 = no refresh (deterministic). Default 0.8 gives good mixing for
        correlated posteriors.
    delta : float
        Step size scaling in the GHMC proposal. Default 0.65.
    target_accept_rate : float
        Target acceptance rate for warmup adaptation.
    dense_mass_matrix : bool
        Use dense mass matrix. Set False for D>30.
    verbose : bool
        Print progress.
    """
    try:
        import blackjax
    except ImportError:
        raise ImportError("blackjax required: pip install blackjax") from None

    from tengri.inference.posterior import Posterior

    init_params, key = _maybe_map_init(fitter, key, init_from, verbose)

    log_posterior_flat_2arg, unravel_fn, init_flat, data_args = _get_flat_logdensity(
        fitter,
        init_params,
    )
    n_dim = len(init_flat)

    if verbose:
        burnin_msg = f", {n_burnin} burn-in" if n_burnin > 0 else ""
        logger.info(
            "GHMC: %d parameters, %d warmup%s, %d samples, alpha=%.1f, delta=%.2f",
            n_dim, n_warmup, burnin_msg, n_samples, alpha, delta
        )

    t0 = time.time()

    # GHMC's momentum generator treats momentum_inverse_scale as a
    # diagonal vector, so we must use diagonal mass matrix regardless
    # of the dense_mass_matrix flag.
    adapt_key = ("hmc", True)  # always diagonal for GHMC
    cached = _get_cached_adaptation(fitter, adapt_key)

    def ld_1arg(pos):
        return log_posterior_flat_2arg(pos, data_args)

    if cached is not None:
        parameters = cached
        if verbose:
            logger.info(
                "  Reusing cached warmup (%.1fs). Step size: %.4f",
                time.time() - t0, float(parameters['step_size'])
            )
    else:
        key, warmup_key = jax.random.split(key)
        warmup_hmc = blackjax.window_adaptation(
            blackjax.hmc,
            ld_1arg,
            is_mass_matrix_diagonal=True,
            target_acceptance_rate=target_accept_rate,
            num_integration_steps=10,
        )
        (_, parameters), _ = warmup_hmc.run(warmup_key, init_flat, num_steps=n_warmup)
        _set_cached_adaptation(fitter, adapt_key, parameters)
        if verbose:
            logger.info(
                "  Warmup complete (%.1fs). Step size: %.4f",
                time.time() - t0, float(parameters['step_size'])
            )

    step_size = parameters["step_size"]
    momentum_inv_scale = parameters["inverse_mass_matrix"]

    key, ghmc_init_key = jax.random.split(key)
    state = blackjax.mcmc.ghmc.init(init_flat, ghmc_init_key, ld_1arg)

    if n_burnin > 0:
        key, burnin_key = jax.random.split(key)
        burnin_keys = jax.random.split(burnin_key, n_burnin)
        state = _ghmc_burnin_scan(
            state,
            burnin_keys,
            log_posterior_flat_2arg,
            step_size,
            momentum_inv_scale,
            alpha,
            delta,
            data_args,
        )
        if verbose:
            logger.info("  Burn-in complete (%d steps discarded)", n_burnin)

    key, sample_key = jax.random.split(key)
    sample_keys = jax.random.split(sample_key, n_samples)

    _, (positions, divergent) = _ghmc_sample_scan(
        state,
        sample_keys,
        log_posterior_flat_2arg,
        step_size,
        momentum_inv_scale,
        alpha,
        delta,
        data_args,
    )
    n_divergent = int(jnp.sum(divergent))

    wall_time = time.time() - t0

    samples_phys = _vmap_samples_to_physical(positions, unravel_fn, fitter._to_physical)
    best_params = _mean_params(samples_phys)

    if verbose:
        logger.info(
            "  GHMC complete in %.1fs. Divergences: %d/%d",
            wall_time, n_divergent, n_samples
        )

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="GHMC (BlackJAX)",
        wall_time_s=wall_time,
        diagnostics={
            "n_warmup": n_warmup,
            "n_burnin": n_burnin,
            "n_samples": n_samples,
            "alpha": alpha,
            "delta": delta,
            "n_divergent": n_divergent,
            "step_size": float(step_size),
        },
        loss_history=None,
        _model=fitter.model,
    )


def run_mclmc(
    fitter,
    *,
    key,
    init_from=None,
    n_warmup=500,
    n_samples=2000,
    verbose=True,
):
    """Microcanonical Langevin Monte Carlo (MCLMC) via BlackJAX.

    State-of-the-art gradient-based sampler. Operates on the
    microcanonical manifold (constant energy), producing one
    sample per gradient evaluation with no accept/reject step.
    Often 10-100x more efficient than NUTS for smooth targets.

    Because there is no Metropolis correction, MCLMC is **biased**
    for finite step sizes. Use ``adjusted_mclmc`` for exact sampling.

    Parameters
    ----------
    n_warmup : int
        Warmup steps for tuning L (trajectory length) and step size.
    n_samples : int
        Posterior samples to collect.
    verbose : bool
        Print progress.
    """
    try:
        import blackjax
    except ImportError:
        raise ImportError("blackjax required: pip install blackjax") from None

    from tengri.inference.posterior import Posterior

    init_params, key = _maybe_map_init(fitter, key, init_from, verbose)

    log_posterior_flat_2arg, unravel_fn, init_flat, data_args = _get_flat_logdensity(
        fitter,
        init_params,
    )

    def ld_1arg(pos):
        return log_posterior_flat_2arg(pos, data_args)

    n_dim = len(init_flat)

    if verbose:
        logger.info("MCLMC: %d parameters, %d warmup, %d samples", n_dim, n_warmup, n_samples)

    t0 = time.time()

    def kernel_factory(inv_mass):
        return blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=ld_1arg,
            inverse_mass_matrix=inv_mass,
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        )

    cached = _get_cached_adaptation(fitter, "mclmc")
    if cached is not None:
        params = cached
        kernel = kernel_factory(params.inverse_mass_matrix)
        key, init_key = jax.random.split(key)
        state = blackjax.mcmc.mclmc.init(init_flat, ld_1arg, init_key)
        if verbose:
            logger.info(
                "  Reusing cached warmup (%.1fs). L=%.4f, step_size=%.4f",
                time.time() - t0, float(params.L), float(params.step_size)
            )
    else:
        key, init_key = jax.random.split(key)
        state = blackjax.mcmc.mclmc.init(init_flat, ld_1arg, init_key)

        key, tune_key = jax.random.split(key)
        state, params, _ = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel_factory,
            num_steps=n_warmup,
            state=state,
            rng_key=tune_key,
            diagonal_preconditioning=True,
        )

        kernel = kernel_factory(params.inverse_mass_matrix)
        _set_cached_adaptation(fitter, "mclmc", params)

        if verbose:
            logger.info(
                "  Warmup complete (%.1fs). L=%.4f, step_size=%.4f",
                time.time() - t0, float(params.L), float(params.step_size)
            )

    key, sample_key = jax.random.split(key)
    sample_keys = jax.random.split(sample_key, n_samples)

    _, positions = _mclmc_sample_scan(
        state,
        sample_keys,
        kernel,
        params.L,
        params.step_size,
    )

    wall_time = time.time() - t0

    samples_phys = _vmap_samples_to_physical(positions, unravel_fn, fitter._to_physical)
    best_params = _mean_params(samples_phys)

    if verbose:
        logger.info("  MCLMC complete in %.1fs (%d samples)", wall_time, n_samples)

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="MCLMC (BlackJAX)",
        wall_time_s=wall_time,
        diagnostics={
            "n_warmup": n_warmup,
            "n_samples": n_samples,
            "L": float(params.L),
            "step_size": float(params.step_size),
        },
        loss_history=None,
        _model=fitter.model,
    )


def run_adjusted_mclmc(
    fitter,
    *,
    key,
    init_from=None,
    n_warmup=500,
    n_samples=2000,
    target_accept_rate=0.65,
    verbose=True,
):
    """Adjusted MCLMC (Metropolis-corrected) via BlackJAX.

    Exact (unbiased) version of MCLMC. Adds a Metropolis correction
    step that guarantees detailed balance, making the chain converge
    to the exact target distribution regardless of step size.

    Slightly less efficient than vanilla MCLMC per step (due to the
    accept/reject overhead), but eliminates the step-size-dependent
    bias.

    Parameters
    ----------
    n_warmup : int
        Warmup steps for tuning step size and trajectory length.
    n_samples : int
        Posterior samples to collect.
    target_accept_rate : float
        Target Metropolis acceptance rate. Default 0.65 balances
        bias correction with mixing efficiency.
    verbose : bool
        Print progress.
    """
    try:
        import blackjax
    except ImportError:
        raise ImportError("blackjax required: pip install blackjax") from None

    from tengri.inference.posterior import Posterior

    init_params, key = _maybe_map_init(fitter, key, init_from, verbose)

    log_posterior_flat_2arg, unravel_fn, init_flat, data_args = _get_flat_logdensity(
        fitter,
        init_params,
    )

    def ld_1arg(pos):
        return log_posterior_flat_2arg(pos, data_args)

    n_dim = len(init_flat)

    if verbose:
        logger.info(
            "Adjusted MCLMC: %d parameters, %d warmup, %d samples, target_accept=%.2f",
            n_dim, n_warmup, n_samples, target_accept_rate
        )

    t0 = time.time()

    cached = _get_cached_adaptation(fitter, "adjusted_mclmc")
    if cached is not None:
        params = cached
        kernel = blackjax.mcmc.adjusted_mclmc.build_kernel(
            logdensity_fn=ld_1arg,
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
            inverse_mass_matrix=params.inverse_mass_matrix,
        )
        state = blackjax.mcmc.adjusted_mclmc.init(init_flat, ld_1arg)
        if verbose:
            logger.info(
                "  Reusing cached warmup (%.1fs). L=%.4f, step_size=%.4f",
                time.time() - t0, float(params.L), float(params.step_size)
            )
    else:
        state = blackjax.mcmc.adjusted_mclmc.init(init_flat, ld_1arg)

        # Adaptation wrapper: blackjax 1.3 adaptation calls the kernel with
        # keyword args (rng_key, state, avg_num_integration_steps, step_size,
        # inverse_mass_matrix). We wrap build_kernel to match.
        def _kernel_for_adaptation(
            rng_key, state, avg_num_integration_steps, step_size, inverse_mass_matrix
        ):
            k = blackjax.mcmc.adjusted_mclmc.build_kernel(
                logdensity_fn=ld_1arg,
                integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
                inverse_mass_matrix=inverse_mass_matrix,
            )
            n_steps = jnp.ceil(avg_num_integration_steps).astype(int)
            return k(rng_key, state, step_size, n_steps)

        key, tune_key = jax.random.split(key)
        state, params, _ = blackjax.adjusted_mclmc_find_L_and_step_size(
            mclmc_kernel=_kernel_for_adaptation,
            num_steps=n_warmup,
            state=state,
            rng_key=tune_key,
            target=target_accept_rate,
            diagonal_preconditioning=True,
        )

        kernel = blackjax.mcmc.adjusted_mclmc.build_kernel(
            logdensity_fn=ld_1arg,
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
            inverse_mass_matrix=params.inverse_mass_matrix,
        )
        _set_cached_adaptation(fitter, "adjusted_mclmc", params)

        if verbose:
            logger.info(
                "  Warmup complete (%.1fs). L=%.4f, step_size=%.4f",
                time.time() - t0, float(params.L), float(params.step_size)
            )

    L = params.L
    step_size = params.step_size
    n_integration_steps = jnp.ceil(L / step_size).astype(int)

    key, sample_key = jax.random.split(key)
    sample_keys = jax.random.split(sample_key, n_samples)

    _, (positions, divergent) = _adjusted_mclmc_sample_scan(
        state,
        sample_keys,
        kernel,
        step_size,
        n_integration_steps,
    )
    n_divergent = int(jnp.sum(divergent))

    wall_time = time.time() - t0

    samples_phys = _vmap_samples_to_physical(positions, unravel_fn, fitter._to_physical)
    best_params = _mean_params(samples_phys)

    if verbose:
        logger.info(
            "  Adjusted MCLMC complete in %.1fs. Divergences: %d/%d",
            wall_time, n_divergent, n_samples
        )

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="Adjusted MCLMC (BlackJAX)",
        wall_time_s=wall_time,
        diagnostics={
            "n_warmup": n_warmup,
            "n_samples": n_samples,
            "n_divergent": n_divergent,
            "L": float(params.L),
            "step_size": float(params.step_size),
            "n_integration_steps": int(n_integration_steps),
        },
        loss_history=None,
        _model=fitter.model,
    )


# ── Elliptical Slice Sampling ─────────────────────────────────────


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
