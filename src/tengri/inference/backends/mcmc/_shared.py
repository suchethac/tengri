"""Shared MCMC infrastructure: kernel getters, scan functions, logdensity helpers.

Internal — imported by per-sampler modules. Not part of the public API.
"""

from __future__ import annotations

import functools

import jax
from jax.flatten_util import ravel_pytree

from tengri.inference._model_cache import get_model_cache


def _get_nuts_kernel():
    """Retrieve the NUTS kernel from BlackJAX."""
    import blackjax.mcmc.nuts

    return blackjax.mcmc.nuts.build_kernel()


@functools.cache
def _get_hmc_kernel():
    """Retrieve and cache the HMC kernel from BlackJAX."""
    import blackjax.mcmc.hmc

    return blackjax.mcmc.hmc.build_kernel()


@functools.cache
def _get_dynamic_hmc_kernel():
    """Retrieve and cache the dynamic HMC kernel from BlackJAX."""
    import blackjax.mcmc.dynamic_hmc

    return blackjax.mcmc.dynamic_hmc.build_kernel()


@functools.cache
def _get_ghmc_kernel():
    """Retrieve and cache the GHMC kernel from BlackJAX."""
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
    """JIT-compiled NUTS sampling scan over multiple steps."""
    kernel = _get_nuts_kernel()

    def ld(pos):
        """Bind log-density function with data arguments."""
        return logdensity_fn_2arg(pos, data_args)

    def _step(s, k):
        """Execute one NUTS step, returning state and sample."""
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
    """JIT-compiled NUTS burn-in scan (samples discarded)."""
    kernel = _get_nuts_kernel()

    def ld(pos):
        """Bind log-density function with data arguments."""
        return logdensity_fn_2arg(pos, data_args)

    def _step(s, k):
        """Execute one NUTS step, returning updated state."""
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
    """JIT-compiled HMC sampling scan over multiple steps."""
    kernel = _get_hmc_kernel()

    def ld(pos):
        """Bind log-density function with data arguments."""
        return logdensity_fn_2arg(pos, data_args)

    def _step(s, k):
        """Execute one HMC step, returning state and sample."""
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
    """JIT-compiled HMC burn-in scan (samples discarded)."""
    kernel = _get_hmc_kernel()

    def ld(pos):
        """Bind log-density function with data arguments."""
        return logdensity_fn_2arg(pos, data_args)

    def _step(s, k):
        """Execute one HMC step, returning updated state."""
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
    """JIT-compiled dynamic HMC sampling scan over multiple steps."""
    kernel = _get_dynamic_hmc_kernel()

    def ld(pos):
        """Bind log-density function with data arguments."""
        return logdensity_fn_2arg(pos, data_args)

    def _step(s, k):
        """Execute one dynamic HMC step, returning state and sample."""
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
    """JIT-compiled dynamic HMC burn-in scan (samples discarded)."""
    kernel = _get_dynamic_hmc_kernel()

    def ld(pos):
        """Bind log-density function with data arguments."""
        return logdensity_fn_2arg(pos, data_args)

    def _step(s, k):
        """Execute one dynamic HMC step, returning updated state."""
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
    """JIT-compiled GHMC sampling scan over multiple steps."""
    kernel = _get_ghmc_kernel()

    def ld(pos):
        """Bind log-density function with data arguments."""
        return logdensity_fn_2arg(pos, data_args)

    def _step(s, k):
        """Execute one GHMC step, returning state and sample."""
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
    """JIT-compiled GHMC burn-in scan (samples discarded)."""
    kernel = _get_ghmc_kernel()

    def ld(pos):
        """Bind log-density function with data arguments."""
        return logdensity_fn_2arg(pos, data_args)

    def _step(s, k):
        """Execute one GHMC step, returning updated state."""
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
    """JIT-compiled MCLMC sampling scan over multiple steps."""
    def _step(s, k):
        """Execute one MCLMC step, returning state and sample."""
        s, _info = kernel(k, s, L, step_size)
        return s, s.position

    return jax.lax.scan(_step, state, keys)


@functools.partial(jax.jit, static_argnums=(2,))
def _adjusted_mclmc_sample_scan(state, keys, kernel, step_size, n_integration_steps):
    """JIT-compiled adjusted MCLMC sampling scan over multiple steps."""
    def _step(s, k):
        """Execute one adjusted MCLMC step, returning state and sample."""
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
    """Retrieve cached adaptation parameters by method key, or None if not cached."""
    mc = get_model_cache(fitter.model)
    cache = mc.get("adaptation")
    if cache is None:
        return None
    engine_key = fitter._engine_cache_key()
    return cache.get((engine_key, method_key))


def _set_cached_adaptation(fitter, method_key, params):
    """Store adaptation parameters on the Model for cross-fitter reuse."""
    cache = get_model_cache(fitter.model).setdefault("adaptation", {})
    engine_key = fitter._engine_cache_key()
    cache[(engine_key, method_key)] = params
