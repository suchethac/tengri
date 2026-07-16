# SPDX-License-Identifier: BSD-3-Clause
"""Standard Hamiltonian Monte Carlo via BlackJAX.

Extracted from mcmc/common.py. Import via ``tengri.inference.backends.mcmc``.
"""

from __future__ import annotations

import logging
import time
import warnings

import jax
import jax.numpy as jnp

from tengri.inference._sample_utils import _maybe_map_init, _mean_params, _vmap_samples_to_physical
from tengri.inference.backends.mcmc._shared import (
    _get_cached_adaptation,
    _get_flat_logdensity,
    _hmc_chain_scan,
    _hmc_warmup_only,
    _parallel_chains,
    _set_cached_adaptation,
    _vmap_chains,
)
from tengri.utils.compile_log import compile_timer

logger = logging.getLogger(__name__)


def _resolve_chain_runner(chain_method, n_chains):
    """Pick the multi-chain executor: vmap (SIMD-batched) or parallel (pmap).

    ``chain_method='parallel'`` runs each chain on its own device — a ~n_chains×
    wall-time win on CPU with enough forced host devices — but silently falls
    back to vmap (with a warning) when too few devices are visible, so a fit
    never fails just because ``XLA_FLAGS`` was not set.
    """
    if chain_method == "vmap":
        return _vmap_chains
    if chain_method == "parallel":
        if jax.device_count() >= n_chains:
            return _parallel_chains
        warnings.warn(
            f"chain_method='parallel' needs >= {n_chains} JAX devices, found "
            f"{jax.device_count()}; falling back to vmap. On CPU set "
            f"XLA_FLAGS=--xla_force_host_platform_device_count={n_chains} before importing "
            f"jax / tengri to enable true parallel chains.",
            RuntimeWarning,
            stacklevel=3,
        )
        return _vmap_chains
    raise ValueError(f"chain_method must be 'vmap' or 'parallel', got {chain_method!r}")


def run_hmc(
    context,
    *,
    key,
    init_from=None,
    n_warmup=300,
    n_burnin=100,
    n_samples=1000,
    n_chains=1,
    n_leapfrog_steps=10,
    target_accept_rate=0.85,
    dense_mass_matrix=True,
    chain_method="vmap",
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
        Post-warmup burn-in steps (discarded). Discarded Python-side
        rather than inside JIT, so changing this does NOT trigger a
        recompile when ``n_burnin + n_samples`` is unchanged.
    n_samples : int
        Posterior samples per chain to collect.
    n_chains : int, default 1
        Number of independent HMC chains, sharing one adapted step size and
        mass matrix. Warmup is adapted once; the chains then sample from
        jittered starts, honored on the **first** call as well as cached ones.
        Final posterior has ``n_chains * n_samples`` samples. Under the default
        ``chain_method="vmap"`` the chains are SIMD-batched, so wall scales ~
        linearly with ``n_chains`` on CPU.
    n_leapfrog_steps : int
        Number of leapfrog integration steps per proposal.
    target_accept_rate : float
        Target acceptance rate for step size adaptation.
    dense_mass_matrix : bool
        Use dense mass matrix. Set False for D>30.
    chain_method : {"vmap", "parallel"}, default "vmap"
        How ``n_chains > 1`` chains are executed. ``"vmap"`` SIMD-batches them
        into one device's kernel (cost ~ ``n_chains`` × one chain). ``"parallel"``
        maps them across physical devices via ``jax.pmap`` (chains run
        concurrently, ~one chain's wall) — on CPU this needs
        ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` set before
        importing jax; it falls back to ``"vmap"`` with a warning if fewer than
        ``n_chains`` devices are visible.
    verbose : bool
        Print progress.
    """
    try:
        import blackjax
    except ImportError:
        raise ImportError("blackjax required for HMC: pip install blackjax") from None

    from tengri.inference.context import InferenceContext
    from tengri.inference.posterior import Posterior

    # ``_shared.py`` helpers still take a Fitter; reach through
    # the context until they migrate.
    context = InferenceContext.from_target(context)
    fitter = context.fitter

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
            n_dim,
            n_warmup,
            burnin_msg,
            n_samples,
            n_leapfrog_steps,
        )

    t0 = time.time()

    adapt_key = ("hmc", not use_dense)
    cached = _get_cached_adaptation(fitter, adapt_key)

    def ld_1arg(pos):
        return log_posterior_flat_2arg(pos, data_args)

    # ── Warmup: adapt (step_size, inverse_mass_matrix) once, then cache. ──
    # Split from sampling so the FIRST call honors n_chains too (previously a
    # fresh multi-chain run silently sampled a single chain and mislabelled it).
    if cached is not None:
        parameters = cached
        if verbose:
            logger.info(
                "  Reusing cached warmup (%.1fs). Step size: %.4f",
                time.time() - t0,
                float(parameters["step_size"]),
            )
    else:
        key, warmup_key = jax.random.split(key)
        with compile_timer("hmc_warmup", fitter.compile_signature(), method="mcmc_hmc"):
            step_size, inv_mass_matrix = _hmc_warmup_only(
                init_flat,
                warmup_key,
                log_posterior_flat_2arg,
                data_args,
                n_warmup,
                n_leapfrog_steps,
                use_dense,
                target_accept_rate,
            )
            jax.block_until_ready(step_size)
        parameters = {"step_size": step_size, "inverse_mass_matrix": inv_mass_matrix}
        _set_cached_adaptation(fitter, adapt_key, parameters)
        if verbose:
            logger.info(
                "  Warmup complete (%.1fs). Step size: %.4f",
                time.time() - t0,
                float(parameters["step_size"]),
            )

    # ── Sampling: single- or multi-chain, honored on first AND cached calls. ──
    key, chain_key = jax.random.split(key)
    if n_chains > 1:

        def _init(p):
            return blackjax.mcmc.hmc.init(p, ld_1arg)

        def _scan(s, ks):
            return _hmc_chain_scan(
                s,
                ks,
                log_posterior_flat_2arg,
                data_args,
                parameters["step_size"],
                parameters["inverse_mass_matrix"],
                n_leapfrog_steps,
            )

        chain_runner = _resolve_chain_runner(chain_method, n_chains)
        with compile_timer("hmc_chain_scan_vmap", fitter.compile_signature(), method="mcmc_hmc"):
            positions, divergent = chain_runner(
                _init,
                _scan,
                init_flat=init_flat,
                chain_key=chain_key,
                n_chains=n_chains,
                n_iter=n_burnin + n_samples,
                n_burnin=n_burnin,
            )
            jax.block_until_ready(positions)
        _multichain_burnin_done = True
    else:
        state = blackjax.mcmc.hmc.init(init_flat, ld_1arg)
        chain_keys = jax.random.split(chain_key, n_burnin + n_samples)
        with compile_timer("hmc_chain_scan", fitter.compile_signature(), method="mcmc_hmc"):
            positions, divergent = _hmc_chain_scan(
                state,
                chain_keys,
                log_posterior_flat_2arg,
                data_args,
                parameters["step_size"],
                parameters["inverse_mass_matrix"],
                n_leapfrog_steps,
            )
            jax.block_until_ready(positions)
        _multichain_burnin_done = False

    # Burnin discard happens Python-side. Multichain branch already
    # discarded per-chain before flattening; skip the global slice there.
    if n_burnin > 0 and not _multichain_burnin_done:
        positions = positions[n_burnin:]
        divergent = divergent[n_burnin:]
    n_divergent = int(jnp.sum(divergent))

    wall_time = time.time() - t0

    samples_phys = _vmap_samples_to_physical(positions, unravel_fn, context.to_physical)
    best_params = _mean_params(samples_phys)

    if verbose:
        logger.info(
            "  HMC complete in %.1fs. Divergences: %d/%d", wall_time, n_divergent, n_samples
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
            "n_chains": n_chains,
            "n_leapfrog_steps": n_leapfrog_steps,
            "n_divergent": n_divergent,
            "step_size": float(parameters["step_size"]),
        },
        loss_history=None,
        _model=context.model,
    )
