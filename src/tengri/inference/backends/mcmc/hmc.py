# SPDX-License-Identifier: BSD-3-Clause
"""Standard Hamiltonian Monte Carlo via BlackJAX.

Extracted from mcmc/common.py. Import via ``tengri.inference.backends.mcmc``.
"""

from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp

from tengri.inference._sample_utils import _maybe_map_init, _mean_params, _vmap_samples_to_physical
from tengri.inference.backends.mcmc._shared import (
    _get_cached_adaptation,
    _get_flat_logdensity,
    _hmc_chain_scan,
    _hmc_full_scan,
    _set_cached_adaptation,
    _vmap_chains,
)
from tengri.utils.compile_log import compile_timer

logger = logging.getLogger(__name__)


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
        Number of independent HMC chains run in parallel via ``jax.vmap``,
        sharing the cached step size and mass matrix. Only honoured when
        the warmup is already cached (i.e. a previous ``run_hmc`` call
        populated the cache). Final posterior has ``n_chains * n_samples``
        samples; wall ≈ one chain's worth on CPU SIMD.
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

    if cached is not None:
        parameters = cached

        def ld_1arg(pos):
            return log_posterior_flat_2arg(pos, data_args)

        if verbose:
            logger.info(
                "  Reusing cached warmup (%.1fs). Step size: %.4f",
                time.time() - t0,
                float(parameters["step_size"]),
            )
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

            with compile_timer(
                "hmc_chain_scan_vmap", fitter.compile_signature(), method="mcmc_hmc"
            ):
                positions, divergent = _vmap_chains(
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
    else:
        _multichain_burnin_done = False
        key, warmup_key = jax.random.split(key)
        key, chain_key = jax.random.split(key)
        chain_keys = jax.random.split(chain_key, n_burnin + n_samples)
        with compile_timer("hmc_full_scan", fitter.compile_signature(), method="mcmc_hmc"):
            positions, divergent, step_size, inv_mass_matrix = _hmc_full_scan(
                init_flat,
                warmup_key,
                chain_keys,
                log_posterior_flat_2arg,
                data_args,
                n_warmup,
                n_leapfrog_steps,
                use_dense,
                target_accept_rate,
            )
            jax.block_until_ready(positions)
        parameters = {"step_size": step_size, "inverse_mass_matrix": inv_mass_matrix}
        _set_cached_adaptation(fitter, adapt_key, parameters)
        if verbose:
            logger.info(
                "  Warmup + chain complete (%.1fs). Step size: %.4f",
                time.time() - t0,
                float(step_size),
            )

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
