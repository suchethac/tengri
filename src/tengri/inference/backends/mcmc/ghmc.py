# SPDX-License-Identifier: BSD-3-Clause
"""Generalized HMC via BlackJAX.

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
    _ghmc_chain_scan,
    _ghmc_full_scan,
    _set_cached_adaptation,
    _vmap_chains,
)

logger = logging.getLogger(__name__)


def run_ghmc(
    context,
    *,
    key,
    init_from=None,
    n_warmup=300,
    n_burnin=100,
    n_samples=1000,
    n_chains=1,
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
    n_warmup: int
        Warmup/adaptation steps.
    n_burnin: int
        Post-warmup burn-in steps (discarded).
    n_samples: int
        Posterior samples to collect.
    alpha: float
        Momentum persistence parameter (0-1). 0 = full refresh (standard HMC),
        1 = no refresh (deterministic). Default 0.8 gives good mixing for
        correlated posteriors.
    delta: float
        Step size scaling in the GHMC proposal. Default 0.65.
    target_accept_rate: float
        Target acceptance rate for warmup adaptation.
    dense_mass_matrix: bool
        Use dense mass matrix. Set False for D>30.
    verbose: bool
        Print progress.
    """
    try:
        import blackjax
    except ImportError:
        raise ImportError("blackjax required: pip install blackjax") from None

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

    if verbose:
        burnin_msg = f", {n_burnin} burn-in" if n_burnin > 0 else ""
        logger.info(
            "GHMC: %d parameters, %d warmup%s, %d samples, alpha=%.1f, delta=%.2f",
            n_dim,
            n_warmup,
            burnin_msg,
            n_samples,
            alpha,
            delta,
        )

    t0 = time.time()

    # GHMC's momentum generator treats momentum_inverse_scale as a
    # diagonal vector, so we must use diagonal mass matrix regardless
    # of the dense_mass_matrix flag.
    # Namespaced to this backend, not "hmc": the cache is keyed by tuple, so borrowing
    # another sampler's prefix is a collision waiting on the next field either side
    # adds. GHMC tunes a different kernel and must not inherit HMC's step size.
    # n_warmup, alpha, delta and target_accept_rate belong in the key: they
    # *produce* the adaptation, so leaving them out makes those knobs silently
    # inert on a model that already holds an entry.
    #
    # The trailing ``True`` pins "always diagonal" and must stay both LAST and on
    # this one line: #1454 asserts on this statement as text, matching up to the
    # first ``)`` and reading the final element. That is what keeps GHMC out of
    # the dense-mass advisory, so the tuning rides in a single grouped element
    # rather than being spliced in after it.
    tuning = (int(n_warmup), float(alpha), float(delta), float(target_accept_rate))
    adapt_key = ("ghmc", tuning, True)
    cached = _get_cached_adaptation(fitter, adapt_key)

    # Both branches must advance the key identically, cache presence is
    # invisible to the caller and must not steer the RNG stream, or two
    # identical ``fit`` calls with one ``key`` return different chains. These
    # two splits used to live only in the ``else``; they are unused on the
    # cached path but keep ``chain_key`` on the same stream position.
    key, warmup_key = jax.random.split(key)
    key, ghmc_init_key = jax.random.split(key)

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

            def _init(p, init_key):
                # Keyword args: blackjax reordered ghmc.init's (rng_key,
                # logdensity_fn) between 1.3 and 1.6, keywords work on both.
                return blackjax.mcmc.ghmc.init(position=p, logdensity_fn=ld_1arg, rng_key=init_key)

            def _scan(s, ks):
                return _ghmc_chain_scan(
                    s,
                    ks,
                    log_posterior_flat_2arg,
                    data_args,
                    parameters["step_size"],
                    parameters["inverse_mass_matrix"],
                    alpha,
                    delta,
                    alpha,
                    delta,
                )

            positions, divergent = _vmap_chains(
                _init,
                _scan,
                init_flat=init_flat,
                chain_key=chain_key,
                n_chains=n_chains,
                n_iter=n_burnin + n_samples,
                n_burnin=n_burnin,
            )
            _multichain_burnin_done = True
        else:
            key, ghmc_init_key = jax.random.split(chain_key)
            state = blackjax.mcmc.ghmc.init(
                position=init_flat, logdensity_fn=ld_1arg, rng_key=ghmc_init_key
            )
            chain_keys = jax.random.split(key, n_burnin + n_samples)
            positions, divergent = _ghmc_chain_scan(
                state,
                chain_keys,
                log_posterior_flat_2arg,
                data_args,
                parameters["step_size"],
                parameters["inverse_mass_matrix"],
                alpha,
                delta,
                alpha,
                delta,
            )
            _multichain_burnin_done = False
    else:
        _multichain_burnin_done = False
        key, chain_key = jax.random.split(key)
        chain_keys = jax.random.split(chain_key, n_burnin + n_samples)
        positions, divergent, step_size, momentum_inv_scale = _ghmc_full_scan(
            init_flat,
            warmup_key,
            ghmc_init_key,
            chain_keys,
            log_posterior_flat_2arg,
            data_args,
            n_warmup,
            target_accept_rate,
            alpha,
            delta,
        )
        parameters = {"step_size": step_size, "inverse_mass_matrix": momentum_inv_scale}
        _set_cached_adaptation(fitter, adapt_key, parameters)
        if verbose:
            logger.info(
                "  Warmup + chain complete (%.1fs). Step size: %.4f",
                time.time() - t0,
                float(step_size),
            )

    # Burnin discarded Python-side. Multichain branch already handled
    # per-chain burnin before flatten; skip the global slice there.
    if n_burnin > 0 and not _multichain_burnin_done:
        positions = positions[n_burnin:]
        divergent = divergent[n_burnin:]
    n_divergent = int(jnp.sum(divergent))

    wall_time = time.time() - t0

    samples_phys = _vmap_samples_to_physical(positions, unravel_fn, context.to_physical)
    best_params = _mean_params(samples_phys)

    if verbose:
        logger.info(
            "  GHMC complete in %.1fs. Divergences: %d/%d", wall_time, n_divergent, n_samples
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
            "n_chains": n_chains,
            "alpha": alpha,
            "delta": delta,
            "n_divergent": n_divergent,
            "step_size": float(parameters["step_size"]),
        },
        loss_history=None,
        _model=context.model,
    )
