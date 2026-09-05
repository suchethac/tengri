# SPDX-License-Identifier: BSD-3-Clause
"""Dynamic HMC via BlackJAX.

Extracted from mcmc/common.py. Import via ``tengri.inference.backends.mcmc``.
"""

from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp

from tengri.inference._sample_utils import _maybe_map_init, _mean_params, _vmap_samples_to_physical
from tengri.inference.backends.mcmc._shared import (
    _dynamic_hmc_chain_scan,
    _get_cached_adaptation,
    _get_flat_logdensity,
    _hmc_warmup_only,
    _set_cached_adaptation,
    _vmap_chains,
    final_window_divergence_frac,
    refuse_dead_sampling,
    refuse_dead_warmup,
    sampling_diagnostics,
)
from tengri.inference.preconditioning import prepare_preconditioning

logger = logging.getLogger(__name__)

#: Integration steps used to *tune* dynamic HMC, not to sample with.
#:
#: ``dynamic_hmc.init`` needs a ``random_generator_arg`` that
#: ``window_adaptation`` cannot supply, so the step size and mass matrix come
#: from plain HMC window adaptation at this fixed trajectory length; the
#: dynamic kernel then draws its own per-step length while sampling. Must stay
#: equal to the ``num_integration_steps`` in
#: ``tengri.inference.backends.mcmc._shared._dynamic_hmc_full_scan``,
#: which the prewarm, hierarchical and catalog paths still call, the two
#: disagreeing would mean a cached adaptation tuned against a different
#: trajectory length than the fused path produces.
_DHMC_WARMUP_LEAPFROG_STEPS = 10


def run_dynamic_hmc(
    context,
    *,
    key,
    init_from=None,
    n_warmup=300,
    n_burnin=100,
    n_samples=1000,
    n_chains=1,
    target_accept_rate=0.85,
    dense_mass_matrix=True,
    precondition: bool | float | None = None,
    verbose=True,
):
    """Dynamic HMC sampling via BlackJAX.

    HMC with dynamic trajectory length selection, adapts the number
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
    precondition : bool, float or None, default None
        Sample in metric-whitened coordinates, mapping draws back afterwards.
        A linear change of variables, so the posterior is unchanged. **Opt-in**
        (#1397): ``None`` (default) and ``False`` are off; ``True`` uses
        :data:`~tengri.inference.preconditioning.DEFAULT_WHITENING_STRENGTH`, and
        a float in ``[0, 1]`` sets the whitening strength (``1.0`` is full
        whitening, which amplifies a misspecified metric without bound, #1442).
        See :func:`~tengri.inference.backends.mcmc.nuts.run_nuts` for the full
        rationale and :mod:`tengri.inference.preconditioning` for the math.
    verbose : bool
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

    # Metric preconditioning (#1301), see ``run_nuts`` for the rationale. Linear
    # change of variables, so the posterior is untouched; draws are mapped back below.
    problem = prepare_preconditioning(
        log_posterior_flat_2arg, init_flat, data_args, precondition=precondition
    )
    log_posterior_flat_2arg, init_flat = problem.logdensity, problem.init_flat

    n_dim = len(init_flat)

    use_dense = dense_mass_matrix and n_dim <= 30

    if verbose:
        burnin_msg = f", {n_burnin} burn-in" if n_burnin > 0 else ""
        logger.info(
            "Dynamic HMC: %d parameters, %d warmup%s, %d samples",
            n_dim,
            n_warmup,
            burnin_msg,
            n_samples,
        )

    t0 = time.time()

    # dynamic_hmc.init needs random_generator_arg, incompatible with
    # window_adaptation. Use HMC warmup to tune step_size/mass matrix,
    # then initialize dynamic_hmc state separately.
    # n_warmup and target_accept_rate belong in the key: they *produce* the
    # adaptation, so leaving them out makes both knobs silently inert on a model
    # that already holds an entry. Grouped into one element and kept on a single
    # line because the namespace guard in test_preconditioning.py reads this
    # statement as text, per line.
    tuning = (int(n_warmup), float(target_accept_rate))
    adapt_key = ("dynamic_hmc", not use_dense, tuning, problem.cache_key)
    cached = _get_cached_adaptation(fitter, adapt_key)

    # Both branches must advance the key identically, cache presence is
    # invisible to the caller and must not steer the RNG stream, or two
    # identical ``fit`` calls with one ``key`` return different chains. These
    # two splits used to live only in the ``else``; they are unused on the
    # cached path but keep ``chain_key`` on the same stream position.
    key, warmup_key = jax.random.split(key)
    key, dhmc_init_key = jax.random.split(key)

    def ld_1arg(pos):
        return log_posterior_flat_2arg(pos, data_args)

    # ── Warmup: adapt (step_size, inverse_mass_matrix) once, then cache. ──
    # Split from sampling so a fresh call and a cached one end in the SAME
    # sampling scan. While the two were fused, a first fit ran warmup+sampling
    # inside `_dynamic_hmc_full_scan` and every later fit ran a sampling-only
    # scan against the cached parameters, structurally different computations,
    # so one pinned `key` returned two different posteriors.
    #
    # The adaptation is plain HMC window adaptation (dynamic_hmc.init needs a
    # random_generator_arg, which window_adaptation cannot supply), so this is
    # `_hmc_warmup_only` rather than a third near-identical helper. The
    # integration-step count matches the one `_dynamic_hmc_full_scan` hardcodes.
    if cached is not None:
        parameters = cached
        # A reused adaptation was tuned in an earlier call, so this fit measured no
        # warmup divergences of its own. The diagnostics key is then ABSENT rather
        # than None: Posterior.save() has no HDF5 representation for None and would
        # warn about a skipped entry on every warm fit (#2088). Presence of the key
        # means "measured in this call".
        warmup_record: dict = {}
        if verbose:
            logger.info(
                "  Reusing cached warmup (%.1fs). Step size: %.4f",
                time.time() - t0,
                float(parameters["step_size"]),
            )
    else:
        step_size, inv_mass_matrix, warmup_divergent = _hmc_warmup_only(
            init_flat,
            warmup_key,
            log_posterior_flat_2arg,
            data_args,
            n_warmup,
            _DHMC_WARMUP_LEAPFROG_STEPS,
            use_dense,
            target_accept_rate,
        )
        jax.block_until_ready(step_size)
        # Refuse before caching and before the sampling scan compiles (#2088).
        warmup_divergence_frac = final_window_divergence_frac(warmup_divergent, n_warmup)
        refuse_dead_warmup(
            warmup_divergence_frac,
            sampler="Dynamic HMC",
            step_size=float(step_size),
            n_warmup=n_warmup,
            n_samples=n_samples,
        )
        warmup_record = (
            {}
            if warmup_divergence_frac is None
            else {"warmup_divergence_frac": warmup_divergence_frac}
        )
        parameters = {"step_size": step_size, "inverse_mass_matrix": inv_mass_matrix}
        _set_cached_adaptation(fitter, adapt_key, parameters)
        if verbose:
            if warmup_divergence_frac is None:
                logger.info(
                    "  Warmup complete (%.1fs). Step size: %.4f",
                    time.time() - t0,
                    float(step_size),
                )
            else:
                logger.info(
                    "  Warmup complete (%.1fs). Step size: %.4f. "
                    "Divergent in the final warmup window: %.0f%%",
                    time.time() - t0,
                    float(step_size),
                    100.0 * warmup_divergence_frac,
                )

    # ── Sampling: one path, whether the adaptation was just tuned or reused. ──
    key, chain_key = jax.random.split(key)
    if n_chains > 1:

        def _init(p, init_key):
            return blackjax.mcmc.dynamic_hmc.init(p, ld_1arg, init_key)

        def _scan(s, ks):
            return _dynamic_hmc_chain_scan(
                s,
                ks,
                log_posterior_flat_2arg,
                data_args,
                parameters["step_size"],
                parameters["inverse_mass_matrix"],
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
        state = blackjax.mcmc.dynamic_hmc.init(init_flat, ld_1arg, dhmc_init_key)
        chain_keys = jax.random.split(chain_key, n_burnin + n_samples)
        positions, divergent = _dynamic_hmc_chain_scan(
            state,
            chain_keys,
            log_posterior_flat_2arg,
            data_args,
            parameters["step_size"],
            parameters["inverse_mass_matrix"],
        )
        _multichain_burnin_done = False

    # Burnin discarded Python-side. Multichain branch already handled
    # per-chain burnin before flatten; skip the global slice there.
    if n_burnin > 0 and not _multichain_burnin_done:
        positions = positions[n_burnin:]
        divergent = divergent[n_burnin:]
    n_divergent = int(jnp.sum(divergent))

    # Post-hoc dead-fit detection: any chain's draws mostly divergent (#2093).
    # Complements the pre-hoc warmup check for fits that collapse after warmup.
    # On failure, evict the cached adaptation so the next fit re-tunes.
    refuse_dead_sampling(
        divergent,
        sampler="Dynamic HMC",
        n_samples=n_samples,
        n_chains=n_chains,
        step_size=float(parameters["step_size"]),
        fitter=fitter,
        method_key=adapt_key,
    )

    wall_time = time.time() - t0

    # Leave the whitened coordinates before the draws are read as parameters.
    positions = problem.restore(positions)

    samples_phys = _vmap_samples_to_physical(positions, unravel_fn, context.to_physical)
    best_params = _mean_params(samples_phys)

    # Compute sampling diagnostics (divergence fraction and unique draw count).
    diag = sampling_diagnostics(positions, divergent)

    if verbose:
        logger.info(
            "  Dynamic HMC complete in %.1fs. Divergences: %d/%d",
            wall_time,
            n_divergent,
            n_samples * n_chains,
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
            "n_chains": n_chains,
            "n_divergent": n_divergent,
            **warmup_record,
            "step_size": float(parameters["step_size"]),
            **diag,
        },
        loss_history=None,
        _model=context.model,
    )
