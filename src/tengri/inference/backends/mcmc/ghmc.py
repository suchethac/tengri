# SPDX-License-Identifier: BSD-3-Clause
"""Generalized HMC via BlackJAX, adapted with MEADS.

Extracted from mcmc/common.py. Import via ``tengri.inference.backends.mcmc``.

Until 2026-08 this backend adapted with ``blackjax.window_adaptation`` -- HMC's
adaptation -- and then ran GHMC with hand-set ``alpha=0.8, delta=0.65``. Window
adaptation dual-averages a step size against a *target acceptance rate*, which
generalized HMC does not have (it uses a non-reversible slice update rather than
a reversible Metropolis step), and it has no way to see the damping ``alpha`` at
all. The damping is what sets GHMC's mixing, so the one knob that mattered was
the one nothing tuned. That is the diagnosis behind ``tier="broken"``.

``_shared._ghmc_meads_scan`` replaces it with ``blackjax.meads_adaptation``, the
adaptation BlackJAX ships *for* this kernel.
"""

from __future__ import annotations

import logging
import time
import warnings

import jax
import jax.numpy as jnp

from tengri.inference._sample_utils import _maybe_map_init, _mean_params, _vmap_samples_to_physical
from tengri.inference.backends.mcmc._shared import (
    _MEADS_JITTER_SCALE,
    _get_cached_adaptation,
    _get_flat_logdensity,
    _ghmc_chain_scan,
    _ghmc_meads_scan,
    _resolve_meads_ensemble,
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
    alpha=None,
    delta=None,
    n_ensemble="auto",
    n_folds=4,
    ensemble_jitter=_MEADS_JITTER_SCALE,
    low_rank_rank=None,
    low_rank_window_fraction=0.5,
    target_accept_rate=None,
    dense_mass_matrix=True,
    verbose=True,
):
    """Generalized HMC (GHMC) sampling via BlackJAX, tuned by MEADS.

    GHMC uses partial momentum refreshment controlled by ``alpha``: at each step
    the momentum is mixed with fresh noise, so chains remember their direction
    and mix better in elongated posteriors like the age-dust-metallicity banana.
    One leapfrog per step by construction, so the cost per step is constant --
    which is what makes it lock-step friendly on an accelerator, unlike NUTS.

    ``alpha`` and the step size are **adapted**, not guessed. MEADS [1]_ runs an
    ensemble of chains and derives both from cross-chain statistics: the step
    size from the maximum eigenvalue of the preconditioned gradient matrix, the
    damping from the maximum eigenvalue of the centered position matrix. It has
    no separate warmup phase, so the ensemble's warmed-up final states are reused
    directly as the sampling chains' initial states.

    Parameters
    ----------
    context : InferenceContext or Fitter
        Inference target.
    key : PRNGKey
        Random key. Advanced identically whether or not an adaptation is cached,
        so two identical calls with one key return identical chains.
    init_from : dict, optional
        Starting parameters; ``None`` runs a MAP seed first.
    n_warmup : int
        MEADS adaptation steps. Costs exactly ``n_warmup * n_ensemble`` gradient
        evaluations, vmapped ``n_ensemble`` wide -- flat in a way a NUTS warmup
        is not.
    n_burnin : int
        Post-adaptation burn-in steps, discarded per chain. MEADS hands over an
        already-warmed ensemble, so this is smaller than a window-adapted
        sampler needs; it is kept because the cached-adaptation path (below) does
        start cold.
    n_samples : int
        Posterior samples to collect **per chain**.
    n_chains : int
        Sampling chains whose draws are concatenated into the posterior. This is
        *not* the MEADS ensemble -- see ``n_ensemble``.

        ``num_chains`` is a required positional argument of
        ``blackjax.meads_adaptation`` and MEADS partitions it into ``n_folds``
        folds, adapting each fold from its neighbor's statistics, so an
        ensemble of one is degenerate rather than merely weak. That constraint
        binds ``n_ensemble``, **not** this argument: the two are deliberately
        different axes, because ``n_chains=1`` is this function's default and
        what every catalog fit uses, and tying the ensemble to it would have
        made the default configuration the one place where MEADS computes
        cross-chain statistics from a single sample. ``n_chains=1`` is therefore
        fully supported and still gets a genuinely adapted step size; an
        undersized *ensemble* is refused loudly instead.
    alpha : float, optional
        Momentum persistence [dimensionless]. ``None`` (default) uses the MEADS
        value. Passing a float pins it and **disables** that half of the
        adaptation, which is what the old ``0.8`` default did unconditionally.
    delta : float, optional
        Slice-translation scaling in the GHMC proposal [dimensionless]. ``None``
        (default) uses the MEADS value, which is ``alpha / 2`` by Algorithm 3.
    n_ensemble : int or ``"auto"``
        Chains MEADS runs *during adaptation*. ``"auto"`` gives 32. MEADS
        estimates two maximum eigenvalues per fold, so it is meaningless on a
        small ensemble: an explicit value giving fewer than four chains per fold
        is refused rather than run. Rounded up to a multiple of ``n_folds`` and
        never smaller than ``n_chains``.
    n_folds : int
        MEADS folds K [dimensionless]. 4 is the paper's value.
    ensemble_jitter : float
        Gaussian dispersion of the initial ensemble around the seed position, in
        unbounded latent units [dimensionless]. MEADS reads the ensemble spread
        as the posterior scale, so this is ~500x ``_vmap_chains``'s decorrelation
        jitter and the two are not interchangeable.
    low_rank_rank : int, optional
        MEADS-LRD. ``None`` (default) adapts the diagonal momentum metric. An
        int adapts a rank-``k`` low-rank metric from the pooled ensemble -- the
        obvious lever given the 1e5-1e8 latent condition numbers
        :mod:`tengri.inference.preconditioning` documents. **Measured and it
        does not help**: at ``rank = D`` the best split-R-hat is 1.81 on nb05
        and 1.46 on nb01 against a 1.01 bar. Exposed so the experiment is
        re-runnable, off by default because it is not an improvement.
    low_rank_window_fraction : float
        Only read when ``low_rank_rank`` is set [dimensionless].
    target_accept_rate : float, optional
        **Ignored, and warns if set.** MEADS does not target an acceptance rate;
        GHMC's non-reversible slice update has no Metropolis step to target.
        Retained only so a call written against the window-adapted signature
        fails loudly rather than with a ``TypeError`` deep in dispatch.
    dense_mass_matrix : bool
        Inert. GHMC's momentum generator treats ``momentum_inverse_scale`` as a
        diagonal vector, so the metric is always diagonal.
    verbose : bool
        Log progress.

    Returns
    -------
    Posterior
        ``diagnostics`` carries the adapted ``step_size``, ``alpha``, ``delta``
        and the ensemble size actually used.

    Raises
    ------
    ImportError
        If BlackJAX is not installed.
    ValueError
        If ``n_ensemble`` is too small for ``n_folds`` -- see
        ``_shared._resolve_meads_ensemble``.

    References
    ----------
    .. [1] M. D. Hoffman and P. Sountsov, "Tuning-Free Generalized Hamiltonian
       Monte Carlo", Proceedings of the 25th International Conference on
       Artificial Intelligence and Statistics (AISTATS), PMLR 151:7799-7813,
       2022. https://proceedings.mlr.press/v151/hoffman22a.html
    """
    try:
        import blackjax
    except ImportError:
        raise ImportError("blackjax required: pip install blackjax") from None

    from tengri.inference.context import InferenceContext
    from tengri.inference.posterior import Posterior

    if target_accept_rate is not None:
        warnings.warn(
            "run_ghmc ignores target_accept_rate: MEADS derives the step size from "
            "the ensemble's preconditioned-gradient eigenvalue, not from an "
            "acceptance target, and GHMC's non-reversible slice update has no "
            "Metropolis acceptance to target. Drop the argument, or pin the "
            "sampler by hand with alpha=/delta= instead.",
            UserWarning,
            stacklevel=2,
        )

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
    n_chains = max(1, int(n_chains))
    ensemble_size = _resolve_meads_ensemble(n_ensemble, n_chains, n_folds)

    if verbose:
        burnin_msg = f", {n_burnin} burn-in" if n_burnin > 0 else ""
        logger.info(
            "GHMC/MEADS: %d parameters, %d adaptation steps over %d ensemble chains%s, "
            "%d samples x %d chains",
            n_dim,
            n_warmup,
            ensemble_size,
            burnin_msg,
            n_samples,
            n_chains,
        )

    t0 = time.time()

    # GHMC's momentum generator treats momentum_inverse_scale as a
    # diagonal vector, so we must use diagonal mass matrix regardless
    # of the dense_mass_matrix flag.
    # Namespaced to this backend, not "hmc": the cache is keyed by tuple, so borrowing
    # another sampler's prefix is a collision waiting on the next field either side
    # adds. GHMC tunes a different kernel and must not inherit HMC's step size.
    # Every knob that *produces* the adaptation belongs in the key -- warmup length,
    # ensemble shape, dispersion, and any hand-pinned alpha/delta -- because leaving
    # one out makes it silently inert on a model that already holds an entry.
    # ``target_accept_rate`` is deliberately absent: MEADS does not read it, and the
    # call above warns rather than pretending it did.
    #
    # The trailing ``True`` pins "always diagonal" and must stay both LAST and on
    # this one line: #1454 asserts on this statement as text, matching up to the
    # first ``)`` and reading the final element. That is what keeps GHMC out of
    # the dense-mass advisory, so the tuning rides in a single grouped element
    # rather than being spliced in after it.
    tuning = (
        int(n_warmup),
        int(ensemble_size),
        int(n_folds),
        float(ensemble_jitter),
        None if alpha is None else float(alpha),
        None if delta is None else float(delta),
        None if low_rank_rank is None else int(low_rank_rank),
        float(low_rank_window_fraction),
    )
    adapt_key = ("ghmc", tuning, True)
    cached = _get_cached_adaptation(fitter, adapt_key)

    # Both branches must advance the key identically, cache presence is
    # invisible to the caller and must not steer the RNG stream, or two
    # identical ``fit`` calls with one ``key`` return different chains.
    key, warmup_key = jax.random.split(key)
    key, chain_key = jax.random.split(key)

    n_iter = n_burnin + n_samples

    if cached is not None:
        parameters = cached
        step_size = parameters["step_size"]
        momentum_inv_scale = parameters["inverse_mass_matrix"]
        used_alpha = float(parameters["alpha"])
        used_delta = float(parameters["delta"])

        def ld_1arg(pos):
            """Bind this galaxy's data to the flat log-posterior."""
            return log_posterior_flat_2arg(pos, data_args)

        if verbose:
            logger.info(
                "  Reusing cached MEADS adaptation (%.1fs). Step size: %.4f, alpha: %.4f",
                time.time() - t0,
                float(step_size),
                used_alpha,
            )

        def _scan(state, keys):
            """Sample one chain from the cached adaptation."""
            return _ghmc_chain_scan(
                state,
                keys,
                log_posterior_flat_2arg,
                data_args,
                step_size,
                momentum_inv_scale,
                used_alpha,
                used_delta,
                used_alpha,
                used_delta,
            )

        if n_chains > 1:

            def _init(p, init_key):
                # Keyword args: blackjax reordered ghmc.init's (rng_key,
                # logdensity_fn) between 1.3 and 1.6, keywords work on both.
                return blackjax.mcmc.ghmc.init(position=p, logdensity_fn=ld_1arg, rng_key=init_key)

            positions, divergent = _vmap_chains(
                _init,
                _scan,
                init_flat=init_flat,
                chain_key=chain_key,
                n_chains=n_chains,
                n_iter=n_iter,
                n_burnin=n_burnin,
            )
        else:
            sample_key, ghmc_init_key = jax.random.split(chain_key)
            state = blackjax.mcmc.ghmc.init(
                position=init_flat, logdensity_fn=ld_1arg, rng_key=ghmc_init_key
            )
            positions, divergent = _scan(state, jax.random.split(sample_key, n_iter))
            if n_burnin > 0:
                positions = positions[n_burnin:]
                divergent = divergent[n_burnin:]
    else:
        chain_keys = jax.random.split(chain_key, n_chains * n_iter).reshape(n_chains, n_iter, 2)
        positions, divergent, step_size, momentum_inv_scale, alpha_a, delta_a = _ghmc_meads_scan(
            init_flat,
            warmup_key,
            chain_keys,
            log_posterior_flat_2arg,
            data_args,
            int(n_warmup),
            int(ensemble_size),
            int(n_folds),
            None if alpha is None else float(alpha),
            None if delta is None else float(delta),
            float(ensemble_jitter),
            None if low_rank_rank is None else int(low_rank_rank),
            float(low_rank_window_fraction),
        )
        used_alpha = float(alpha_a)
        used_delta = float(delta_a)
        parameters = {
            "step_size": step_size,
            "inverse_mass_matrix": momentum_inv_scale,
            "alpha": used_alpha,
            "delta": used_delta,
        }
        _set_cached_adaptation(fitter, adapt_key, parameters)

        # Per-chain burn-in, discarded before the flatten so ``n_burnin`` applies
        # to *each* chain -- the same contract ``_vmap_chains`` enforces.
        if n_burnin > 0:
            positions = positions[:, n_burnin:]
            divergent = divergent[:, n_burnin:]
        positions = positions.reshape(-1, positions.shape[-1])
        divergent = divergent.reshape(-1)

        if verbose:
            logger.info(
                "  MEADS + chain complete (%.1fs). Step size: %.4f, alpha: %.4f, delta: %.4f",
                time.time() - t0,
                float(step_size),
                used_alpha,
                used_delta,
            )

    n_divergent = int(jnp.sum(divergent))

    wall_time = time.time() - t0

    samples_phys = _vmap_samples_to_physical(positions, unravel_fn, context.to_physical)
    best_params = _mean_params(samples_phys)

    if verbose:
        logger.info(
            "  GHMC complete in %.1fs. Divergences: %d/%d",
            wall_time,
            n_divergent,
            positions.shape[0],
        )

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="GHMC (BlackJAX, MEADS-adapted)",
        wall_time_s=wall_time,
        diagnostics={
            "n_warmup": n_warmup,
            "n_burnin": n_burnin,
            "n_samples": n_samples,
            "n_chains": n_chains,
            "n_ensemble": ensemble_size,
            "n_folds": int(n_folds),
            "alpha": used_alpha,
            "delta": used_delta,
            "alpha_adapted": alpha is None,
            "delta_adapted": delta is None,
            "low_rank_rank": None if low_rank_rank is None else int(low_rank_rank),
            "n_divergent": n_divergent,
            "step_size": float(parameters["step_size"]),
        },
        loss_history=None,
        _model=context.model,
    )
