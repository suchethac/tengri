# SPDX-License-Identifier: BSD-3-Clause
"""ChEES-HMC via BlackJAX: cross-chain adapted trajectory length, lock-step kept.

``bench/reports/2026-08-17_quickstart_nuts_vs_hmc.md`` closed with a line this
module exists to test: *"trajectory length is a property of the posterior, not of
the notebook."* Three reports measured fixed-``L`` HMC against NUTS on the three
notebook posteriors and found no single global ``L`` that serves them -- ``L=20``
leaves ``sfh_tsnorm_log_total_mass`` at ESS 2.3 while ``L=160`` costs 125 s to
reach R-hat 1.024.

NUTS solves that by choosing a trajectory length *per chain, per step*, which is
exactly what breaks lock-step: a vmapped batch of NUTS chains runs at the speed
of its deepest tree. ChEES-HMC [1]_ chooses **one** trajectory length for the
whole ensemble by maximizing the Change in the Estimator of the Expected Square
across chains, then jitters it with a Halton sequence. Every chain draws its
leapfrog count from the same distribution at the same iteration, so the ensemble
stays in step while ``L`` is still learned from the posterior.

The metric is not learned here, and that is deliberate
------------------------------------------------------
``blackjax.chees_adaptation`` can estimate a diagonal ``inverse_mass_matrix``
from its own ensemble (``mass_matrix_estimation="diagonal"``). This backend
leaves it at ``None`` and takes the geometry from
:mod:`tengri.inference.preconditioning`'s analytic ``J^T N^-1 J + I`` instead.

That is not a stylistic preference. ``mcmc_ghmc``'s MEADS adaptation derives its
momentum scale from the adapting ensemble's own per-fold standard deviation, and
the measured consequence (``bench/reports/2026-08-30_ghmc_meads_adaptation.md``)
is an unopposed positive feedback loop: wider ensemble, larger momentum, longer
excursions, wider ensemble, ending at split-R-hat 1.1e10. Acceptance never
objected -- 0.989 straight through the blow-up -- because energy genuinely is
conserved under the same inflated metric that produced the excursions. An
ensemble-estimated metric is a loop; an analytic one computed once at the MAP is
not, and it whitens condition numbers of 1e5-1e8 down to 1.0 at that point.

References
----------
.. [1] M. D. Hoffman, A. Radul and P. Sountsov, "An Adaptive-MCMC Scheme for
   Setting Trajectory Lengths in Hamiltonian Monte Carlo", Proceedings of the
   24th International Conference on Artificial Intelligence and Statistics
   (AISTATS), PMLR 130:3907-3915, 2021.
   https://proceedings.mlr.press/v130/hoffman21a.html
"""

from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp

from tengri.inference._sample_utils import _maybe_map_init, _mean_params, _vmap_samples_to_physical
from tengri.inference.backends.mcmc._shared import (
    _CHEES_CHAIN_JITTER_SCALE,  # noqa: F401  (named here so callers can find it)
    _CHEES_JITTER_SCALE,
    _CHEES_LEARNING_RATE,
    _chees_cached_chain_scan,
    _chees_scan,
    _get_cached_adaptation,
    _get_flat_logdensity,
    _resolve_chees_ensemble,
    _set_cached_adaptation,
)
from tengri.inference.preconditioning import prepare_preconditioning
from tengri.utils.compile_log import compile_timer

logger = logging.getLogger(__name__)

#: ChEES's own dual-averaging target, and NOT NUTS's 0.8.
#:
#: Each ChEES step is a *fixed*-length HMC proposal, whose asymptotically optimal
#: acceptance rate is 0.651 (Beskos+2013), not the higher value NUTS is tuned to.
#: Named here because carrying the NUTS number across is the obvious mistake and
#: it would be invisible: the sampler would still run, just with a step size
#: chosen for a different proposal.
CHEES_TARGET_ACCEPT_RATE = 0.651


def run_chees(
    context,
    *,
    key,
    init_from=None,
    n_warmup=500,
    n_burnin=100,
    n_samples=1000,
    n_chains=1,
    n_ensemble="auto",
    ensemble_jitter=_CHEES_JITTER_SCALE,
    chain_jitter=None,
    jitter_amount=1.0,
    target_accept_rate=CHEES_TARGET_ACCEPT_RATE,
    max_leapfrog_steps=200,
    learning_rate=_CHEES_LEARNING_RATE,
    mass_matrix_estimation=None,
    dense_mass_matrix=False,
    precondition: bool | float | None = None,
    verbose=True,
):
    """ChEES-HMC sampling via BlackJAX: the trajectory length is learned.

    An ensemble of ``n_ensemble`` chains adapts one step size (dual averaging)
    and one trajectory length (Adam on the ChEES criterion) from cross-chain
    statistics. The first ``n_chains`` of the ensemble's warmed-up final states
    then sample, and their draws are what the returned posterior holds.

    Parameters
    ----------
    context : InferenceContext or Fitter
        Inference target.
    key : PRNGKey
        Random key. Advanced identically whether or not a cached adaptation
        exists, so two identical calls with one key return identical chains.
    init_from : dict, optional
        Starting parameters; ``None`` runs a MAP seed first.
    n_warmup : int
        ChEES adaptation steps. Each costs ``n_ensemble * L`` gradients with
        ``L`` the current adapted length, so the budget is bounded by
        ``n_warmup * n_ensemble * max_leapfrog_steps`` and not by the posterior
        -- unlike a NUTS warmup.
    n_burnin : int
        Post-adaptation burn-in, discarded per chain.
    n_samples : int
        Posterior samples to collect **per chain**.
    n_chains : int
        Sampling chains whose draws are concatenated into the posterior.

        **This is not the ensemble.** ``num_chains`` is a required positional
        argument of ``blackjax.chees_adaptation``, and its trajectory-length
        gradient is built from centered positions *across* chains, so an
        ensemble of one centers to exactly zero and never adapts. That
        constraint binds ``n_ensemble``, not this argument -- see
        ``tengri.inference.backends.mcmc._shared._resolve_chees_ensemble``,
        which also records why the ensemble axis is chains-within-galaxy and
        never galaxies-within-batch.
    n_ensemble : int or ``"auto"``
        Chains ChEES runs *during adaptation*. ``"auto"`` gives 32. Never smaller
        than ``n_chains``; an explicit value below 4 is refused rather than run.
    ensemble_jitter : float
        Gaussian dispersion of the **adaptation ensemble** around the seed
        position, in unbounded latent units [dimensionless]. A
        criterion-estimation dial: ChEES maximizes the change in the cross-chain
        expected square, so a dispersed ensemble carries a large expected square
        before the sampler moves anything and the optimizer settles on a shorter
        trajectory. See ``tengri.inference.backends.mcmc._shared._CHEES_JITTER_SCALE``.
    chain_jitter : float, optional
        Gaussian overdispersion of the **sampling chains** around the seed
        position [dimensionless]. A diagnostic dial, and a different chain set
        from ``ensemble_jitter``'s -- which is the whole point of separating
        them.

        ``None`` (default) seeds the sampling chains from the adaptation
        ensemble's warmed final states: no warmup is discarded, but those chains
        are correlated with the ensemble that tuned the sampler, so split R-hat
        over them is closer to a consistency check than to an independent test.
        A float seeds them independently instead, which is what makes R-hat a
        real test -- R-hat only detects non-convergence when its chains start
        overdispersed relative to the posterior, and chains started at one point
        can share a non-equilibrium basin and still score clean.
        ``tengri.inference.backends.mcmc._shared._CHEES_CHAIN_JITTER_SCALE``
        is the suggested width.
    jitter_amount : float
        Fraction of the adapted trajectory length that is jittered, in
        ``[0, 1]``. BlackJAX's 1.0 draws each step's length uniformly from
        ``(0, L]`` via a Halton sequence.
    target_accept_rate : float
        Dual-averaging target [dimensionless]. Defaults to
        :data:`CHEES_TARGET_ACCEPT_RATE` (0.651), which is ChEES's value, not
        NUTS's 0.8.
    max_leapfrog_steps : int
        Hard cap on leapfrog steps per proposal. Lowered from BlackJAX's 1000 to
        200 because the cap is what bounds a warmup step's cost: at 1000 a single
        adaptation step over a 32-chain ensemble is 32,000 gradient evaluations,
        and on the D <= 8 photometry posteriors the adapted length settles one to
        two orders of magnitude below that. Raise it for a posterior with a
        genuinely long correlation length; do not raise it to "be safe".
    learning_rate : float
        Adam step on ``log`` trajectory length [dimensionless].
    mass_matrix_estimation : ``None`` or ``"diagonal"``
        ``None`` (default) pins ``inverse_mass_matrix`` to ones and takes the
        geometry from ``precondition=`` instead. ``"diagonal"`` lets ChEES
        estimate the metric from its own ensemble.

        **Prefer the analytic metric.** The ensemble-estimated route is the same
        shape as the one that broke ``mcmc_ghmc``: MEADS's momentum scale *is*
        the ensemble's per-fold standard deviation, and the loop it closes
        (wider ensemble, larger momentum, longer excursions, wider ensemble) ran
        to split-R-hat 1.1e10 with acceptance at 0.989 throughout, because
        acceptance cannot see a metric that inflates along with the trajectory it
        is measuring. ChEES's Metropolis step and dual-averaged step size make
        the loop weaker here, not absent. Exposed so the ablation is re-runnable
        from a call rather than an edit; off by default because a metric this
        codebase can compute analytically should not be guessed from 32 chains.

        **It is an ablation, not a configuration, and setting it changes the
        sampler in a second way.** BlackJAX 1.6.2 turns its trajectory-length
        floor on exactly when a mass matrix is being estimated, and that branch
        calls ``float()`` on a traced step size (``chees_adaptation.py`` ~line
        990), so the pair raises ``ConcretizationTypeError`` under *any* ``jit``
        -- single fit as much as catalog vmap, and independently of tengri.
        Every tengri ChEES entry point is jitted, so the only way this option can
        run at all is with the floor off, which ``_chees_scan`` does for you and
        warns about. Measured in
        ``bench/reports/2026-08-31_catalog_preconditioning.md``.
    dense_mass_matrix : bool
        Must be ``False``. ChEES's kernel metric is a *diagonal*
        ``inverse_mass_matrix`` -- identity under the default
        ``mass_matrix_estimation=None`` -- so there is no dense option to take.
        The argument exists because the shared Hamiltonian-backend surface
        carries it and a silent ``TypeError`` deep in dispatch is worse than a
        named refusal; ``True`` raises rather than being quietly ignored, which
        is the failure ``run_ghmc``'s ``target_accept_rate`` records. A
        non-identity geometry comes from ``precondition=``, not from here.
    precondition : bool, float or None
        Sample in metric-whitened coordinates (#1301), the analytic
        ``J^T N^-1 J + I`` built once at the initial point. Opt-in: ``None``
        (default) and ``False`` are off, ``True`` uses
        :data:`~tengri.inference.preconditioning.DEFAULT_WHITENING_STRENGTH`, a
        float in ``[0, 1]`` sets the strength. See
        :mod:`tengri.inference.preconditioning`.
    verbose : bool
        Log progress.

    Returns
    -------
    Posterior
        ``diagnostics`` carries the adapted ``step_size`` and ``n_leapfrog_steps``
        -- the latter is the number this backend exists to learn -- plus the
        ensemble size actually used.

    Raises
    ------
    ImportError
        If BlackJAX or optax is not installed.
    ValueError
        If ``n_ensemble`` is below the floor, or ``mass_matrix_estimation`` is
        neither ``None`` nor ``"diagonal"``.
    """
    try:
        import blackjax  # noqa: F401
    except ImportError:
        raise ImportError("blackjax required for ChEES-HMC: pip install blackjax") from None
    try:
        import optax  # noqa: F401
    except ImportError:
        raise ImportError(
            "optax required for ChEES-HMC: chees_adaptation takes an "
            "optax.GradientTransformation for the trajectory-length step. "
            "pip install optax"
        ) from None

    if mass_matrix_estimation not in (None, "diagonal"):
        raise ValueError(
            f"mass_matrix_estimation must be None or 'diagonal', got {mass_matrix_estimation!r}."
        )
    if dense_mass_matrix:
        raise ValueError(
            "run_chees has no dense mass matrix: the ChEES kernel's metric is a "
            "diagonal inverse_mass_matrix, and under the default "
            "mass_matrix_estimation=None it is the identity for the whole run. "
            "For a non-identity geometry pass precondition=True, which whitens "
            "with the analytic J^T N^-1 J + I built once at the initial point -- "
            "a full metric, and one that cannot feed back on the ensemble the "
            "way an ensemble-estimated one can."
        )

    from tengri.inference.context import InferenceContext
    from tengri.inference.posterior import Posterior

    # ``_shared.py`` helpers still take a Fitter; reach through the context until
    # they migrate.
    context = InferenceContext.from_target(context)
    fitter = context.fitter

    init_params, key = _maybe_map_init(fitter, key, init_from, verbose)

    log_posterior_flat_2arg, unravel_fn, init_flat, data_args = _get_flat_logdensity(
        fitter,
        init_params,
    )

    # Metric preconditioning (#1301). A linear change of variables, so the
    # posterior is untouched and the draws are mapped back below. This is where
    # ChEES's geometry comes from -- see the module docstring for why it is not
    # taken from the ensemble.
    problem = prepare_preconditioning(
        log_posterior_flat_2arg, init_flat, data_args, precondition=precondition
    )
    log_posterior_flat_2arg, init_flat = problem.logdensity, problem.init_flat

    n_dim = len(init_flat)
    n_chains = max(1, int(n_chains))
    ensemble_size = _resolve_chees_ensemble(n_ensemble, n_chains)
    n_iter = int(n_burnin) + int(n_samples)

    if verbose:
        logger.info(
            "ChEES-HMC: %d parameters, %d adaptation steps over %d ensemble chains, "
            "%d samples x %d chains",
            n_dim,
            n_warmup,
            ensemble_size,
            n_samples,
            n_chains,
        )

    t0 = time.time()

    # Every knob that *produces* the adaptation belongs in the key -- leaving one
    # out makes it silently inert on a model that already holds an entry, which
    # is the failure ``_adaptation_cache_key``'s docstring records. Namespaced to
    # "chees" rather than borrowing "hmc": the tuple key would otherwise collide
    # with a step size adapted for a different kernel.
    tuning = (
        int(n_warmup),
        int(ensemble_size),
        float(ensemble_jitter),
        float(jitter_amount),
        float(target_accept_rate),
        int(max_leapfrog_steps),
        float(learning_rate),
        mass_matrix_estimation,
        None if chain_jitter is None else float(chain_jitter),
    )
    adapt_key = ("chees", tuning, problem.cache_key)
    cached = _get_cached_adaptation(fitter, adapt_key)

    # Both branches must advance the key identically: cache presence is invisible
    # to the caller and must not steer the RNG stream, or two identical ``fit``
    # calls with one key return different chains.
    key, warmup_key = jax.random.split(key)
    key, chain_key = jax.random.split(key)

    chain_keys = jax.random.split(chain_key, n_chains * n_iter).reshape(n_chains, n_iter, 2)

    if cached is not None:
        # The cached path re-runs the adaptation's *sampling* half only if the
        # warmed ensemble was also cached; it is not, because the states are
        # ``n_ensemble x D`` of live sampler state rather than a handful of
        # scalars. So a cache hit here reuses the tuned (step size, L, metric) and
        # starts the chains cold from ``init_flat``, which is what ``n_burnin`` is
        # for. Recorded rather than hidden: it is the reason a cached ChEES call
        # is not bit-identical to the first one.
        step_size = cached["step_size"]
        inverse_mass_matrix = cached["inverse_mass_matrix"]
        n_leapfrog = cached["n_leapfrog"]
        if verbose:
            logger.info(
                "  Reusing cached ChEES adaptation (%.1fs). Step size: %.4g, L: %.1f",
                time.time() - t0,
                float(step_size),
                float(n_leapfrog),
            )
        with compile_timer("chees_cached_scan", fitter.compile_signature(), method="mcmc_chees"):
            positions, divergent = _chees_cached_chain_scan(
                init_flat,
                chain_keys,
                log_posterior_flat_2arg,
                data_args,
                step_size,
                inverse_mass_matrix,
                n_leapfrog,
                float(jitter_amount),
                int(n_iter),
                None if chain_jitter is None else float(chain_jitter),
            )
            jax.block_until_ready(positions)
    else:
        with compile_timer("chees_warmup_scan", fitter.compile_signature(), method="mcmc_chees"):
            positions, divergent, step_size, inverse_mass_matrix, n_leapfrog = _chees_scan(
                init_flat,
                warmup_key,
                chain_keys,
                log_posterior_flat_2arg,
                data_args,
                int(n_warmup),
                int(ensemble_size),
                int(n_chains),
                int(n_iter),
                float(ensemble_jitter),
                float(jitter_amount),
                float(target_accept_rate),
                int(max_leapfrog_steps),
                float(learning_rate),
                mass_matrix_estimation,
                None if chain_jitter is None else float(chain_jitter),
            )
            jax.block_until_ready(positions)
        _set_cached_adaptation(
            fitter,
            adapt_key,
            {
                "step_size": step_size,
                "inverse_mass_matrix": inverse_mass_matrix,
                "n_leapfrog": n_leapfrog,
            },
        )
        if verbose:
            logger.info(
                "  ChEES adaptation + chains complete (%.1fs). Step size: %.4g, L: %.1f",
                time.time() - t0,
                float(step_size),
                float(n_leapfrog),
            )

    # Per-chain burn-in, discarded before the flatten so ``n_burnin`` applies to
    # *each* chain -- the contract ``_vmap_chains`` enforces.
    if n_burnin > 0:
        positions = positions[:, n_burnin:]
        divergent = divergent[:, n_burnin:]
    positions = positions.reshape(-1, positions.shape[-1])
    divergent = divergent.reshape(-1)

    n_divergent = int(jnp.sum(divergent))
    wall_time = time.time() - t0

    # Leave the whitened coordinates before the draws are read as parameters.
    positions = problem.restore(positions)

    samples_phys = _vmap_samples_to_physical(positions, unravel_fn, context.to_physical)
    best_params = _mean_params(samples_phys)

    if verbose:
        logger.info(
            "  ChEES-HMC complete in %.1fs. Divergences: %d/%d",
            wall_time,
            n_divergent,
            positions.shape[0],
        )

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="ChEES-HMC (BlackJAX)",
        wall_time_s=wall_time,
        diagnostics={
            "n_warmup": n_warmup,
            "n_burnin": n_burnin,
            "n_samples": n_samples,
            "n_chains": n_chains,
            "n_ensemble": ensemble_size,
            "ensemble_jitter": float(ensemble_jitter),
            "chain_jitter": None if chain_jitter is None else float(chain_jitter),
            "n_divergent": n_divergent,
            "step_size": float(step_size),
            # The number this backend exists to learn. Reported so a row in
            # bench/ can be read against the fixed-L HMC rows it is replacing.
            "n_leapfrog_steps": float(n_leapfrog),
            "target_accept_rate": float(target_accept_rate),
            "mass_matrix_estimation": mass_matrix_estimation,
            "preconditioned": bool(problem.enabled),
            "metric_condition": problem.metric_condition,
            "whitened_condition": problem.whitened_condition,
        },
        loss_history=None,
        _model=context.model,
    )
