# SPDX-License-Identifier: BSD-3-Clause
"""Tempered SMC via BlackJAX: anneal from the exact prior, lock-step within a rung.

Every sampler this project has measured starts *at* the posterior -- from a MAP
seed, in a basin the optimizer found -- and asks a Markov chain to explore it.
On ``05_fitting_photometry`` seeds 0 and 1 that has never worked: those mocks
inject an SED whose 14 bands span 9.1e4 and 3.1e4 in flux, and NUTS returns
split-R-hat 1.4e13 and 2.4e13 while ChEES returns 1.24 and 1.06
(``bench/reports/2026-08-30_chees_hmc.md``). Annealing from the prior to the
posterior is the standard tool for a narrow ridge a cold start cannot find, and
it is the reason this backend exists.

Two things make tempering unusually cheap to wire in tengri specifically:

* **The lambda = 0 target is exact and free.** In the standardized latent space
  the prior is exactly ``N(0, I)``, so the initial particles are i.i.d. draws.
  There is no warmup and no burn-in, and **the MAP is used only to build the
  preconditioning metric, never as a starting point.**
* **The prior/likelihood split already exists.** ``build_loss_fn`` is the data
  term plus ``standardized_neg_log_prior``, which is precisely the pair
  ``blackjax.adaptive_tempered_smc`` asks for.

What SMC costs, stated up front
-------------------------------
``n_particles * n_temperatures * n_mcmc_steps * n_leapfrog_steps`` gradient
evaluations, for ``n_particles`` draws. Three of those four are chosen by the
caller; the rung count is chosen by the posterior. That product is reported as
``diagnostics["gradients_per_draw"]`` rather than left to be reconstructed,
because this project has already found one sampler winning on wall clock while
losing on gradients-per-effective-sample (``bench/reports/2026-08-30_mclmc_tuning.md``)
and the distinction decided that report.

What the diagnostics mean here, and what they do not
----------------------------------------------------
A resampled particle population is **exchangeable**. The autocorrelation ESS
estimator reads draws as a time series and therefore reports roughly the
particle count for any SMC output, degenerate or not; it is not a convergence
count for this backend and must not be read as one. The two numbers that are
informative are ``_shared._smc_ancestor_ess``
(how far the resampling collapsed the population) and the split R-hat across
``n_chains`` **fully independent** populations, which share no state, no
adaptation and no step size -- so unlike ChEES's default, this R-hat is an
independent test rather than a consistency check.

References
----------
.. [1] P. Del Moral, A. Doucet and A. Jasra, "An adaptive sequential Monte Carlo
   method for approximate Bayesian computation", *Statistics and Computing*,
   22:1009-1020, 2012. arXiv:1210.6811
.. [2] C. Dai, J. Heng, P. E. Jacob and N. Whiteley, "An Invitation to Sequential
   Monte Carlo Samplers", *Journal of the American Statistical Association*,
   117:1587-1600, 2022. arXiv:2007.11936
"""

from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp

from tengri.inference._sample_utils import _maybe_map_init, _mean_params, _vmap_samples_to_physical
from tengri.inference.backends.mcmc._shared import (
    _SMC_DEFAULT_PARTICLES,
    _SMC_INIT_STEP_SIZE,
    _SMC_MAX_TEMPERATURES,
    _SMC_STEP_SIZE_GAIN,
    _SMC_TARGET_ESS,
    _get_flat_logdensity,
    _get_flat_prior_and_likelihood,
    _smc_scan,
)
from tengri.inference.preconditioning import prepare_preconditioning
from tengri.utils.compile_log import compile_timer

logger = logging.getLogger(__name__)

#: Inner-kernel acceptance the step-size controller drives toward.
#:
#: 0.651 is the asymptotically optimal acceptance rate for a *fixed*-length HMC
#: proposal (Beskos+2013), which is what an inner move is here -- the same
#: reasoning, and the same number, as
#: :data:`~tengri.inference.backends.mcmc.chees.CHEES_TARGET_ACCEPT_RATE`, and
#: deliberately not NUTS's 0.8.
SMC_TARGET_ACCEPT_RATE = 0.651


def run_smc(
    context,
    *,
    key,
    init_from=None,
    n_particles=_SMC_DEFAULT_PARTICLES,
    n_chains=1,
    n_mcmc_steps=5,
    n_leapfrog_steps=20,
    target_ess=_SMC_TARGET_ESS,
    step_size=_SMC_INIT_STEP_SIZE,
    step_size_gain=_SMC_STEP_SIZE_GAIN,
    target_accept_rate=SMC_TARGET_ACCEPT_RATE,
    max_temperatures=_SMC_MAX_TEMPERATURES,
    fixed_ladder=None,
    precondition: bool | float | None = None,
    verbose=True,
    **_ignored_budget,
):
    """Tempered Sequential Monte Carlo: prior to posterior, one rung at a time.

    ``n_chains`` **independent** particle populations are run in one vmapped
    program. Each starts from exact i.i.d. prior draws and anneals to the
    posterior; the returned samples are every population's final particles,
    concatenated.

    Parameters
    ----------
    context : InferenceContext or Fitter
        Inference target.
    key : PRNGKey
        Random key.
    init_from : dict, optional
        **Used only to build the preconditioning metric**, never as a starting
        point -- SMC starts from the prior by construction. ``None`` runs a MAP
        seed first, which is wasted work when ``precondition`` is off; pass a
        MAP result to avoid paying for it twice.
    n_particles : int
        Particles per population. This is a *width*, not a chain length.
    n_chains : int
        Independent populations. Their split R-hat is the convergence
        diagnostic, and it is a real one because they share nothing.
    n_mcmc_steps : int
        Inner-HMC moves applied to every particle at every rung.
    n_leapfrog_steps : int
        Leapfrogs per inner move. Fixed, so a rung's cost is exactly
        ``n_particles * n_mcmc_steps * n_leapfrog_steps`` gradients.
    target_ess : float
        Weight-ESS fraction the adaptive schedule bisects for, in ``(0, 1)``.
        Lower gives fewer, larger temperature steps -- cheaper and cruder.
    step_size : float
        Inner-kernel step size at the first rung [latent units].
    step_size_gain : float
        Gain of the acceptance-driven step-size controller; ``0.0`` pins the
        step size for the whole run. See
        ``_shared._SMC_STEP_SIZE_GAIN`` for
        why a scalar acceptance controller does not reproduce MEADS's feedback
        loop.
    target_accept_rate : float
        Acceptance the controller drives toward. Defaults to
        :data:`SMC_TARGET_ACCEPT_RATE` (0.651), not NUTS's 0.8.
    max_temperatures : int
        Rung cap. A run that hits it did **not** reach the posterior and says so
        through ``diagnostics["reached_target"]``.
    fixed_ladder : int, optional
        ``None`` (default) runs the adaptive schedule, whose rung count is
        data-dependent and therefore a ``lax.while_loop``. An int runs a uniform
        ``lambda_k = k / K`` ladder of that many rungs as a fixed-length
        ``lax.scan``: fully lock-step, and a different sampler. Quote which one.
    precondition : bool, float or None
        Sample in metric-whitened coordinates (#1301), the analytic
        ``J^T N^-1 J + I`` built once at ``init_from``. Opt-in. **Both** the
        prior and the likelihood are re-expressed in the whitened coordinates,
        and the initial particles are drawn as ``A^-1 xi`` so they are still
        exact prior draws in the new basis.
    verbose : bool
        Log progress.
    **_ignored_budget
        Swallows ``n_warmup`` / ``n_burnin`` / ``n_samples``, which SMC has no
        analogue of: there is no warmup to run, nothing to burn in, and the draw
        count is ``n_particles``. Accepted rather than rejected so a caller can
        sweep this backend beside the chain samplers without a special case, and
        recorded in ``diagnostics["ignored_kwargs"]`` so a row that thought it
        asked for 4000 draws can see that it did not.

    Returns
    -------
    Posterior
        ``diagnostics`` carries ``n_temperatures``, ``log_evidence``,
        ``min_ancestor_ess``, ``gradients_per_draw`` and ``reached_target``.

    Raises
    ------
    ImportError
        If BlackJAX is not installed.
    ValueError
        If ``target_ess`` is not in ``(0, 1)`` or ``n_particles`` is below 2.
    """
    try:
        import blackjax  # noqa: F401
    except ImportError:
        raise ImportError("blackjax required for tempered SMC: pip install blackjax") from None

    n_particles = int(n_particles)
    if n_particles < 2:
        raise ValueError(
            f"n_particles={n_particles} is not a population. Tempered SMC reweights and "
            "resamples across particles, so a population of one has no weight variance "
            "to temper against and the adaptive schedule has nothing to bisect on."
        )
    if not 0.0 < float(target_ess) < 1.0:
        raise ValueError(
            f"target_ess must be a FRACTION of the particle count in (0, 1), got "
            f"{target_ess!r}. It is the weight-ESS the tempering schedule aims at, not a "
            "draw count."
        )

    from tengri.inference.context import InferenceContext
    from tengri.inference.posterior import Posterior

    context = InferenceContext.from_target(context)
    fitter = context.fitter

    # The MAP is the metric's expansion point and nothing else. When
    # preconditioning is off it is still needed for the latent ravel/unravel
    # structure, which is why it is not skipped outright.
    init_params, key = _maybe_map_init(fitter, key, init_from, verbose)

    log_posterior_flat_2arg, unravel_fn, init_flat, data_args = _get_flat_logdensity(
        fitter, init_params
    )
    logprior_flat_2arg, loglik_flat_2arg = _get_flat_prior_and_likelihood(fitter, init_params)

    # The metric is built from the POSTERIOR, as everywhere else in this
    # codebase, and then applied to both halves of the split. Building it from
    # the likelihood alone would whiten a different geometry than the one the
    # lambda = 1 target actually has.
    problem = prepare_preconditioning(
        log_posterior_flat_2arg, init_flat, data_args, precondition=precondition
    )

    n_dim = int(init_flat.shape[0])
    if problem.enabled:
        matrix = problem.preconditioner.matrix

        def logprior_sampled(position, data_args):
            """Prior in the whitened coordinates; the constant log|det A| is dropped."""
            return logprior_flat_2arg(matrix @ position, data_args)

        def loglik_sampled(position, data_args):
            """Likelihood in the whitened coordinates."""
            return loglik_flat_2arg(matrix @ position, data_args)

        # zeta = A^-1 xi maps an exact N(0, I) prior draw into the sampled basis.
        prior_draw_matrix = problem.preconditioner.inverse
    else:
        logprior_sampled = logprior_flat_2arg
        loglik_sampled = loglik_flat_2arg
        prior_draw_matrix = jnp.eye(n_dim, dtype=init_flat.dtype)

    n_chains = max(1, int(n_chains))
    if n_chains < 2:
        # Not a refusal -- one population is a legitimate and cheaper fit -- but
        # the diagnostic it hands back is not the one the column header says.
        # ``Posterior.rhat`` splits the flattened draws in half, which for two
        # populations is exactly population 0 against population 1 and is a real
        # between-run test. For ONE population it is the first half of an
        # exchangeable particle set against the second half of the same set,
        # which is ~1.0 by construction whatever the fit did. That is the
        # contamination pattern this project has now hit five times, and the only
        # defence is to say so at the call site.
        logger.warning(
            "mcmc_smc with n_chains=1: split R-hat over ONE particle population "
            "compares two halves of an exchangeable set and reads ~1.0 whatever "
            "happened. Pass n_chains>=2 for a between-population R-hat, and read "
            "diagnostics['min_ancestor_ess'] either way."
        )
    key, run_key = jax.random.split(key)
    run_keys = jax.random.split(run_key, n_chains)

    if verbose:
        logger.info(
            "Tempered SMC: %d parameters, %d particles x %d independent populations, "
            "%d inner HMC moves of L=%d per rung%s",
            n_dim,
            n_particles,
            n_chains,
            int(n_mcmc_steps),
            int(n_leapfrog_steps),
            "" if fixed_ladder is None else f", fixed ladder of {int(fixed_ladder)} rungs",
        )

    t0 = time.time()
    with compile_timer("smc_scan", fitter.compile_signature(), method="mcmc_smc"):
        (
            particles,
            log_z,
            n_temperatures,
            final_lambda,
            final_step_size,
            n_divergent,
            accept_sum,
            min_ancestor_ess,
        ) = _smc_scan(
            prior_draw_matrix,
            run_keys,
            data_args,
            logprior_sampled,
            loglik_sampled,
            n_particles,
            int(n_mcmc_steps),
            int(n_leapfrog_steps),
            float(target_ess),
            float(step_size),
            float(step_size_gain),
            float(target_accept_rate),
            int(max_temperatures),
            None if fixed_ladder is None else int(fixed_ladder),
        )
        jax.block_until_ready(particles)
    wall_time = time.time() - t0

    n_temperatures = [int(v) for v in n_temperatures]
    final_lambda = [float(v) for v in final_lambda]
    reached_target = all(v >= 1.0 for v in final_lambda)
    if not reached_target:
        logger.warning(
            "Tempered SMC stopped at lambda = %s after %s rungs (cap %d): these draws are "
            "from a TEMPERED distribution, not the posterior. Raise max_temperatures, "
            "raise n_mcmc_steps, or lower target_ess.",
            final_lambda,
            n_temperatures,
            int(max_temperatures),
        )

    positions = particles.reshape(-1, particles.shape[-1])
    positions = problem.restore(positions)

    samples_phys = _vmap_samples_to_physical(positions, unravel_fn, context.to_physical)
    best_params = _mean_params(samples_phys)

    # The whole cost of the fit, per draw, in gradient evaluations. Reported
    # rather than reconstructed: a rung is n_mcmc_steps * n_leapfrog_steps
    # gradients per particle, and one particle is one draw.
    grads_per_draw = float(max(n_temperatures) * int(n_mcmc_steps) * int(n_leapfrog_steps))
    total_divergent = int(jnp.sum(n_divergent))
    # The DENOMINATOR a divergence rate needs here, and it is not
    # ``total_draws()``. Every chain sampler makes one Metropolis transition per
    # kept draw, so kept draws and transitions are the same number and
    # ``n_divergent / total_draws`` is a rate. SMC makes ``n_temperatures *
    # n_mcmc_steps`` transitions per particle and keeps ONE draw from each, so
    # that ratio overshoots by exactly that factor -- it read 205 % on the first
    # measured row, which at least announces itself; a configuration with fewer
    # rungs would have produced a plausible-looking number instead. This is the
    # #2087 arithmetic one sampler further out, and the fix is the same shape:
    # publish the denominator rather than let a reader assume one.
    n_inner_transitions = int(sum(n_temperatures)) * n_particles * int(n_mcmc_steps)

    if verbose:
        logger.info(
            "  SMC complete in %.1fs. Rungs: %s. log Z: %s. Divergent inner "
            "transitions: %d/%d. Worst ancestor ESS: %.1f of %d.",
            wall_time,
            n_temperatures,
            [round(float(v), 3) for v in log_z],
            total_divergent,
            n_inner_transitions,
            float(jnp.min(min_ancestor_ess)),
            n_particles,
        )

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="Tempered SMC (BlackJAX)",
        wall_time_s=wall_time,
        diagnostics={
            # ``n_samples`` is PER POPULATION and ``n_chains`` counts the
            # populations, so ``total_draws`` and every divergence RATE built on
            # it are correct (#2087).
            "n_samples": n_particles,
            "n_chains": n_chains,
            "n_particles": n_particles,
            "n_mcmc_steps": int(n_mcmc_steps),
            "n_leapfrog_steps": int(n_leapfrog_steps),
            "n_temperatures": n_temperatures,
            "final_lambda": final_lambda,
            "reached_target": reached_target,
            "fixed_ladder": None if fixed_ladder is None else int(fixed_ladder),
            "target_ess": float(target_ess),
            # Divergent INNER transitions, summed over particles, rungs and
            # moves -- not divergent draws. Its denominator is
            # ``n_inner_transitions`` beside it, never ``total_draws()``.
            "n_divergent": total_divergent,
            "n_inner_transitions": n_inner_transitions,
            "accept_rate": [
                float(a) / n if n else None
                for a, n in zip(accept_sum, n_temperatures, strict=True)
            ],
            "step_size": float(jnp.mean(final_step_size)),
            # log Z comes free with the weights. One value per independent
            # population: their SPREAD is the error bar, and no single run can
            # produce one.
            "log_evidence": [float(v) for v in log_z],
            "log_evidence_spread": (
                float(jnp.max(log_z) - jnp.min(log_z)) if n_chains > 1 else None
            ),
            # NOT an autocorrelation ESS. See _smc_ancestor_ess.
            "min_ancestor_ess": float(jnp.min(min_ancestor_ess)),
            "gradients_per_draw": grads_per_draw,
            "preconditioned": bool(problem.enabled),
            "metric_condition": problem.metric_condition,
            "whitened_condition": problem.whitened_condition,
            "ignored_kwargs": sorted(_ignored_budget) or None,
        },
        loss_history=None,
        _model=context.model,
    )
