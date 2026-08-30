# SPDX-License-Identifier: BSD-3-Clause
"""MCLMC and Adjusted MCLMC via BlackJAX.

Extracted from mcmc/common.py. Import via ``tengri.inference.backends.mcmc``.

**Requires blackjax >= 1.6.** Earlier versions have fundamentally incompatible
APIs. In 1.5, build_kernel requires logdensity_fn and inverse_mass_matrix but
mclmc_find_L_and_step_size does not. In 1.6+, build_kernel takes only the
integrator and mclmc_find_L_and_step_size takes logdensity_fn as an optional
parameter.
"""

from __future__ import annotations

import functools
import logging
import time

import jax
import jax.numpy as jnp

from tengri.inference._sample_utils import _maybe_map_init, _mean_params, _vmap_samples_to_physical
from tengri.inference.backends.mcmc._shared import (
    _adjusted_mclmc_sample_scan,
    _get_cached_adaptation,
    _get_flat_logdensity,
    _set_cached_adaptation,
    _vmap_chains,
)
from tengri.inference.preconditioning import prepare_preconditioning

logger = logging.getLogger(__name__)

#: Fraction of ``n_warmup`` handed to each of BlackJAX's three MCLMC tuning
#: stages (step size, diagonal preconditioner, momentum-decoherence length L).
#:
#: BlackJAX defaults these to ``0.1 / 0.1 / 0.1``, i.e. it spends **30%** of
#: ``num_steps`` on tuning because its ``num_steps`` means "the number of MCMC
#: steps that will subsequently be run", not "the warmup budget". Passing
#: ``n_warmup`` there — which is what a sampler wrapper naturally does — turned
#: a 500-step warmup into 166 integrator steps of tuning, one or two momentum
#: decoherence times. The preconditioner is estimated from a streaming variance
#: over the *second* stage's draws, so a chain that has barely moved reports a
#: posterior 5-20x narrower than it is, and the tuner then sizes the step for
#: that collapsed scale. Measured on ``01_why_jax`` (D=7): 166 tuning steps gave
#: sqrt(diag) = [0.13, 0.07, 0.01, 0.05, 0.14, 0.08, 0.07] and a step size of
#: 0.645 (11x NUTS's own adapted step on the same posterior); 5000 tuning steps
#: gave [0.71, 0.12, 0.05, 0.51, 0.38, 0.87, 1.06] and 0.090.
#:
#: Summing to 1.0 makes ``n_warmup`` mean what it says: the number of integrator
#: steps spent in warmup. (BlackJAX adds ``num_steps2 // 3`` more steps to
#: re-fit the step size after the preconditioner changes, so the true total is
#: ~1.1x ``n_warmup``.)
_MCLMC_TUNE_FRACS = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

#: Fraction of ``n_warmup`` per stage for the **adjusted** tuner. Each of its
#: adaptation steps runs a whole trajectory (``L / step_size`` integrator steps,
#: pinned to 2 by BlackJAX's ``target_num_integration_steps``), so 0.25 + 0.25
#: of ``n_warmup`` kernel calls is ~``n_warmup`` integrator steps. Its third
#: stage is off by default upstream and stays off here.
_ADJUSTED_MCLMC_TUNE_FRACS = (0.25, 0.25)

#: How far above its target the achieved EEVPD may sit before the run warns.
#:
#: Measured on ``05_fitting_photometry`` (D=8, 14 bands, six seeds): four seeds
#: land at 3.6-6.6x the target and are ordinary, while two land at 394x and
#: 322,000x — and **every one of those six seeds reports max split-R-hat below
#: 1.01**. One order of magnitude separates the two populations cleanly.
_ENERGY_VAR_WARN_RATIO = 10.0

#: Default target energy-error variance per dimension (EEVPD).
#:
#: **This is MCLMC's tuning target and it is not an acceptance rate.** MCLMC is
#: unadjusted — the dynamics carry no Metropolis accept/reject step at all — so
#: there is no acceptance probability to aim at. What the step-size adaptation
#: controls instead is ``Var[dE] / D``, exploiting ``Var[E] = O(eps^6)`` for the
#: integrator. Reusing an HMC ``target_accept_rate`` here would be meaningless.
#: 5e-4 is the value from Robnik et al. and BlackJAX's own default.
_DEFAULT_ENERGY_VAR = 5e-4


@functools.partial(jax.jit, static_argnums=(2, 5))
def _mclmc_sample_scan_with_energy(
    state, keys, kernel, L, step_size, logdensity_fn, inverse_mass_matrix
):
    """MCLMC sampling scan that also returns the per-step energy error.

    The sibling in ``_shared.py`` keeps positions only. An unadjusted sampler
    has no divergence count to report — there is no accept step that could
    reject — so the energy error *is* the diagnostic, and it has to leave the
    scan to be reportable. Returns ``(positions, energy_change, nonans)`` so
    both the multi-chain (``_vmap_chains``) and single-chain callers get the
    same three leaves with iterations on the leading axis.
    """

    def _step(s, k):
        """Advance MCLMC by one step, returning position, energy error and the finite flag."""
        s, info = kernel(
            rng_key=k,
            state=s,
            logdensity_fn=logdensity_fn,
            inverse_mass_matrix=inverse_mass_matrix,
            L=L,
            step_size=step_size,
        )
        return s, (s.position, info.energy_change, info.nonans)

    return jax.lax.scan(_step, state, keys)


class MCLMCEnergyErrorWarning(UserWarning):
    """An unadjusted MCLMC run finished far above the energy error it tuned for.

    **R-hat cannot see this, and that is the entire point.** MCLMC has no
    Metropolis correction, so a step size above the integrator's stability scale
    does not get rejected — it displaces the stationary distribution instead. The
    chains then mix perfectly well *to the wrong target*, and split-R-hat, which
    only asks whether the chains agree with each other, reports success.

    Measured on ``05_fitting_photometry`` (D=8, 14 bands, 2 chains, 40000 draws,
    six seeds, 2026-08-30): every seed returned max split-R-hat < 1.01, and two of
    them returned an energy-error variance per dimension of 1.5e-01 and 8.4e+01
    against a 5e-4 target — 300x and 170,000x high — with R-hat reading 1.0084 and
    1.0007 respectively. On the seed at 8.4e+01 the largest single-step energy
    change was 2.0e+03.

    This is the unadjusted analogue of the warning in
    ``bench/reports/2026-08-17_nb01_nb05_nuts_vs_hmc.md``, that zero divergences
    is not evidence of convergence for a fixed-trajectory sampler — and it is
    sharper here, because a fixed-trajectory HMC at least *has* an accept step
    that could have rejected. This sampler has none, so nothing in the run
    catches it except this number.

    What to do, in the order measured:

    * Raise ``n_warmup``. The step size is tuned from a streaming variance over
      warmup draws, and a short warmup sizes it for a posterior the chain has not
      seen yet.
    * Lower ``desired_energy_var``. It is the knob this warning is about.
    * Do **not** reach for BlackJAX's ``desired_energy_var_max_ratio`` cutoff on
      the kernel. It looks like the fix and inverts the failure: the cutoff
      reverts high-energy steps, the step-size adaptation then observes only the
      small energy changes that survived, concludes the step is far too
      conservative, and runs it up. Measured on the same seed: step size 674
      (against 0.14 without the cutoff), EEVPD 7.0e-08, and R-hat 1.3316 at ESS
      0.8. Feeding a diagnostic back into the adaptation that is supposed to be
      diagnosed by it is a loop, not a guard.
    * Treat the posterior as unusable until the number comes down. A confident
      wrong answer is the failure mode this warning exists to prevent.
    """


def _warn_if_energy_error_high(energy):
    """Emit :class:`MCLMCEnergyErrorWarning` when the achieved EEVPD is far off target.

    Fires at :data:`_ENERGY_VAR_WARN_RATIO` times the target. Deliberately
    unconditional on ``verbose``: this is a correctness signal about the
    posterior that was just produced, not progress chatter, and a caller who
    silenced logging still needs it.
    """
    achieved = energy["energy_var_per_dim"]
    target = energy["energy_var_per_dim_target"]
    if not (achieved > _ENERGY_VAR_WARN_RATIO * target):
        return
    import warnings

    warnings.warn(
        f"MCLMC finished with energy-error variance per dimension "
        f"{achieved:.3g}, {achieved / target:.0f}x the {target:.1e} it tuned "
        f"for (largest single-step energy change "
        f"{energy['max_abs_energy_change']:.3g}). MCLMC is unadjusted: there is "
        f"no accept step to reject an over-large step, so this displaces the "
        f"distribution being sampled rather than showing up as a rejection. "
        f"split-R-hat cannot detect it -- chains mix perfectly well to a wrong "
        f"target -- so do not read a good R-hat here as convergence. Raise "
        f"n_warmup (currently tuning from a chain that may not have explored "
        f"yet) or lower desired_energy_var, and re-check this number before "
        f"using the posterior. Do NOT arm BlackJAX's "
        f"desired_energy_var_max_ratio cutoff: it feeds the diagnostic back "
        f"into the adaptation and drives the step size up instead of down "
        f"(measured: step size 674, R-hat 1.33, ESS 0.8).",
        MCLMCEnergyErrorWarning,
        stacklevel=3,
    )


def _energy_diagnostics(energy_change, nonans, n_dim, desired_energy_var):
    """Summarize an MCLMC run's energy error — the unadjusted analogue of divergences.

    ``n_nonfinite_steps`` counts steps the kernel had to revert because the
    proposal was not finite; it is the closest thing MCLMC has to a divergence,
    and it is deliberately *not* named ``n_divergent``, because a reader who
    sees that key expects a Metropolis rejection count.
    """
    de = jnp.asarray(energy_change)
    return {
        "energy_var_per_dim": float(jnp.var(de) / n_dim),
        "energy_var_per_dim_target": float(desired_energy_var),
        "max_abs_energy_change": float(jnp.max(jnp.abs(de))),
        "n_nonfinite_steps": int(jnp.sum(~jnp.asarray(nonans))),
    }


def run_mclmc(
    context,
    *,
    key,
    init_from=None,
    n_warmup=5000,
    n_samples=20000,
    n_chains=1,
    desired_energy_var=_DEFAULT_ENERGY_VAR,
    precondition=None,
    verbose=True,
):
    """Microcanonical Langevin Monte Carlo (MCLMC) via BlackJAX.

    Gradient-based sampler on the microcanonical manifold. One integrator
    step per draw, **no accept/reject step at all**, so the cost per draw is
    constant — no ragged control flow to break lock-step across vmapped
    chains, unlike NUTS.

    Because there is no Metropolis correction, MCLMC is **biased**
    for finite step sizes. Use ``adjusted_mclmc`` for exact sampling.

    Parameters
    ----------
    n_warmup : int, default 5000
        Integrator steps spent tuning the step size, the diagonal
        preconditioner and the momentum-decoherence length ``L``. Split
        equally across BlackJAX's three tuning stages (see
        ``_MCLMC_TUNE_FRACS``), so this is the actual warmup cost rather
        than 30% of it.
    n_samples : int, default 20000
        Draws to collect. **One draw is one integrator step** (two gradient
        evaluations under the McLachlan integrator), not a whole trajectory,
        so this number is *not* comparable to ``n_samples`` for NUTS — a NUTS
        draw on a D=7 photometry posterior costs ~50 gradient evaluations.
        Setting it to a NUTS-like 2000 buys ~4000 gradient evaluations and
        returns a chain whose autocorrelation time exceeds its own length;
        that, not the tuner, is why this backend read as "ESS ~ 1" for a year.
        Successive draws are ~``L / step_size`` apart (measured 40-50 steps on
        the D=7/D=8 photometry mocks), so budget accordingly.
    n_chains : int
        Chains to run under ``jax.vmap``. All start from the same point
        (MAP + jitter), so split-R-hat here is a mixing check, not an
        overdispersed-start check.
    desired_energy_var : float, default 5e-4
        Target **energy-error variance per dimension** (EEVPD). This is the
        unadjusted sampler's tuning target and it replaces, rather than
        renames, HMC's ``target_accept_rate``: there is no accept step whose
        rate could be targeted. Lower values buy a smaller step size.
    precondition : bool, float or None, default None
        tengri's **analytic** metric whitening
        (:mod:`tengri.inference.preconditioning`, ``G = JᵀN⁻¹J + I``), off by
        default. It stacks with BlackJAX's own diagonal preconditioner, which
        this backend always leaves on: the analytic map removes the
        off-diagonal correlation the diagonal one cannot see, and the diagonal
        one then rescales whatever residual anisotropy the whitening left. They
        are not redundant, but they are also not free — measure before turning
        this on, and see ``bench/reports/2026-08-30_mclmc_tuning.md`` for what
        the combination did on the photometry mocks.
    verbose : bool
        Print progress.

    Returns
    -------
    Posterior
        ``diagnostics`` carries ``L``, ``step_size`` and — in place of the
        divergence count an adjusted sampler would report —
        ``energy_var_per_dim`` (the achieved EEVPD, to be read against
        ``energy_var_per_dim_target``), ``max_abs_energy_change`` and
        ``n_nonfinite_steps``. There is deliberately no ``n_divergent`` key:
        a zero there would be a statement about a mechanism this sampler does
        not have.

    Notes
    -----
    Tuning is BlackJAX's :func:`blackjax.mclmc_find_L_and_step_size`, the
    tuner written for the *unadjusted* kernel.
    :func:`blackjax.adjusted_mclmc_find_L_and_step_size` is a different
    function targeting an acceptance rate and belongs to
    :func:`run_adjusted_mclmc`; the two are not interchangeable.

    References
    ----------
    .. [1] Robnik, J., De Luca, G. B., Silverstein, E., & Seljak, U. (2023).
       "Microcanonical Hamiltonian Monte Carlo". Journal of Machine Learning
       Research, 24(311), 1-34. arXiv:2212.08549.
    .. [2] Robnik, J., & Seljak, U. (2024). "Fluctuation without dissipation:
       Microcanonical Langevin Monte Carlo". Proceedings of the 6th Symposium
       on Advances in Approximate Bayesian Inference. arXiv:2303.18221.
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

    # Analytic metric whitening (#1301), the same seam NUTS uses. Off unless
    # asked for; BlackJAX's diagonal preconditioner runs on top of whatever
    # basis this leaves, so the two compose rather than compete — the analytic
    # map is a full linear change of variables and the diagonal one is a
    # per-coordinate rescale of the result.
    problem = prepare_preconditioning(
        log_posterior_flat_2arg, init_flat, data_args, precondition=precondition
    )
    log_posterior_flat_2arg, init_flat = problem.logdensity, problem.init_flat
    if problem.enabled and verbose:
        logger.info(
            "MCLMC preconditioning: strength=%.2f, cond %.2e -> %.2e at the initial point",
            problem.strength,
            problem.metric_condition,
            problem.whitened_condition,
        )

    def ld_1arg(pos):
        """Closure binding log-posterior with data arguments."""
        return log_posterior_flat_2arg(pos, data_args)

    n_dim = len(init_flat)

    if verbose:
        logger.info("MCLMC: %d parameters, %d warmup, %d samples", n_dim, n_warmup, n_samples)

    t0 = time.time()

    # In blackjax >= 1.6, build_kernel takes only integrator (not logdensity_fn)
    kernel = blackjax.mcmc.mclmc.build_kernel(
        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
    )

    # n_warmup belongs in the key: it *produces* the adaptation, so leaving it
    # out makes the knob silently inert on a model that already holds an entry.
    # So does the EEVPD target, and so does the whitening basis — a step size
    # and a mass matrix tuned in one basis are meaningless in another (#1442).
    adapt_key = ("mclmc", int(n_warmup), float(desired_energy_var), problem.cache_key)
    cached = _get_cached_adaptation(fitter, adapt_key)

    # Both branches must advance the key identically, cache presence is
    # invisible to the caller and must not steer the RNG stream, or two
    # identical ``fit`` calls with one ``key`` return different chains.
    # ``tune_key`` is unused when the adaptation is reused.
    key, init_key = jax.random.split(key)
    key, tune_key = jax.random.split(key)
    state = blackjax.mcmc.mclmc.init(init_flat, ld_1arg, init_key)

    if cached is not None:
        params = cached
        if verbose:
            logger.info(
                "  Reusing cached warmup (%.1fs). L=%.4f, step_size=%.4f",
                time.time() - t0,
                float(params.L),
                float(params.step_size),
            )
    else:
        frac1, frac2, frac3 = _MCLMC_TUNE_FRACS
        state, params, n_tuning_steps = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            logdensity_fn=ld_1arg,
            num_steps=n_warmup,
            state=state,
            rng_key=tune_key,
            frac_tune1=frac1,
            frac_tune2=frac2,
            frac_tune3=frac3,
            desired_energy_var=desired_energy_var,
            diagonal_preconditioning=True,
        )

        _set_cached_adaptation(fitter, adapt_key, params)

        if verbose:
            logger.info(
                "  Warmup complete (%.1fs, %d integrator steps). "
                "L=%.4f, step_size=%.4f (%.0f steps per decoherence)",
                time.time() - t0,
                int(n_tuning_steps),
                float(params.L),
                float(params.step_size),
                float(params.L) / float(params.step_size),
            )

    key, sample_key = jax.random.split(key)
    if n_chains > 1:

        def _init(p, init_key):
            return blackjax.mcmc.mclmc.init(p, ld_1arg, init_key)

        def _scan(s, ks):
            _, out = _mclmc_sample_scan_with_energy(
                s, ks, kernel, params.L, params.step_size, ld_1arg, params.inverse_mass_matrix
            )
            return out

        positions, energy_change, nonans = _vmap_chains(
            _init,
            _scan,
            init_flat=init_flat,
            chain_key=sample_key,
            n_chains=n_chains,
            n_iter=n_samples,
            n_burnin=0,
        )
    else:
        sample_keys = jax.random.split(sample_key, n_samples)
        _, (positions, energy_change, nonans) = _mclmc_sample_scan_with_energy(
            state,
            sample_keys,
            kernel,
            params.L,
            params.step_size,
            ld_1arg,
            params.inverse_mass_matrix,
        )

    energy = _energy_diagnostics(energy_change, nonans, n_dim, desired_energy_var)
    _warn_if_energy_error_high(energy)

    wall_time = time.time() - t0

    # Draws leave the sampler in the whitened basis; identity when disabled.
    positions = problem.restore(positions)
    samples_phys = _vmap_samples_to_physical(positions, unravel_fn, context.to_physical)
    best_params = _mean_params(samples_phys)

    if verbose:
        logger.info(
            "  MCLMC complete in %.1fs (%d steps/chain). "
            "EEVPD %.2e (target %.2e), %d non-finite steps",
            wall_time,
            n_samples,
            energy["energy_var_per_dim"],
            energy["energy_var_per_dim_target"],
            energy["n_nonfinite_steps"],
        )

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="MCLMC (BlackJAX)",
        wall_time_s=wall_time,
        diagnostics={
            "n_warmup": n_warmup,
            "n_samples": n_samples,
            "n_chains": n_chains,
            "L": float(params.L),
            "step_size": float(params.step_size),
            "steps_per_decoherence": float(params.L) / float(params.step_size),
            "preconditioned": bool(problem.enabled),
            **energy,
        },
        loss_history=None,
        _model=context.model,
    )


def run_adjusted_mclmc(
    context,
    *,
    key,
    init_from=None,
    n_warmup=500,
    n_samples=2000,
    n_chains=1,
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
        Integrator steps spent tuning the step size, the diagonal
        preconditioner and ``L`` (see ``_ADJUSTED_MCLMC_TUNE_FRACS``).
    n_samples : int
        Posterior samples to collect. BlackJAX's tuner pins
        ``L = 2 * step_size``, so a draw here is a ~2-step trajectory —
        cheaper than a NUTS draw by roughly the same factor
        :func:`run_mclmc` is, and this count is likewise not comparable to
        ``n_samples`` for NUTS.
    target_accept_rate : float
        Target Metropolis acceptance rate. Default 0.65 balances
        bias correction with mixing efficiency. Meaningful **here and not in**
        :func:`run_mclmc`: this variant does carry a Metropolis correction,
        so :func:`blackjax.adjusted_mclmc_find_L_and_step_size` — a different
        tuner from the unadjusted one — takes an acceptance target.
    verbose : bool
        Print progress.
    """
    try:
        import blackjax
    except ImportError:
        raise ImportError("blackjax required: pip install blackjax") from None

    from tengri.inference.context import InferenceContext
    from tengri.inference.posterior import Posterior

    context = InferenceContext.from_target(context)
    fitter = context.fitter
    init_params, key = _maybe_map_init(fitter, key, init_from, verbose)

    log_posterior_flat_2arg, unravel_fn, init_flat, data_args = _get_flat_logdensity(
        fitter,
        init_params,
    )

    def ld_1arg(pos):
        """Closure binding log-posterior with data arguments."""
        return log_posterior_flat_2arg(pos, data_args)

    n_dim = len(init_flat)

    if verbose:
        logger.info(
            "Adjusted MCLMC: %d parameters, %d warmup, %d samples, target_accept=%.2f",
            n_dim,
            n_warmup,
            n_samples,
            target_accept_rate,
        )

    t0 = time.time()

    # In blackjax >= 1.6, build_kernel takes only integrator (not logdensity_fn)
    kernel = blackjax.mcmc.adjusted_mclmc.build_kernel(
        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
    )

    # See run_mclmc: the settings that produce the adaptation must be in the key.
    adapt_key = ("adjusted_mclmc", int(n_warmup), float(target_accept_rate))
    cached = _get_cached_adaptation(fitter, adapt_key)

    # Both branches must advance the key identically, see run_mclmc above.
    # ``tune_key`` is unused when the adaptation is reused.
    key, tune_key = jax.random.split(key)
    state = blackjax.mcmc.adjusted_mclmc.init(init_flat, ld_1arg)

    if cached is not None:
        params = cached
        if verbose:
            logger.info(
                "  Reusing cached warmup (%.1fs). L=%.4f, step_size=%.4f",
                time.time() - t0,
                float(params.L),
                float(params.step_size),
            )
    else:
        frac1, frac2 = _ADJUSTED_MCLMC_TUNE_FRACS
        state, params, _ = blackjax.adjusted_mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            logdensity_fn=ld_1arg,
            num_steps=n_warmup,
            state=state,
            rng_key=tune_key,
            target=target_accept_rate,
            frac_tune1=frac1,
            frac_tune2=frac2,
            diagonal_preconditioning=True,
        )

        _set_cached_adaptation(fitter, adapt_key, params)

        if verbose:
            logger.info(
                "  Warmup complete (%.1fs). L=%.4f, step_size=%.4f",
                time.time() - t0,
                float(params.L),
                float(params.step_size),
            )

    L = params.L
    step_size = params.step_size
    inverse_mass_matrix = params.inverse_mass_matrix
    n_integration_steps = jnp.ceil(L / step_size).astype(int)

    key, sample_key = jax.random.split(key)
    if n_chains > 1:

        def _init(p):
            return blackjax.mcmc.adjusted_mclmc.init(p, ld_1arg)

        def _scan(s, ks):
            _, (p, d) = _adjusted_mclmc_sample_scan(
                s, ks, kernel, step_size, n_integration_steps, ld_1arg, inverse_mass_matrix
            )
            return p, d

        positions, divergent = _vmap_chains(
            _init,
            _scan,
            init_flat=init_flat,
            chain_key=sample_key,
            n_chains=n_chains,
            n_iter=n_samples,
            n_burnin=0,
        )
    else:
        sample_keys = jax.random.split(sample_key, n_samples)
        _, (positions, divergent) = _adjusted_mclmc_sample_scan(
            state,
            sample_keys,
            kernel,
            step_size,
            n_integration_steps,
            ld_1arg,
            inverse_mass_matrix,
        )
    n_divergent = int(jnp.sum(divergent))

    wall_time = time.time() - t0

    samples_phys = _vmap_samples_to_physical(positions, unravel_fn, context.to_physical)
    best_params = _mean_params(samples_phys)

    if verbose:
        logger.info(
            "  Adjusted MCLMC complete in %.1fs. Divergences: %d/%d",
            wall_time,
            n_divergent,
            n_samples * n_chains,
        )

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="Adjusted MCLMC (BlackJAX)",
        wall_time_s=wall_time,
        diagnostics={
            "n_warmup": n_warmup,
            "n_samples": n_samples,
            "n_chains": n_chains,
            "n_divergent": n_divergent,
            "L": float(params.L),
            "step_size": float(params.step_size),
            "n_integration_steps": int(n_integration_steps),
        },
        loss_history=None,
        _model=context.model,
    )


# ── Elliptical Slice Sampling ─────────────────────────────────────
