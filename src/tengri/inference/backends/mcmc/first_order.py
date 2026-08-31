# SPDX-License-Identifier: BSD-3-Clause
"""Barker and MALA: one gradient per step, no branch anywhere in the program.

Every sampler tengri has measured so far answers bad geometry by *spending
more*: NUTS by doubling its tree, ChEES by learning a longer trajectory,
fixed-`L` HMC by being handed a larger `L`. Each of those buys quality with a
``lax.while_loop`` or with leapfrog steps, and
``bench/reports/2026-08-31_catalog_preconditioning.md`` measured what that is
worth here -- the learned trajectory length is a **net negative** against a
fixed ``L = 10`` at equal mass matrix, and the preconditioner carries the
performance.

This module takes the opposite side of that trade. A Barker or MALA step is one
gradient, one elementwise proposal, one Metropolis test -- no trajectory, no
tree, no fixed-point solve. The whole compiled program is two ``lax.scan``
calls of statically known length, which matters for a reason the reports
already measured: **75% of a cold NUTS fit is XLA**, 189.4 s against 46.8 s
warm, and MCLMC's fixed-length scan compiled 14x cheaper than NUTS's ragged
tree builder for exactly this reason
(``bench/reports/2026-08-30_mclmc_tuning.md``).

**Barker is the candidate; MALA is its control, and the control is not
optional.** Barker's published claim [1]_ is not "cheaper" -- it is *robust to
step-size misspecification*: its skew-symmetric proposal degrades gracefully in
a direction whose scale the global step size gets wrong, where MALA's Gaussian
proposal overshoots and rejects. tengri's posterior is exactly that shape
(condition 1e5-1e8 raw; whitened to 1.0 at the MAP but 1e2-1e5 one sigma away),
so the claim is directly testable -- but only against a sampler identical in
every other respect. ``bench/reports/2026-08-31_catalog_preconditioning.md``
Finding 5 is what an uncontrolled comparison costs: 40% of an apparent sampler
deficit was a mass matrix one arm carried and the other did not.

So both arms run through one code path, one adaptation (step size only, dual
averaging at ``FIRST_ORDER_TARGET_ACCEPT_RATE`` = 0.574), and one identity mass
matrix. ``proposal=``
selects between them and is reachable from an ordinary call, not only from an
edit -- the ChEES backend claimed that of its own ablation knob and it was not
true (``bench/reports/2026-08-31_catalog_preconditioning.md``, "The
obstruction").

Geometry comes from ``inference/preconditioning.py``'s analytic
``J^T N^-1 J + I`` via ``precondition=``, never from an estimated mass matrix.
That is the same design decision ``chees.py`` records and for the same reason:
an adaptation that derives its metric from the ensemble it is adapting closes a
feedback loop (``bench/reports/2026-08-30_ghmc_meads_adaptation.md`` reached
split-R-hat 1.13e10 that way).

**There is no divergence count in the diagnostics, deliberately.** Barker and
MALA are Metropolis-corrected, so an over-large step is *rejected* rather than
flagged; there is no energy-error threshold and therefore no divergence to
report. Writing ``n_divergent = 0`` would claim a mechanism the samplers do not
have -- the error ``bench/reports/2026-08-30_mclmc_tuning.md`` names for
unadjusted samplers. ``acceptance_rate`` is reported instead, and a rate far
below its target is the signal a divergence count would have carried.

References
----------
.. [1] S. Livingstone and G. Zanella, "The Barker proposal: combining
   robustness and efficiency in gradient-based MCMC," Journal of the Royal
   Statistical Society Series B, 84, 496 (2022). arXiv:1908.11812.
   :doi:`10.1111/rssb.12482`
.. [2] G. O. Roberts and J. S. Rosenthal, "Optimal scaling of discrete
   approximations to Langevin diffusions," Journal of the Royal Statistical
   Society Series B, 60, 255 (1998). :doi:`10.1111/1467-9868.00123`
"""

from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp

from tengri.inference._sample_utils import (
    _maybe_map_init,
    _mean_params,
    _vmap_samples_to_physical,
)
from tengri.inference.backends.mcmc._shared import (
    FIRST_ORDER_TARGET_ACCEPT_RATE,
    _first_order_chain_scan,
    _first_order_full_scan,
    _get_cached_adaptation,
    _get_flat_logdensity,
    _set_cached_adaptation,
)
from tengri.inference.preconditioning import prepare_preconditioning
from tengri.utils.compile_log import compile_timer

logger = logging.getLogger(__name__)

#: Proposals this backend can run. ``"barker"`` is the candidate and
#: ``"mala"`` is its control; see the module docstring for why both are here.
FIRST_ORDER_PROPOSALS = ("barker", "mala")


def run_first_order(
    context,
    *,
    key,
    init_from=None,
    proposal: str = "barker",
    n_warmup=1000,
    n_burnin=0,
    n_samples=2000,
    n_chains=1,
    target_accept_rate: float = FIRST_ORDER_TARGET_ACCEPT_RATE,
    precondition: bool | float | None = None,
    verbose=True,
):
    """Barker (default) or MALA sampling via BlackJAX.

    Parameters
    ----------
    context : InferenceContext or Fitter
        Inference target; see ADR-0010.
    key : PRNGKey
        Random key. Split identically on the cached and uncached paths, so
        cache presence never steers the RNG stream.
    init_from : optional
        Initial parameters, or ``None`` to run a MAP first.
    proposal : {"barker", "mala"}, optional
        Which first-order proposal to run. ``"barker"`` is the candidate;
        ``"mala"`` is the control that isolates the proposal from everything
        else, and is reachable from a call so the ablation does not need an
        edit. Default ``"barker"``.
    n_warmup : int, optional
        Dual-averaging steps for the step size. The **only** thing adapted --
        the mass matrix stays at the identity for both arms, because MALA's
        BlackJAX kernel cannot take one and an unequal metric would make the
        comparison uninterpretable. Default 1000.
    n_burnin : int, optional
        Post-warmup draws discarded, Python-side. Default 0.
    n_samples : int, optional
        Kept draws per chain. A first-order step moves far less than an HMC
        trajectory, so this should be an order of magnitude above a NUTS
        ``n_samples`` -- the same units caveat that
        ``bench/reports/2026-08-30_mclmc_tuning.md`` found had been read as a
        sampler defect. Default 2000.
    n_chains : int, optional
        Independent chains, sampled sequentially from jittered starts and
        sharing one adapted step size. Default 1.
    target_accept_rate : float, optional
        Dual-averaging target [dimensionless]. Default
        ``FIRST_ORDER_TARGET_ACCEPT_RATE`` = 0.574, the first-order
        optimal-scaling value [2]_ -- **not** HMC's 0.8 and not ChEES's 0.651.
    precondition : bool, float, or None, optional
        Analytic ``J^T N^-1 J + I`` whitening strength; see
        ``inference/preconditioning.py``. ``None`` disables it.
    verbose : bool, optional
        Log progress. Default True.

    Returns
    -------
    Posterior
        Draws in physical parameters, with ``diagnostics`` carrying
        ``acceptance_rate``, ``step_size``, ``proposal`` and the
        preconditioning record. **No ``n_divergent`` key** -- see the module
        docstring.

    Raises
    ------
    ImportError
        If blackjax is not installed.
    ValueError
        If ``proposal`` is not one of :data:`FIRST_ORDER_PROPOSALS`.

    Notes
    -----
    The compiled program contains no ``while_loop`` and no data-dependent
    branch, so its cost per draw is constant and its trace is one shape.

    References
    ----------
    .. [1] S. Livingstone and G. Zanella, JRSS-B, 84, 496 (2022).
       arXiv:1908.11812.
    .. [2] G. O. Roberts and J. S. Rosenthal, JRSS-B, 60, 255 (1998).
    """
    try:
        import blackjax  # noqa: F401
    except ImportError:
        raise ImportError("blackjax required for Barker/MALA: pip install blackjax") from None

    if proposal not in FIRST_ORDER_PROPOSALS:
        raise ValueError(
            f"proposal must be one of {FIRST_ORDER_PROPOSALS}, got {proposal!r}. "
            "'barker' is the candidate; 'mala' is its control and differs in the "
            "proposal alone."
        )

    from tengri.inference.context import InferenceContext
    from tengri.inference.posterior import Posterior

    context = InferenceContext.from_target(context)
    fitter = context.fitter

    init_params, key = _maybe_map_init(fitter, key, init_from, verbose)
    log_posterior_flat_2arg, unravel_fn, init_flat, data_args = _get_flat_logdensity(
        fitter,
        init_params,
    )

    problem = prepare_preconditioning(
        log_posterior_flat_2arg, init_flat, data_args, precondition=precondition
    )
    log_posterior_flat_2arg, init_flat = problem.logdensity, problem.init_flat

    n_dim = len(init_flat)
    n_chains = max(1, int(n_chains))
    n_iter = int(n_burnin) + int(n_samples)

    if verbose:
        logger.info(
            "%s: %d parameters, %d warmup, %d burn-in, %d samples, %d chain(s)",
            proposal.upper(),
            n_dim,
            n_warmup,
            n_burnin,
            n_samples,
            n_chains,
        )

    t0 = time.time()

    # ``problem.cache_key`` folds the whitening basis in: a step size tuned in
    # one basis is a finite float that samples happily and badly in another.
    tuning = (int(n_warmup), float(target_accept_rate), proposal)
    adapt_key = ("first_order", tuning, problem.cache_key)
    cached = _get_cached_adaptation(fitter, adapt_key)

    # Both branches advance the key identically -- cache presence is invisible
    # to the caller and must not change the chains a given key produces.
    key, warmup_key = jax.random.split(key)
    key, chain_key = jax.random.split(key)
    chain_keys = jax.random.split(chain_key, n_chains * n_iter)
    chain_keys = chain_keys.reshape(n_chains, n_iter, 2)

    # Chains start jittered around the expansion point. 1e-3 matches
    # ``_vmap_chains``'s default so a multi-chain R-hat here is comparable with
    # every other backend's -- and carries the same caveat
    # (``bench/reports/2026-08-30_chees_hmc.md``: chains that start too close
    # together produce an R-hat that is a consistency check, not a test).
    jitter_key, key = jax.random.split(key)
    starts = init_flat[None, :] + 1e-3 * jax.random.normal(jitter_key, shape=(n_chains, n_dim))

    positions_per_chain = []
    accept_per_chain = []

    if cached is not None:
        step_size = cached["step_size"]
        if verbose:
            logger.info(
                "  Reusing cached warmup (%.1fs). Step size: %.4g",
                time.time() - t0,
                float(step_size),
            )
        for c in range(n_chains):
            with compile_timer(
                "first_order_chain_scan",
                fitter.compile_signature(),
                method=f"mcmc_{proposal}",
            ):
                pos, acc = _first_order_chain_scan(
                    starts[c],
                    chain_keys[c],
                    step_size,
                    log_posterior_flat_2arg,
                    data_args,
                    proposal,
                )
                jax.block_until_ready(pos)
            positions_per_chain.append(pos)
            accept_per_chain.append(acc)
    else:
        with compile_timer(
            "first_order_full_scan",
            fitter.compile_signature(),
            method=f"mcmc_{proposal}",
        ):
            pos, acc, step_size = _first_order_full_scan(
                starts[0],
                warmup_key,
                chain_keys[0],
                log_posterior_flat_2arg,
                data_args,
                int(n_warmup),
                proposal,
                float(target_accept_rate),
            )
            jax.block_until_ready(pos)
        positions_per_chain.append(pos)
        accept_per_chain.append(acc)
        _set_cached_adaptation(fitter, adapt_key, {"step_size": step_size})
        if verbose:
            logger.info(
                "  Warmup complete (%.1fs). Step size: %.4g",
                time.time() - t0,
                float(step_size),
            )
        for c in range(1, n_chains):
            with compile_timer(
                "first_order_chain_scan",
                fitter.compile_signature(),
                method=f"mcmc_{proposal}",
            ):
                pos, acc = _first_order_chain_scan(
                    starts[c],
                    chain_keys[c],
                    step_size,
                    log_posterior_flat_2arg,
                    data_args,
                    proposal,
                )
                jax.block_until_ready(pos)
            positions_per_chain.append(pos)
            accept_per_chain.append(acc)

    positions = jnp.stack(positions_per_chain)
    accept = jnp.stack(accept_per_chain)
    if n_burnin > 0:
        positions = positions[:, n_burnin:]
        accept = accept[:, n_burnin:]
    positions = positions.reshape(-1, positions.shape[-1])
    acceptance_rate = float(jnp.mean(accept))

    wall_time = time.time() - t0

    # Leave the whitened coordinates before the draws are read as parameters.
    positions = problem.restore(positions)
    samples_phys = _vmap_samples_to_physical(positions, unravel_fn, context.to_physical)
    best_params = _mean_params(samples_phys)

    if verbose:
        logger.info(
            "  %s complete in %.1fs. Acceptance: %.3f",
            proposal.upper(),
            wall_time,
            acceptance_rate,
        )

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method=f"{proposal.capitalize()} (BlackJAX)",
        wall_time_s=wall_time,
        diagnostics={
            "n_warmup": n_warmup,
            "n_burnin": n_burnin,
            "n_samples": n_samples,
            "n_chains": n_chains,
            "proposal": proposal,
            "step_size": float(step_size),
            "acceptance_rate": acceptance_rate,
            "acceptance_rate_target": float(target_accept_rate),
            # One gradient of the log-posterior per proposal, always, with no
            # branch that could make it two. This is the column
            # ``bench/scripts/benchmark_notebook_sampler.py`` reads to price a
            # draw against a NUTS tree.
            "n_gradients_per_draw": 1,
            "preconditioned": bool(problem.enabled),
            "metric_condition": problem.metric_condition,
            "whitened_condition": problem.whitened_condition,
        },
        loss_history=None,
        _model=context.model,
    )


def _forward(proposal, context, kwargs):
    """Call :func:`run_first_order` for ``proposal``, refusing a caller override.

    ``proposal=`` is the ablation axis, and ``mcmc_barker`` and ``mcmc_mala``
    are registered as two backends precisely so a campaign row names which arm
    it ran. Silently accepting ``fit(method="mcmc_mala", proposal="barker")``
    would let a row be labeled one thing and be the other, which is the
    fixture-drift failure ``bench/reports/2026-08-30_chees_hmc.md`` spends a
    section on. Callers who want to sweep the axis should call
    :func:`run_first_order` directly.
    """
    if "proposal" in kwargs:
        raise ValueError(
            f"proposal= is fixed to {proposal!r} by this backend. Call "
            "run_first_order(context, proposal=...) to choose it, or use the "
            "other backend name."
        )
    return run_first_order(context, proposal=proposal, **kwargs)


def run_barker(
    context,
    *,
    key,
    init_from=None,
    n_warmup=1000,
    n_burnin=0,
    n_samples=2000,
    n_chains=1,
    target_accept_rate: float = FIRST_ORDER_TARGET_ACCEPT_RATE,
    precondition: bool | float | None = None,
    verbose=True,
):
    """Barker-proposal MCMC. :func:`run_first_order` with ``proposal="barker"``.

    The signature is spelled out rather than delegated through ``**kwargs``
    because ``register_backend(..., accepts_precondition=True)`` is checked
    against the runner's **real** signature by
    ``tests/contract/test_preconditioning_capability.py``, and a ``**kwargs``
    wrapper makes that declaration unverifiable -- the capability would be
    asserted and never confirmed.

    Parameters
    ----------
    context : InferenceContext or Fitter
        Inference target.
    key, init_from, n_warmup, n_burnin, n_samples, n_chains, \
    target_accept_rate, precondition, verbose
        Identical to :func:`run_first_order`, which documents every one.

    Returns
    -------
    Posterior
        See :func:`run_first_order`.
    """
    return _forward(
        "barker",
        context,
        dict(
            key=key,
            init_from=init_from,
            n_warmup=n_warmup,
            n_burnin=n_burnin,
            n_samples=n_samples,
            n_chains=n_chains,
            target_accept_rate=target_accept_rate,
            precondition=precondition,
            verbose=verbose,
        ),
    )


def run_mala(
    context,
    *,
    key,
    init_from=None,
    n_warmup=1000,
    n_burnin=0,
    n_samples=2000,
    n_chains=1,
    target_accept_rate: float = FIRST_ORDER_TARGET_ACCEPT_RATE,
    precondition: bool | float | None = None,
    verbose=True,
):
    """MALA. The control arm for :func:`run_barker`, registered so it is runnable.

    Same code path, same step-size-only dual averaging, same identity mass
    matrix; the proposal is the only difference. See :func:`run_barker` for why
    the signature is spelled out.

    Parameters
    ----------
    context : InferenceContext or Fitter
        Inference target.
    key, init_from, n_warmup, n_burnin, n_samples, n_chains, \
    target_accept_rate, precondition, verbose
        Identical to :func:`run_first_order`, which documents every one.

    Returns
    -------
    Posterior
        See :func:`run_first_order`.
    """
    return _forward(
        "mala",
        context,
        dict(
            key=key,
            init_from=init_from,
            n_warmup=n_warmup,
            n_burnin=n_burnin,
            n_samples=n_samples,
            n_chains=n_chains,
            target_accept_rate=target_accept_rate,
            precondition=precondition,
            verbose=verbose,
        ),
    )
