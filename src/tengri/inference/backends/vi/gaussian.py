# SPDX-License-Identifier: BSD-3-Clause
"""Gaussian variational inference: BlackJAX ``meanfield_vi`` and ``fullrank_vi``.

Both fit **one** Gaussian to the whole posterior by stochastic gradient descent on
the ELBO. They differ in exactly one thing, and it is the thing that matters here:

* ``meanfield`` carries a diagonal covariance, ``D`` scale parameters.
* ``fullrank`` carries a Cholesky factor, ``D (D + 1) / 2`` parameters.

tengri's posteriors are **tilted**: ``inference/preconditioning.py`` measures the
metric condition number at 8.5e4 to 3.1e8 on every configuration tested, with
``age`` / ``dust`` / ``metallicity`` correlated. A diagonal Gaussian has no way to
represent a tilt, so mean-field cannot be right about the marginal widths of a
correlated posterior no matter how long it is optimized; it reports the
*conditional* widths instead, which are narrower. That is a structural property of
the family, not a tuning failure, and the measured under-dispersion is in
``bench/reports/2026-08-31_vi_speed_evaluation.md``.

Why these are here at all
-------------------------
Not for fidelity. Both are a **fixed-length scan of one cheap step** -- no tree
doubling, no accept/reject, no ragged control flow -- which is the property
``bench/reports/2026-08-30_mclmc_tuning.md`` measured as worth 14x on XLA compile
against NUTS, and compile is 75% of a NUTS fit. What they buy is graph size; what
they cost is the tilt. Read the two together or not at all.

Requires: ``pip install blackjax optax``
"""

from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp

from tengri.inference._sample_utils import _mean_params, _vmap_samples_to_physical

logger = logging.getLogger(__name__)

__all__ = ["run_gaussian_vi", "run_gaussian_vi_fitter"]

#: Adam step size for the ELBO optimizer. BlackJAX's own VI notebooks use 1e-2 and
#: so does Stan's ADVI; tengri's latents are standardized, so the natural scale of a
#: variational mean is O(1) and a rate tuned on standardized coordinates transfers.
DEFAULT_LEARNING_RATE: float = 1e-2

#: Monte-Carlo draws per ELBO gradient. This is the knob that costs forward-model
#: evaluations: each step evaluates the log-density ``n_mc_samples`` times, so a
#: fit costs ``n_steps * n_mc_samples`` gradients and nothing else. BlackJAX
#: defaults to 5.
DEFAULT_MC_SAMPLES: int = 5


def _family(name: str):
    """Resolve ``"meanfield"`` / ``"fullrank"`` to the BlackJAX module."""
    import blackjax

    try:
        return {"meanfield": blackjax.meanfield_vi, "fullrank": blackjax.fullrank_vi}[name]
    except KeyError:
        raise ValueError(f"family must be 'meanfield' or 'fullrank', got {name!r}") from None


def run_gaussian_vi(
    *,
    key,
    log_posterior_flat,
    init_flat,
    unravel_fn,
    to_physical_fn,
    model,
    family="fullrank",
    n_steps=2000,
    n_samples=2000,
    n_mc_samples=DEFAULT_MC_SAMPLES,
    learning_rate=DEFAULT_LEARNING_RATE,
    restore_fn=None,
    precondition_diagnostics=None,
    verbose=True,
):
    """Fit one Gaussian to the posterior by SGD on the ELBO, then draw from it.

    Parameters
    ----------
    key : PRNGKey
        Random key.
    log_posterior_flat : callable
        ``log_p(xi) -> scalar`` on the flattened latent vector, data already bound.
    init_flat : ndarray, shape (D,)
        Starting variational mean, normally the MAP.
    unravel_fn : callable
        Flat vector back to a parameter dict.
    to_physical_fn : callable
        Unbounded parameter dict to physical space.
    model : Model
        Forward model, stored on the returned :class:`Posterior`.
    family : {'fullrank', 'meanfield'}
        Covariance family. See the module docstring for why the choice is not a
        tuning detail on a tilted posterior.
    n_steps : int
        ELBO optimization steps. This is a **fixed** count: the scan length is a
        compile-time constant and there is no convergence test inside the loop, so
        the graph is the same size for every posterior. ``elbo_history`` on the
        returned diagnostics is what says whether the count was enough.
    n_samples : int
        Posterior draws from the fitted Gaussian. Essentially free -- one
        multivariate-normal sample each, no forward model.
    n_mc_samples : int
        Monte-Carlo draws per ELBO gradient. The only knob that costs forward
        evaluations; see :data:`DEFAULT_MC_SAMPLES`.
    learning_rate : float
        Adam step size. See :data:`DEFAULT_LEARNING_RATE`.
    restore_fn : callable, optional
        Maps draws out of preconditioned coordinates, i.e.
        ``PreconditionedProblem.restore``. Identity when omitted.
    precondition_diagnostics : dict, optional
        Recorded verbatim into ``diagnostics``. Two fits whitened at different
        strengths are not comparable and a strength that is not written down will
        be assumed to have been the default (#1442).
    verbose : bool
        Log progress.

    Returns
    -------
    Posterior
        Draws from the fitted Gaussian. ``diagnostics['elbo_final']`` and
        ``diagnostics['elbo_history']`` are the only evidence that the optimizer
        converged; a Gaussian VI fit has no R-hat and no divergence count, and a
        posterior it returns is never a claim that the fit was good.

    Notes
    -----
    The returned draws are i.i.d. from a Gaussian, so an effective sample size
    computed from them equals ``n_samples`` by construction and says nothing at
    all about posterior quality. Do not read one.

    References
    ----------
    .. [1] A. Kucukelbir, D. Tran, R. Ranganath, A. Gelman & D. M. Blei,
       "Automatic Differentiation Variational Inference," Journal of Machine
       Learning Research 18(14), 1-45 (2017). arXiv:1603.00788.
    .. [2] G. Roeder, Y. Wu & D. Duvenaud, "Sticking the Landing: Simple,
       Lower-Variance Gradient Estimators for Variational Inference," NeurIPS
       (2017). arXiv:1703.09194.
    """
    from tengri.inference.backends.mcmc._shared import _check_blackjax_floor

    _check_blackjax_floor()
    try:
        import optax
    except ImportError:
        raise ImportError("optax required for Gaussian VI: pip install optax") from None

    from tengri.inference.posterior import Posterior

    algorithm = _family(family)

    t0 = time.time()
    init_flat = jnp.asarray(init_flat)
    n_dim = int(init_flat.shape[0])
    if verbose:
        logger.info(
            "Gaussian VI (%s): %d parameters, %d ELBO steps x %d MC draws, lr=%.1e",
            family,
            n_dim,
            n_steps,
            n_mc_samples,
            learning_rate,
        )

    optimizer = optax.adam(learning_rate)
    state = algorithm.init(init_flat, optimizer)

    def _one_step(carry, step_key):
        new_state, info = algorithm.step(
            step_key,
            carry,
            log_posterior_flat,
            optimizer,
            num_samples=n_mc_samples,
        )
        return new_state, info.elbo

    key, scan_key, sample_key = jax.random.split(key, 3)

    # One fixed-length scan. The whole reason this backend exists: the graph does
    # not depend on the posterior, so its compile cost is a property of the model
    # rather than of the geometry -- unlike a tree-doubling `while` loop.
    @jax.jit
    def _optimize(initial_state, run_key):
        return jax.lax.scan(_one_step, initial_state, jax.random.split(run_key, n_steps))

    final_state, elbo_history = _optimize(state, scan_key)
    jax.block_until_ready(elbo_history)

    t_fit = time.time() - t0
    elbo_final = float(jnp.mean(elbo_history[-max(n_steps // 20, 1) :]))
    if verbose:
        logger.info("  ELBO optimization complete (%.1fs), final ELBO %.2f", t_fit, elbo_final)

    samples_flat = algorithm.sample(sample_key, final_state, n_samples)
    samples_flat = jnp.reshape(jnp.asarray(samples_flat), (n_samples, n_dim))

    if restore_fn is not None:
        samples_flat = restore_fn(samples_flat)

    samples_phys = _vmap_samples_to_physical(samples_flat, unravel_fn, to_physical_fn)
    best_params = _mean_params(samples_phys)

    wall_time = time.time() - t0
    if verbose:
        logger.info("  Gaussian VI (%s) complete in %.1fs", family, wall_time)

    diagnostics = {
        "family": family,
        "n_dim": n_dim,
        "n_steps": n_steps,
        "n_samples": n_samples,
        "n_mc_samples": n_mc_samples,
        "learning_rate": learning_rate,
        # Forward-model evaluations the fit actually spent. The draws are free, so
        # this is the whole cost and it is known before the fit starts -- which is
        # the difference from every sampler in this package.
        "n_logdensity_evals": int(n_steps) * int(n_mc_samples),
        "elbo_final": elbo_final,
        "elbo_history": elbo_history,
        "optimizer_seconds": t_fit,
        # Non-finite ELBO steps. An Adam run that walked through a NaN keeps
        # walking and returns a finite, wrong Gaussian, so the count is not
        # optional and there is no acceptance test that would have caught it.
        "n_nonfinite_elbo": int(jnp.sum(~jnp.isfinite(elbo_history))),
    }
    if precondition_diagnostics:
        diagnostics.update(precondition_diagnostics)

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method=f"Gaussian VI, {family} (BlackJAX)",
        wall_time_s=wall_time,
        diagnostics=diagnostics,
        loss_history=-elbo_history,
        _model=model,
    )


def run_gaussian_vi_fitter(context, *, key, init_from=None, precondition=None, **kwargs):
    """Gaussian VI through the :class:`InferenceContext` interface.

    Parameters
    ----------
    context : InferenceContext or Fitter
        Inference target, normalized on entry.
    key : PRNGKey
        Random key.
    init_from : Posterior or dict, optional
        Starting point; ``None`` runs a MAP seed first. The variational mean is
        initialized there, so this is the same MAP-seeding contract the
        Hamiltonian backends have.
    precondition : bool, float or None
        Whiten the latent coordinates with the analytic ``J^T N^-1 J + I`` metric
        before optimizing, and map the draws back afterwards. Off by default
        (#1397). For a Gaussian VI fit this is not only a geometry fix: Adam takes
        one global step size, so an unwhitened posterior at cond 1e5-1e8 forces
        that step to the stiffest direction and the soft directions converge at
        that same rate. See :mod:`tengri.inference.preconditioning`.
    **kwargs
        Forwarded to :func:`run_gaussian_vi` (``family``, ``n_steps``,
        ``n_samples``, ``n_mc_samples``, ``learning_rate``, ``verbose``).

    Returns
    -------
    Posterior
        Draws from the fitted Gaussian.
    """
    from tengri.inference.backends.mcmc._shared import _get_flat_logdensity
    from tengri.inference.context import InferenceContext
    from tengri.inference.preconditioning import prepare_preconditioning

    context = InferenceContext.from_target(context)
    init_params = context.initial_params(key, init_from=init_from)

    log_posterior_flat_2arg, unravel_fn, init_flat, data_args = _get_flat_logdensity(
        context.fitter,
        init_params,
    )

    problem = prepare_preconditioning(
        log_posterior_flat_2arg, init_flat, data_args, precondition=precondition
    )
    if problem.enabled and kwargs.get("verbose", True):
        logger.info(
            "Gaussian VI preconditioning: strength=%.2f, cond %.2e -> %.2e at the initial point",
            problem.strength,
            problem.metric_condition,
            problem.whitened_condition,
        )

    whitened_logdensity = problem.logdensity

    def log_posterior_flat(pos):
        """Bind ``data_args`` from the enclosing scope, as BlackJAX's VI expects."""
        return whitened_logdensity(pos, data_args)

    return run_gaussian_vi(
        key=key,
        log_posterior_flat=log_posterior_flat,
        init_flat=problem.init_flat,
        unravel_fn=unravel_fn,
        to_physical_fn=context.to_physical,
        model=context.model,
        restore_fn=problem.restore,
        precondition_diagnostics={
            "precondition_enabled": problem.enabled,
            "precondition_strength": problem.strength,
            "metric_condition": problem.metric_condition,
            "whitened_condition": problem.whitened_condition,
        },
        **kwargs,
    )
