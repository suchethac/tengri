# SPDX-License-Identifier: BSD-3-Clause
"""Pathfinder: fast approximate posterior via quasi-Newton L-BFGS path.

Pathfinder (Zhang et al. 2022) traces the L-BFGS optimization trajectory
and fits a sequence of Gaussian approximations along the path. It picks
the best approximation (by ELBO) and draws samples from it.

~10x faster than NUTS for approximate posteriors, and excellent as a
warm-start initializer for NUTS or Ray Tracing chains.

Requires: ``pip install blackjax``
"""

import time

import jax
import jax.numpy as jnp

from tengri.inference._sample_utils import _mean_params, _vmap_samples_to_physical


def run_pathfinder(
    *,
    key,
    log_posterior_flat,
    init_flat,
    unravel_fn,
    to_physical_fn,
    model,
    n_samples=2000,
    maxiter=30,
    maxcor=10,
    n_elbo_draws=25,
    restore_fn=None,
    precondition_diagnostics=None,
    verbose=True,
):
    """Run BlackJAX Pathfinder for fast approximate posterior.

    Parameters
    ----------
    key : PRNGKey
        Random key.
    log_posterior_flat : callable
        Log-density on flattened parameter vector, cached via
        ``_get_flat_logdensity()`` for stable JIT identity.
    init_flat : jnp.ndarray
        Initial parameters as a flat 1-D array.
    unravel_fn : callable
        Converts flat array back to parameter dict.
    to_physical_fn : callable
        Converts unbounded param dict to physical space.
    model : Model
        Forward model (stored in Posterior).
    n_samples : int
        Number of posterior samples to draw.
    maxiter : int
        Maximum L-BFGS iterations along the path.
    maxcor : int
        L-BFGS memory (number of past gradients to store).
    n_elbo_draws : int
        Monte-Carlo draws used to estimate the ELBO at each L-BFGS iterate, which
        is how Pathfinder picks the best Gaussian along the path. Default 25,
        matching Stan's ``num_elbo_draws``. **This is a memory knob, not an
        accuracy knob:** the draws are vmapped through the full forward model, so
        peak memory scales as ``n_elbo_draws * maxiter * <cost of one SED>``.
        BlackJAX's own default is 200, which drove a 7-parameter photometry fit to
        26 GB and OOM-killed the slow test tier. Verified against blackjax 1.6.2
        rather than remembered: ``blackjax.vi.pathfinder.approximate`` declares
        ``num_samples: int = 200`` and so does ``VIAlgorithm.init``.

        **A reader copying blackjax's own Pathfinder page will hit that
        default.** That page never passes ``num_samples`` in either of the two
        styles it shows, tuning ``ftol`` instead, so its examples run at 200 ELBO
        draws -- exactly the configuration #1029 measured at 25.65 GB. Do not
        port a snippet from it without setting this.
    restore_fn : callable, optional
        Maps draws out of preconditioned coordinates, i.e.
        ``PreconditionedProblem.restore``. Identity when omitted.

        Pathfinder builds its Gaussian from a **low-rank** inverse Hessian read
        off the L-BFGS history
        (``blackjax.optimizers.lbfgs.lbfgs_inverse_hessian_formula_1``), and
        :mod:`tengri.inference.preconditioning` measures this posterior's raw
        metric at condition 8.5e4 to 3.1e8. A low-rank estimate of a curvature
        that stiff is the natural place for the covariance to come back
        near-singular, so whitening first is the one geometric lever available
        here. Measured in ``bench/reports/2026-08-31_vi_speed_evaluation.md``.
    precondition_diagnostics : dict, optional
        Recorded verbatim into ``diagnostics``. Two fits whitened at different
        strengths are not comparable (#1442).
    verbose : bool
        Print progress.

    Returns
    -------
    Posterior
        Approximate posterior samples from the best Gaussian along the path.

    Notes
    -----
    **Not exposed, and upstream's recommended lever:** ``approximate`` also takes
    ``ftol`` (default 1e-5), ``gtol`` (1e-8) and ``maxls`` (1000), and blackjax's
    Pathfinder page tunes ``ftol`` as its answer to L-BFGS trouble -- "L-BFGS can
    occasionally fail to converge from a bad initialization; retry until the ELBO
    path is finite." None of the three is reachable through ``fit()`` today, so a
    user following that advice cannot apply it here. Adding them is a one-line
    passthrough and is deliberately not done in the PR that measured this.

    **Precision.** Everything measured for this backend is x64. blackjax's page
    states plainly that "L-BFGS algorithm struggles with float32s and
    log-likelihood functions; it's suggested to use double precision numbers",
    so do NOT assume Pathfinder inherits the float32 safety that
    ``bench/reports/2026-08-31_float32_fitting_path.md`` established for the
    sampling path.

    ``n_samples`` (posterior draws, cheap -- one Gaussian sample each) and
    ``n_elbo_draws`` (path-selection draws, expensive -- one forward model each)
    are different quantities. Raising ``n_samples`` costs almost nothing; raising
    ``n_elbo_draws`` costs a forward evaluation per draw per iterate.

    References
    ----------
    .. [1] L. Zhang, B. Carpenter, A. Gelman & A. Vehtari, "Pathfinder:
       Parallel quasi-Newton variational inference," Journal of Machine Learning
       Research 23(306), 1-49 (2022). arXiv:2108.03782.
    """
    from tengri.inference.backends.mcmc._shared import _check_blackjax_floor

    _check_blackjax_floor()
    try:
        import blackjax
    except ImportError:
        raise ImportError("blackjax required for Pathfinder: pip install blackjax") from None

    from tengri.inference.posterior import Posterior

    t0 = time.time()

    n_dim = len(init_flat)

    if verbose:
        print(f"Pathfinder: {n_dim} parameters, maxiter={maxiter}, {n_samples} samples")

    key, approx_key, sample_key = jax.random.split(key, 3)

    # Use the module-level ``approximate`` / ``sample`` functions (they take
    # ``logdensity_fn`` explicitly) rather than the instance form
    # ``blackjax.pathfinder(logdensity).approximate(...)``. blackjax 1.4+ made
    # ``blackjax.pathfinder(logdensity)`` return a ``VIAlgorithm`` that no longer
    # carries ``.approximate`` (AttributeError), whereas the module functions
    # have kept a stable signature across ≥1.3, so this works on old and new
    # blackjax without pinning.
    # ``num_samples`` here is blackjax's ELBO-draw count, NOT the posterior draw
    # count -- it defaults to 200 and each draw is a full forward evaluation.
    state, _info = blackjax.pathfinder.approximate(
        approx_key,
        log_posterior_flat,
        init_flat,
        num_samples=n_elbo_draws,
        maxiter=maxiter,
        maxcor=maxcor,
    )

    if verbose:
        t_approx = time.time() - t0
        print(f"  Approximation complete ({t_approx:.1f}s)")

    samples_flat, log_q = blackjax.pathfinder.sample(sample_key, state, n_samples)

    if verbose:
        print(f"  Drew {n_samples} samples (mean log q = {float(jnp.mean(log_q)):.2f})")

    if restore_fn is not None:
        samples_flat = restore_fn(samples_flat)

    samples_phys = _vmap_samples_to_physical(samples_flat, unravel_fn, to_physical_fn)
    best_params = _mean_params(samples_phys)

    wall_time = time.time() - t0
    if verbose:
        print(f"  Pathfinder complete in {wall_time:.1f}s")

    diagnostics = {
        "n_samples": n_samples,
        "maxiter": maxiter,
        "maxcor": maxcor,
        "n_elbo_draws": n_elbo_draws,
        "mean_log_q": float(jnp.mean(log_q)),
        # Draws that came back non-finite. The L-BFGS inverse-Hessian estimate can
        # be near-singular on a stiff posterior, and a Gaussian sampled from a
        # near-singular covariance returns finite-looking garbage or NaN without
        # any accept step to reject it, so the count is not optional.
        "n_nonfinite_draws": int(jnp.sum(~jnp.isfinite(samples_flat).all(axis=-1))),
    }
    if precondition_diagnostics:
        diagnostics.update(precondition_diagnostics)

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="Pathfinder (BlackJAX)",
        wall_time_s=wall_time,
        diagnostics=diagnostics,
        loss_history=None,
        _model=model,
    )
