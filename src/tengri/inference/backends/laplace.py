# SPDX-License-Identifier: BSD-3-Clause
"""Laplace approximation: Gaussian posterior from Hessian at MAP.

The cheapest uncertainty estimate available, compute the Hessian of the
loss function H(xi) at the MAP estimate, invert to get a covariance matrix,
then draw samples from N(theta_MAP, H^{-1}).

Also provides a Laplace evidence estimate (the loss carries the prior
unnormalized as 0.5*sum(xi^2), so the (2*pi)^{D/2} factors cancel):
    log Z_laplace = -H(theta_MAP) - 0.5*log(det(H))

Before the Gaussian is built, the MAP estimate is polished by a few damped
saddle-free Newton steps on the exact Hessian (see :mod:`tengri.inference.newton`),
so the expansion point is a stationary point rather than wherever the optimizer
stopped.

Works entirely in unbounded parameter space (where the loss is smooth),
then transforms samples to physical space.
"""

from __future__ import annotations

import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from tengri.config.exceptions import (
    ImportanceSamplingUnreliableWarning,
    LaplaceNotAtModeWarning,
    LaplaceVarianceCeilingWarning,
    warn_measured,
)
from tengri.inference._sample_utils import _mean_params, _vmap_samples_to_physical
from tengri.inference.backends._psis import pareto_k_threshold
from tengri.inference.importance import StudentTProposal, importance_sample
from tengri.inference.newton import _loss_derivatives_jit, newton_polish

# Newton decrement above which the expansion point is reported as off-mode
# [nats]. An offset of ``delta`` standard deviations along a single direction
# gives ``d = delta^2 / 2``, so this is ~0.45 sigma. Measured on a 64-galaxy
# DRW-field population: converged fits scored 0.0005-0.075, under-converged
# ones 1.3 upward (issue #1537).
DEFAULT_STATIONARITY_TOL = 0.1

# Gradient max-norm at which the Newton polish stops. Far below any reporting
# tolerance: the polish is for reaching the mode, not for passing the check.
_NEWTON_POLISH_GRAD_TOL = 1e-10


def _regularize_hessian(hessian, min_eigenvalue, regularize=True):
    """Floor the Hessian spectrum, and report the variance that floors imply.

    Clipping is not damping. ``cov = H^-1``, so flooring an eigenvalue at
    ``min_eigenvalue`` does not cap that direction's variance, it *assigns* it
    ``1 / min_eigenvalue`` (``1e6``, i.e. std 1000, at the default). The
    directions the data constrain least therefore come back with the widest
    draws, and the width is an artifact of the floor (#1515).

    The count was previously computed and shown only under ``verbose``, which
    is off by default, so a fit could silently return floor-derived error bars.
    Extracted from :func:`run_laplace` so the reporting has one home and can be
    exercised directly on a known-singular Hessian.

    Parameters
    ----------
    hessian : jnp.ndarray, shape (n, n)
        Symmetric Hessian of the loss at the expansion point.
    min_eigenvalue : float
        Floor applied to the eigenvalue spectrum.
    regularize : bool, optional
        When ``False`` the spectrum is returned untouched and no warning is
        issued. Default ``True``.

    Returns
    -------
    eigenvalues : jnp.ndarray
        The raw spectrum, before flooring.
    eigenvalues_clipped : jnp.ndarray
        The floored spectrum (identical to ``eigenvalues`` when
        ``regularize=False``).
    eigenvectors : jnp.ndarray
        Eigenvectors of ``hessian``; the caller reuses them for the Newton
        decrement rather than decomposing twice.
    hessian_reg : jnp.ndarray
        Hessian rebuilt from the floored spectrum.
    n_clipped : int
        How many directions hit the floor.

    Warns
    -----
    LaplaceVarianceCeilingWarning
        When any direction is clipped, carrying the count, the assigned
        variance and standard deviation, and the smallest unclipped eigenvalue.
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(hessian)

    if not regularize:
        return eigenvalues, eigenvalues, eigenvectors, hessian, 0

    n_clipped = int(jnp.sum(eigenvalues < min_eigenvalue))
    eigenvalues_clipped = jnp.maximum(eigenvalues, min_eigenvalue)
    hessian_reg = eigenvectors @ jnp.diag(eigenvalues_clipped) @ eigenvectors.T

    if n_clipped > 0:
        n_dim = int(eigenvalues.size)
        implied_variance = 1.0 / float(min_eigenvalue)
        implied_std = float(np.sqrt(implied_variance))
        unclipped = eigenvalues[eigenvalues >= min_eigenvalue]
        smallest_unclipped = float(jnp.min(unclipped)) if unclipped.size else float("nan")
        warn_measured(
            f"Laplace clipped {n_clipped}/{n_dim} Hessian eigenvalues at "
            f"min_eigenvalue={min_eigenvalue:g}. Because cov = H^-1, each "
            f"clipped direction is *assigned* variance {implied_variance:.6g} "
            f"(std {implied_std:.6g}) in the unconstrained parameterization, "
            f"that width is the floor, not a measurement, and it lands on the "
            f"directions the data determine least well. Smallest unclipped "
            f"eigenvalue is {smallest_unclipped:.4g}. If a clipped direction is "
            f"an exact degeneracy (two parameters entering the model only in "
            f"combination, as met_alpha_fe and met_logzsol do, #1095), hold "
            f"one of the pair Fixed rather than lowering the floor.",
            LaplaceVarianceCeilingWarning,
            stacklevel=3,
            n_clipped=n_clipped,
            n_dim=n_dim,
            min_eigenvalue=float(min_eigenvalue),
            implied_variance=implied_variance,
            implied_std=implied_std,
            smallest_unclipped_eigenvalue=smallest_unclipped,
        )

    return eigenvalues, eigenvalues_clipped, eigenvectors, hessian_reg, n_clipped


def _newton_decrement(grad_flat, eigenvalues, eigenvectors):
    """Loss drop a quadratic model predicts between here and the mode [nats].

    ``d = 0.5 g^T H^-1 g``, evaluated in the Hessian eigenbasis so no second
    inverse is needed. Zero exactly at a stationary point, and invariant under
    affine reparameterization, unlike ``||g||``, which rescales with the
    parameters and so cannot carry a fixed threshold.

    Parameters
    ----------
    grad_flat : array_like, shape (n_dim,)
        Gradient of the loss at the expansion point, in unbounded space.
    eigenvalues : array_like, shape (n_dim,)
        Hessian eigenvalues, already floored to be positive.
    eigenvectors : array_like, shape (n_dim, n_dim)
        Corresponding eigenvectors, as columns.

    Returns
    -------
    float
        Predicted loss drop [nats]. An offset of ``delta`` standard deviations
        along one direction gives ``delta**2 / 2``. ``inf`` when any eigenvalue
        is non-positive: the quadratic model then has no minimum to descend to,
        and the point cannot be a mode whatever the gradient does.

    Notes
    -----
    The non-positive case is not a corner case to tidy away; it is the one
    input that breaks the formula's own algebra. A negative eigenvalue makes
    the sum **negative**, so a naive ``d > tol`` test reads as "converged" at a
    saddle. Reachable whenever ``regularize=False`` leaves the spectrum
    unfloored.
    """
    eigenvalues = jnp.asarray(eigenvalues)
    if not bool(jnp.all(eigenvalues > 0)):
        return float("inf")
    projected = eigenvectors.T @ grad_flat
    return float(0.5 * jnp.sum(projected**2 / eigenvalues))


def _validate_importance_args(importance_draws, importance_df):
    """Reject importance settings the Pareto check cannot use."""
    if isinstance(importance_draws, bool) or not isinstance(importance_draws, (int, np.integer)):
        raise ValueError(f"importance_draws must be a non-negative int, got {importance_draws!r}")
    if importance_draws < 0:
        raise ValueError(f"importance_draws must be a non-negative int, got {importance_draws}")
    if 0 < importance_draws < 10:
        raise ValueError(f"importance_draws must be 0 or at least 10, got {importance_draws}")
    if isinstance(importance_df, bool) or not (
        np.isfinite(float(importance_df)) and float(importance_df) > 0.0
    ):
        raise ValueError(f"importance_df must be a positive finite number, got {importance_df!r}")


def _importance_check(
    *,
    key,
    logdensity_fn,
    data_args,
    center,
    cov,
    n_samples,
    importance_draws,
    importance_df,
):
    """Reweight a Student-t proposal at the mode and resample the posterior from it.

    The proposal is centered on the polished mode with the Laplace covariance. The
    weights are PSIS-smoothed, and ``n_samples`` draws are resampled with
    replacement in proportion to them.

    Returns
    -------
    samples_flat : jnp.ndarray, shape (n_samples, D)
        Resampled posterior draws in unbounded space.
    diagnostics : dict
        ``pareto_k``, ``pareto_k_threshold``, ``importance_ess``,
        ``log_evidence_is``, ``log_evidence_is_err``.

    Warns
    -----
    ImportanceSamplingUnreliableWarning
        When the PSIS tail shape exceeds the threshold for the draw count, or is NaN.
    """
    key_is, key_resample = jax.random.split(key)
    n_dim = int(center.shape[0])
    proposal = StudentTProposal(
        mean=center,
        chol=jnp.linalg.cholesky(cov),
        df=float(importance_df),
    )
    log_norm = -0.5 * n_dim * math.log(2.0 * math.pi)
    result = importance_sample(
        logdensity_fn,
        data_args,
        proposal,
        importance_draws,
        key_is,
        log_norm=log_norm,
    )

    weights = jnp.exp(jnp.asarray(result.psis.log_weights))
    indices = jax.random.choice(
        key_resample, importance_draws, shape=(n_samples,), replace=True, p=weights
    )
    samples_flat = jnp.asarray(result.draws)[indices]

    pareto_k = float(result.psis.pareto_k)
    threshold = pareto_k_threshold(importance_draws)
    if not pareto_k <= threshold:
        warn_measured(
            f"Laplace importance check is unreliable: Pareto k-hat {pareto_k:.3g} "
            f"exceeds the threshold {threshold:.3g} for {importance_draws} draws "
            f"(ESS {result.ess:.4g}). The importance ratios are heavy-tailed, so "
            f"the reweighted posterior and log_evidence_is are dominated by a few "
            f"draws, and the Gaussian/Student-t approximation misses posterior mass. "
            f"Use mcmc_nuts or nss for this model.",
            ImportanceSamplingUnreliableWarning,
            stacklevel=3,
            pareto_k=pareto_k,
            pareto_k_threshold=threshold,
            n_importance_draws=int(importance_draws),
            importance_ess=float(result.ess),
        )

    diagnostics = {
        "pareto_k": pareto_k,
        "pareto_k_threshold": threshold,
        "importance_ess": float(result.ess),
        "log_evidence_is": float(result.log_z),
        "log_evidence_is_err": float(result.log_z_err),
    }
    return samples_flat, diagnostics


def run_laplace(
    *,
    key,
    logdensity_fn,
    data_args,
    position,
    unravel_fn,
    to_physical_fn,
    model,
    n_samples=2000,
    regularize=True,
    min_eigenvalue=1e-6,
    stationarity_tol=DEFAULT_STATIONARITY_TOL,
    newton_steps=2,
    importance_draws=0,
    importance_df=5.0,
    verbose=True,
):
    """Run Laplace approximation from a MAP estimate.

    Parameters
    ----------
    key : PRNGKey
        Random key for sampling.
    logdensity_fn : callable
        Log-density (maximized): ``(flat position, data_args) -> scalar``. The
        same function the samplers use; the loss is its negation.
    data_args : pytree
        Observed data passed through to ``logdensity_fn``.
    position : array_like, shape (n_dim,)
        Flat MAP estimate in unbounded space.
    unravel_fn : callable
        Maps a flat position back to the parameter pytree.
    to_physical_fn : callable
        Converts an unbounded parameter pytree to physical space.
    model : Model
        Forward model (stored in Posterior).
    n_samples : int
        Number of posterior samples to draw.
    regularize : bool
        Clip small Hessian eigenvalues to ensure positive definiteness.
    min_eigenvalue : float
        Minimum eigenvalue threshold (only if regularize=True).
    stationarity_tol : float
        Newton decrement [nats] above which the expansion point is reported as
        off-mode via :class:`~tengri.config.exceptions.LaplaceNotAtModeWarning`.
        Default 0.1, an offset of ~0.45 standard deviations. Raise it to
        silence the check; the decrement is reported in ``diagnostics`` either
        way.
    newton_steps : int, optional
        Maximum number of damped saddle-free Newton polish steps taken from the
        expansion point before the expansion (see Notes). ``0`` disables the
        polish. Default 2.
    importance_draws : int, optional
        Number of Student-t proposal draws for the importance check (see Notes).
        ``0`` disables the check and the samples are Gaussian. A positive value
        must be at least 10. Default 0.
    importance_df : float, optional
        Degrees of freedom of the Student-t proposal used by the importance
        check. Default 5.0.
    verbose : bool
        Print progress.

    Returns
    -------
    Posterior
        Samples from the Gaussian approximation, with Laplace log-evidence.
        With ``importance_draws > 0`` the samples are resampled from the
        importance draws in proportion to their PSIS-smoothed weights, and the
        diagnostics carry the importance check (see Notes).
        ``Posterior.log_evidence`` is populated from the Gaussian approximation
        formula (see Notes), valid only when stated guards are met.

    Warns
    -----
    LaplaceNotAtModeWarning
        When the Newton decrement exceeds ``stationarity_tol``, meaning
        ``position`` is not a stationary point of the loss and ``H^-1`` is
        therefore not a covariance (issue #1537).
    ImportanceSamplingUnreliableWarning
        When ``importance_draws > 0`` and the PSIS tail shape exceeds the
        reliability threshold, or is NaN.

    Notes
    -----
    **Newton polish**: up to ``newton_steps`` damped saddle-free Newton steps
    (Dauphin et al. 2014 [1]_) on the exact Hessian are taken from the
    optimizer's endpoint before the expansion. A step is accepted only if its
    loss is finite and strictly lower than the current loss. The covariance is
    always built from the Hessian at the polished point; L-BFGS's own
    inverse-Hessian estimate is never used for the covariance.

    **Posterior samples & uncertainty**: ``cov = H^-1`` holds only at a mode.
    ``run_map`` takes a fixed number of Adam steps with no convergence test, so
    an under-converged expansion point reaches here routinely and produces a
    confident, plausible, wrong posterior, typically far too narrow, since a
    point on a steep slope carries much higher curvature than the mode below it.
    The Newton decrement

    .. math::

        d = \\tfrac{1}{2}\\, g^{T} H^{-1} g

    measures this, where :math:`g` is the loss gradient at the expansion point
    and :math:`H` the (eigenvalue-floored) Hessian; :math:`d` is in nats and is
    zero exactly at a mode.

    **Laplace log-evidence**: The value in ``Posterior.log_evidence`` is the
    Gaussian approximation in standardized ξ-space, given by

    .. math::

        \\log Z = -H(\\theta^*) - \\tfrac{1}{2} \\log \\det H

    where :math:`H` is the Hessian of the loss at the expansion point
    :math:`\\theta^*`. The loss carries the standardized prior *unnormalized*
    (:math:`\\tfrac12\\sum\\xi^2`, no :math:`-\\tfrac{D}{2}\\log 2\\pi` term),
    which exactly cancels the :math:`(2\\pi)^{D/2}` Gaussian-integral factor,
    hence no explicit :math:`2\\pi` term appears. This evidence is
    reparametrization-invariant within the standardized ξ~N(0,1) prior space
    used internally.

    Trust the evidence only when **both** of these conditions hold:

    - ``diagnostics["newton_decrement"] ≤ stationarity_tol`` (default 0.1 nats):
      expansion point is close to a stationary point of the loss.
    - ``diagnostics["n_clipped_eigenvalues"] == 0``: no Hessian eigenvalues
      were floored (the spectrum is positive-definite and well-conditioned).

    **Importance check**: with ``importance_draws > 0``, a Student-t proposal
    (``importance_df`` degrees of freedom) centered on the polished mode with the
    Laplace covariance is importance-sampled against the same flat log-density
    the samplers use, with the normalized prior restored. The importance ratios
    are PSIS-smoothed [2]_, and the posterior samples are resampled from the
    draws in proportion to the smoothed weights. The Pareto shape ``k-hat`` is
    reported as ``diagnostics["pareto_k"]``. Above ``min(1 - 1/log10(n), 0.7)``
    (``n`` the number of importance draws) the reweighted posterior and
    ``diagnostics["log_evidence_is"]`` are unreliable, and
    :class:`~tengri.config.exceptions.ImportanceSamplingUnreliableWarning` is
    emitted. ``Posterior.log_evidence`` remains the Laplace value.

    **Known failure mode**: Posterior mass piled against a physical prior
    boundary (|ξ| large in standardized space) breaks the Gaussian approximation;
    the evidence can be severely biased. Remedy: cross-check against an
    independent evidence method (e.g., ``method="nss"``) once per model family
    to detect this.

    Evidence scatter (uncertainties) are not reported; the Laplace
    approximation has no intrinsic sampling error, only model-error bias which
    cannot be quantified without comparison to other methods.

    References
    ----------
    .. [1] Dauphin, Y. N., Pascanu, R., Gulcehre, C., Cho, K., Ganguli, S., &
       Bengio, Y. (2014). Identifying and attacking the saddle point problem in
       high-dimensional non-convex optimization. Advances in Neural Information
       Processing Systems 27. arXiv:1406.2572.
    .. [2] Vehtari, A., Simpson, D., Gelman, A., Yao, Y., & Gabry, J. (2024).
       Pareto Smoothed Importance Sampling. Journal of Machine Learning
       Research, 25(72), 1-58. arXiv:1507.02646.
    """
    from tengri.inference.posterior import Posterior

    if isinstance(newton_steps, bool) or not isinstance(newton_steps, (int, np.integer)):
        raise ValueError(f"newton_steps must be a non-negative int, got {newton_steps!r}")
    if newton_steps < 0:
        raise ValueError(f"newton_steps must be a non-negative int, got {newton_steps}")
    _validate_importance_args(importance_draws, importance_df)

    t0 = time.time()

    theta_flat = jnp.asarray(position)
    n_dim = len(theta_flat)

    if verbose:
        print(f"Laplace: {n_dim} parameters, {n_samples} samples")

    # Loss gradient and exact Hessian at the optimizer's endpoint.
    grad_flat, hessian = _loss_derivatives_jit(logdensity_fn, theta_flat, data_args)

    # Decrement at the endpoint, before any polish, on the raw spectrum.
    initial_eigenvalues, initial_eigenvectors = jnp.linalg.eigh(hessian)
    decrement_initial = _newton_decrement(grad_flat, initial_eigenvalues, initial_eigenvectors)

    # Damped saddle-free Newton polish: the Hessian, gradient and decrement below
    # all describe the polished point.
    n_newton_accepted = 0
    if newton_steps > 0:
        theta_flat, n_newton_accepted, grad_flat, hessian = newton_polish(
            theta_flat,
            logdensity_fn,
            data_args,
            max_steps=int(newton_steps),
            tol=_NEWTON_POLISH_GRAD_TOL,
            loss_grad=grad_flat,
            loss_hessian=hessian,
        )
    if verbose:
        print(f"  Newton polish: {n_newton_accepted} step(s) accepted")

    (
        _eigenvalues_raw,
        eigenvalues_clipped,
        eigenvectors,
        hessian_reg,
        n_clipped,
    ) = _regularize_hessian(hessian, min_eigenvalue=min_eigenvalue, regularize=regularize)

    # Covariance = H^{-1}
    cov = jnp.linalg.inv(hessian_reg)

    # Symmetrize covariance (numerical safety)
    cov = 0.5 * (cov + cov.T)

    if verbose and n_clipped > 0:
        print(f"  Regularized: {n_clipped}/{n_dim} eigenvalues clipped")

    # Is the expansion point actually a mode?  cov = H^-1 only means covariance
    # there; nothing upstream guarantees it, and nothing downstream can detect
    # it (i.i.d. Gaussian draws score R-hat ~ 1 whatever the shape).
    grad_norm = float(jnp.linalg.norm(grad_flat))
    decrement = _newton_decrement(grad_flat, eigenvalues_clipped, eigenvectors)

    indefinite = not bool(jnp.all(eigenvalues_clipped > 0))
    if indefinite:
        n_bad = int(jnp.sum(eigenvalues_clipped <= 0))
        warn_measured(
            f"Laplace Hessian is indefinite at the expansion point: "
            f"{n_bad}/{n_dim} eigenvalues are non-positive (most negative "
            f"{float(jnp.min(eigenvalues_clipped)):.4g}), gradient norm "
            f"{grad_norm:.4g}. The loss curves *downward* along those "
            f"directions, so this is a saddle or a maximum, not a mode, and "
            f"H^-1 is not a covariance, the draws are meaningless rather "
            f"than merely mis-scaled. Leave regularize=True (the default) to "
            f"floor the spectrum at min_eigenvalue, or re-run the optimizer "
            f"from a different start.",
            LaplaceNotAtModeWarning,
            stacklevel=2,
            grad_norm=grad_norm,
            newton_decrement=decrement,
            n_nonpositive_eigenvalues=n_bad,
            min_eigenvalue=float(jnp.min(eigenvalues_clipped)),
        )
    elif decrement > stationarity_tol:
        warn_measured(
            f"Laplace expansion point is not a mode: Newton decrement "
            f"{decrement:.4g} nats (tolerance {stationarity_tol:g}), gradient "
            f"norm {grad_norm:.4g}. The loss still drops by ~{decrement:.4g} "
            f"nats toward the true mode, so H^-1 describes the curvature of a "
            f"slope rather than a peak and the posterior is likely far too "
            f"narrow. Raise n_map_steps (10x converged every affected fit in "
            f"issue #1537), or pass an already-converged init_from. "
            f"Set stationarity_tol to silence this.",
            LaplaceNotAtModeWarning,
            stacklevel=2,
            grad_norm=grad_norm,
            newton_decrement=decrement,
            stationarity_tol=stationarity_tol,
        )

    if verbose:
        print(f"  Newton decrement: {decrement:.4g} nats (|grad| {grad_norm:.4g})")

    importance_diagnostics = {}
    if importance_draws > 0:
        if not bool(jnp.all(eigenvalues_clipped > 0)):
            raise ValueError(
                "importance_draws needs a positive-definite Hessian at the expansion "
                "point; the spectrum is indefinite (use regularize=True or a converged "
                "init_from)."
            )
        samples_flat, importance_diagnostics = _importance_check(
            key=key,
            logdensity_fn=logdensity_fn,
            data_args=data_args,
            center=theta_flat,
            cov=cov,
            n_samples=n_samples,
            importance_draws=int(importance_draws),
            importance_df=importance_df,
        )
    else:
        # Draw samples from N(theta_MAP, H^{-1})
        samples_flat = jax.random.multivariate_normal(key, theta_flat, cov, shape=(n_samples,))

    # Laplace log-evidence estimate. The loss is -log L + 0.5*sum(xi^2), i.e. the
    # prior enters UNNORMALIZED (standardized_neg_log_prior omits -(D/2) log 2pi).
    # Restoring that constant cancels the (2pi)^{D/2} Gaussian-integral factor:
    #   Z = (2pi)^{-D/2} * exp(-loss(xi_hat)) * (2pi)^{D/2} |H|^{-1/2}
    # so log Z = -loss_at_map - 0.5 log det H, with no explicit 2pi term. Keeping
    # the +(D/2) log 2pi here would bias log Z upward by 0.92 nats per free
    # parameter and tilt model comparison toward higher-D models.
    loss_at_map = -float(logdensity_fn(theta_flat, data_args))
    log_det_h = jnp.sum(jnp.log(eigenvalues_clipped))
    log_evidence = -loss_at_map - 0.5 * log_det_h

    if verbose:
        print(f"  Loss at MAP: {loss_at_map:.2f}")
        print(f"  Laplace log-evidence: {float(log_evidence):.2f}")
        print(f"  Condition number: {float(eigenvalues_clipped[-1] / eigenvalues_clipped[0]):.1e}")

    # Unravel and convert to physical space
    samples_phys = _vmap_samples_to_physical(samples_flat, unravel_fn, to_physical_fn)
    best_params = _mean_params(samples_phys)

    wall_time = time.time() - t0
    if verbose:
        print(f"  Laplace complete in {wall_time:.1f}s")

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="Laplace",
        wall_time_s=wall_time,
        diagnostics={
            "n_samples": n_samples,
            "log_evidence": float(log_evidence),
            "loss_at_map": loss_at_map,
            "eigenvalues": eigenvalues_clipped,
            "n_clipped_eigenvalues": n_clipped,
            "condition_number": float(eigenvalues_clipped[-1] / eigenvalues_clipped[0]),
            "newton_decrement": decrement,
            "newton_decrement_initial": decrement_initial,
            "newton_steps_accepted": n_newton_accepted,
            "grad_norm_at_expansion": grad_norm,
            **importance_diagnostics,
        },
        loss_history=None,
        log_evidence=float(log_evidence),
        _model=model,
    )
