# SPDX-License-Identifier: BSD-3-Clause
"""Metric preconditioning of the standardized latent space.

Every free parameter in tengri is standardized, so the sampled objective is

.. math::

    \\mathcal{H}(\\xi) = \\tfrac{1}{2}\\chi^2 + \\tfrac{1}{2}\\xi^\\top\\xi,

and the prior contributes *exactly* the identity to the metric. The remaining
curvature is the likelihood's, :math:`J^\\top N^{-1} J`, and on the correlated-field
posterior it is severe: :math:`\\mathrm{cond}(\\nabla^2 \\mathcal{H}) \\sim 10^5` at
the MAP. A diagonal mass matrix cannot cover that, and a dense one estimated from
warmup draws is both noisy and memory-hungry.

This module supplies the metric analytically instead, as a linear change of
variables :math:`\\xi = A\\zeta` with :math:`A A^\\top = G^{-1}`, so the curvature in
the sampled coordinates is :math:`A^\\top G A = I`. Because the map is linear its
Jacobian is a constant, so the sampled distribution is unchanged and draws are
recovered exactly by :math:`\\xi = A\\zeta` — no importance weights, no bias.

Notes
-----
This is the same metric NIFTy hands to MGVI/geoVI (``I + J^T N^-1 J``; see
``nifty8/re/evi.py``), the difference being that NIFTy recomputes it at every
iteration while a Hamiltonian sampler needs one fixed metric for the whole chain.

**The stiffness is not a corner case.** Measured across parametric and stochastic
SFHs, photometry / emission lines / spectroscopy, and D = 7 to 73, the raw posterior
condition number ran from :math:`8.5\\times10^4` to :math:`3.1\\times10^8` — every
configuration tested. Preconditioning whitened each to exactly 1.0 at the MAP.

**One fixed metric is a large improvement, not a complete one.** These posteriors are
genuinely non-Gaussian, so the curvature changes over the region a chain explores. One
posterior standard deviation away the whitened stiffness runs 3.7e2 to 1.7e5 —
an improvement of 16x to 1800x on the raw problem, never a regression, but not the
1.0 held at the expansion point. Closing that last gap needs a *position-dependent*
metric, which is exactly what MGVI/geoVI provide and a fixed mass matrix cannot.

References
----------
.. [1] J. Knollmüller & T. A. Enßlin, "Metric Gaussian Variational Inference,"
   arXiv:1901.11033 (2019).
.. [2] P. Frank, R. Leike & T. A. Enßlin, "Geometric Variational Inference,"
   Entropy, 23, 853 (2021). :doi:`10.3390/e23070853`
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

__all__ = [
    "PRECONDITION_MAX_DIM",
    "LinearPreconditioner",
    "PreconditionedProblem",
    "metric_preconditioner",
    "negative_hessian_metric",
    "preconditioned_logdensity",
    "prepare_preconditioning",
]

#: Smallest eigenvalue the metric may carry. The standardized prior contributes
#: exactly ``I`` and the Gauss-Newton likelihood term is positive semi-definite, so
#: a true metric eigenvalue below 1 can only be residual curvature — the term
#: Gauss-Newton drops. Flooring there keeps the metric positive definite without
#: needing the residual Jacobian.
PRIOR_METRIC_FLOOR: float = 1.0


def negative_hessian_metric(
    logdensity_fn: Callable,
    position: jnp.ndarray,
    data_args,
    *,
    floor: float = PRIOR_METRIC_FLOOR,
) -> jnp.ndarray:
    """Metric of a log-density at a point: :math:`-\\nabla^2 \\log p`, floored.

    Parameters
    ----------
    logdensity_fn : callable
        ``log_p(position, data_args) -> scalar`` in the standardized latent space.
    position : array_like, shape (D,)
        Flat latent position to expand around — normally the MAP.
    data_args : pytree
        Observed-data tensors, passed through to ``logdensity_fn``.
    floor : float, optional
        Lower clip applied to the eigenvalue *magnitudes* [dimensionless]. Default
        :data:`PRIOR_METRIC_FLOOR` (1.0). Pass ``0.0`` to disable clipping when
        the curvature is already known to be positive definite.

    Returns
    -------
    metric : ndarray, shape (D, D)
        Symmetric positive-definite metric, eigenvalues ``max(|lambda|, floor)``.

    Notes
    -----
    Scales by the **magnitude** of the curvature, so a direction of steep *negative*
    curvature is treated as steep rather than flat — the saddle-free Newton choice
    [3]_. This matters whenever the expansion point is not a true stationary point: a
    field fit whose MAP had not converged carried a ``-51`` eigenvalue, and flooring
    that to ``+1`` left the direction mis-scaled by 51x. Using ``|lambda|`` whitens it
    correctly. Where the point *is* stationary, all eigenvalues are positive and this
    is identical to a plain floor.

    **JIT/grad/vmap-safe**: builds the dense ``(D, D)`` Hessian via
    :func:`jax.hessian` (:math:`O(D)` backward passes) and symmetrizes it, then
    rescales the eigenvalues through an :func:`jax.numpy.linalg.eigh` reconstruction.
    Intended to be called once per fit, not inside a chain.

    References
    ----------
    .. [3] Y. N. Dauphin et al., "Identifying and attacking the saddle point problem
       in high-dimensional non-convex optimization," NeurIPS 27 (2014).
       arXiv:1406.2572.
    """
    hess = jax.hessian(lambda v: logdensity_fn(v, data_args))(jnp.asarray(position))
    metric = -0.5 * (hess + hess.T)
    eigenvalues, eigenvectors = jnp.linalg.eigh(metric)
    scale = jnp.maximum(jnp.abs(eigenvalues), floor)
    return (eigenvectors * scale) @ eigenvectors.T


@dataclass(frozen=True)
class LinearPreconditioner:
    """Linear change of latent variables :math:`\\xi = A\\zeta` that whitens curvature.

    Attributes
    ----------
    matrix : ndarray, shape (D, D)
        The map ``A`` from sampled coordinates to standardized latents.
    inverse : ndarray, shape (D, D)
        ``A^-1``, the map back.
    """

    matrix: jnp.ndarray
    inverse: jnp.ndarray

    def to_xi(self, zeta: jnp.ndarray) -> jnp.ndarray:
        """Map sampled coordinates to standardized latents (``xi = A zeta``)."""
        return self.matrix @ zeta

    def to_latent(self, xi: jnp.ndarray) -> jnp.ndarray:
        """Map standardized latents to sampled coordinates (``zeta = A^-1 xi``)."""
        return self.inverse @ xi

    def wrap(self, logdensity_fn: Callable) -> Callable:
        """Re-express a 2-arg log-density in the sampled coordinates.

        Parameters
        ----------
        logdensity_fn : callable
            ``log_p(xi, data_args) -> scalar``.

        Returns
        -------
        wrapped : callable
            ``log_p(A @ zeta, data_args) -> scalar``.

        Notes
        -----
        The constant Jacobian :math:`\\log|\\det A|` is dropped: it shifts the
        log-density by a constant and so leaves the sampled distribution
        untouched. Callers that need a normalized density (evidence estimates)
        must add it back.

        Samplers take ``logdensity_fn`` as a *static* argument because JAX keys
        its compilation cache on function identity, so a wrapped density
        recompiles once per preconditioner — build it once per fit and reuse it.
        """

        def wrapped(zeta, data_args):
            return logdensity_fn(self.matrix @ zeta, data_args)

        return wrapped


def metric_preconditioner(metric: jnp.ndarray) -> LinearPreconditioner:
    """Build the preconditioner that whitens a metric.

    Parameters
    ----------
    metric : array_like, shape (D, D)
        Symmetric positive-definite metric ``G``, e.g. from
        :func:`negative_hessian_metric`.

    Returns
    -------
    preconditioner : LinearPreconditioner
        Satisfies ``A A^T = G^-1``, hence ``A^T G A = I``.

    Raises
    ------
    ValueError
        If ``metric`` is not positive definite.

    Notes
    -----
    Uses the Cholesky factor :math:`G = L L^\\top` and sets :math:`A = L^{-\\top}`,
    so :math:`A A^\\top = (L L^\\top)^{-1} = G^{-1}` and :math:`A^{-1} = L^\\top`.
    Triangular, so both directions cost :math:`O(D^2)`.
    """
    metric = jnp.asarray(metric)
    # Non-finite FIRST, and separately. A NaN metric also fails Cholesky, so folding
    # the two together reported "not positive definite" for a matrix whose entries
    # were NaN — and advised building it with the very function that had just built
    # it (#1397). An eigenvalue floor cannot rescue NaN; only a finite input can.
    if not bool(jnp.all(jnp.isfinite(metric))):
        raise ValueError(
            "metric is not finite — it contains NaN or inf, so no factorization "
            "exists. The metric is the curvature at the expansion point, so this "
            "means the log-density or its second derivatives are undefined there. "
            "Check the point the metric was built at (a MAP fit ending in loss=nan "
            "is the usual cause), or run with precondition=False."
        )
    lower = jnp.linalg.cholesky(metric)
    if not bool(jnp.all(jnp.isfinite(lower))):
        raise ValueError(
            "metric is not positive definite — Cholesky failed on a finite matrix. "
            "Build it with `negative_hessian_metric`, whose eigenvalue floor "
            "guarantees positive definiteness."
        )
    identity = jnp.eye(metric.shape[0], dtype=metric.dtype)
    inverse_lower = jax.scipy.linalg.solve_triangular(lower, identity, lower=True)
    return LinearPreconditioner(matrix=inverse_lower.T, inverse=lower.T)


def preconditioned_logdensity(
    logdensity_fn: Callable,
    init_flat: jnp.ndarray,
    data_args,
    *,
    floor: float = PRIOR_METRIC_FLOOR,
) -> tuple[Callable, LinearPreconditioner, jnp.ndarray]:
    """Re-express a log-density in coordinates where its curvature is white.

    The one call a sampler backend needs: build the metric at ``init_flat``,
    derive the whitening map, and return the wrapped density together with the
    transported starting point.

    Parameters
    ----------
    logdensity_fn : callable
        ``log_p(xi, data_args) -> scalar`` in the standardized latent space.
    init_flat : array_like, shape (D,)
        Starting position — normally the MAP, which is where the metric is most
        representative of the region a chain explores.
    data_args : pytree
        Observed-data tensors, passed through to ``logdensity_fn``.
    floor : float, optional
        Eigenvalue floor for the metric. Default :data:`PRIOR_METRIC_FLOOR`.

    Returns
    -------
    wrapped : callable
        ``log_p(A @ zeta, data_args) -> scalar``, to be sampled in place of
        ``logdensity_fn``.
    preconditioner : LinearPreconditioner
        Map draws back with ``preconditioner.to_xi(zeta)``.
    init_zeta : ndarray, shape (D,)
        ``init_flat`` expressed in the sampled coordinates.

    Notes
    -----
    **Not JIT-safe** — the positive-definiteness check in
    :func:`metric_preconditioner` reads a concrete boolean and raises
    ``TracerBoolConversionError`` under trace. Call once per fit, outside any
    transform, then pass ``wrapped`` to the sampler as its static log-density.
    The returned ``wrapped`` callable is itself fully traceable.
    """
    init_flat = jnp.asarray(init_flat)
    # Diagnose the expansion point before the metric, so the error names the actual
    # defect. #1397: notebook 01 died reporting "not positive definite" when the real
    # chain was ``MAP init done (loss=nan)`` -> NaN Hessian -> NaN Cholesky. The point
    # is upstream of everything here and is the only thing the caller can act on.
    # ``bool`` rather than ``int``: under trace this raises TracerBoolConversionError,
    # the same signal ``metric_preconditioner``'s guard gives, so "you traced this" has
    # one failure mode instead of two. The count is only needed for the message.
    if bool(jnp.any(~jnp.isfinite(init_flat))):
        n_bad = int(jnp.sum(~jnp.isfinite(init_flat)))
        raise ValueError(
            f"cannot build a preconditioning metric: the expansion point is not "
            f"finite ({n_bad} of {init_flat.shape[0]} coordinates are NaN or inf). "
            f"The curvature there is undefined, so no metric exists. This is an "
            f"upstream initialization failure — a MAP fit ending in loss=nan is the "
            f"usual cause. Check data scaling and prior bounds, pass a finite "
            f"init_from, or run with precondition=False."
        )
    metric = negative_hessian_metric(logdensity_fn, init_flat, data_args, floor=floor)
    preconditioner = metric_preconditioner(metric)
    return (
        preconditioner.wrap(logdensity_fn),
        preconditioner,
        preconditioner.to_latent(init_flat),
    )


#: Largest ``D`` at which preconditioning has a measured cost profile. Above it the
#: ``O(D^3)`` factorization is untested, so an explicit request is honored but the cost
#: is the caller's to own. This is **advisory** — it does not enable anything.
#:
#: Measured on the field model (CPU, f64): the Hessian is **flat at ~2 s** from D=25 to
#: D=521 — ``jax.hessian`` is ``jacfwd(jacrev)``, which vectorizes rather than taking D
#: sequential backward passes — and ``eigh`` + Cholesky is 0.11 s with 2.2 MB of storage
#: at D=521. Only the ``O(D^3)`` factorization grows, so this sits an octave above the
#: largest configuration measured.
PRECONDITION_MAX_DIM: int = 1024


def _resolve_precondition(precondition: bool | None, n_dim: int) -> bool:
    """Resolve ``precondition=None`` — the opt-in policy.

    **Preconditioning is off unless asked for.** It shipped auto-on and the default
    broke fits that had been working (#1397):

    * ``notebooks/01_why_jax.py`` stopped running entirely. A NaN MAP init makes the
      metric non-finite, and the guard turned a working fit into a hard ``ValueError``.
    * ``notebooks/07_joint_photo_spec.py`` regressed from max R-hat 1.014 to **1.839**
      while reporting **zero** divergences — the health signal inverted, so a
      divergence check scores the broken arm as the healthiest of three.

    Set against that, no throughput win was ever demonstrated where these fits live:
    at 5 seeds per configuration the median was 1.87x ESS/s at D=7 but 0.84x — a
    *loss* — at D=8. The conditioning evidence (cond 8.5e4-3.1e8 whitened to 1.0 at
    the MAP, every configuration) is real and deterministic, but conditioning is not
    convergence, and only the latter is what a default is promising.

    A feature that can turn a converging fit into a non-converging one is a feature
    the caller has to ask for.

    Parameters
    ----------
    precondition : bool or None
        ``None`` (default — off), ``True`` (on), or ``False`` (off).
    n_dim : int
        Flat latent dimension. Unused by the policy; retained because callers pass it
        and because a future cost-based warning belongs here.

    Returns
    -------
    bool
        Effective setting. ``True`` only when explicitly requested.
    """
    if precondition is None:
        return False
    return bool(precondition)


@dataclass(frozen=True)
class PreconditionedProblem:
    """A log-density ready to sample, plus the map back — or a faithful identity.

    The single object a sampler backend needs. When preconditioning is off this is
    the identity in every field, so a backend has no ``if`` to write and no branch to
    forget.

    That last point is the reason this type exists. The transform is applied in one
    place and undone in another, hundreds of lines apart, and draws left in the
    whitened coordinates are finite, correctly shaped, and wrong — no exception, no
    warning, a posterior silently reported in the wrong basis. Three backends each
    carried their own copy of ``if preconditioner is not None: positions @ A.T``.

    Attributes
    ----------
    logdensity : callable
        ``log_p(zeta, data_args) -> scalar`` to hand the sampler. The original
        function itself when disabled.
    init_flat : ndarray, shape (D,)
        Starting position in the sampled coordinates.
    enabled : bool
        Whether the coordinates were actually whitened. Fold this into any adaptation
        cache key: a step size or mass matrix tuned in one basis is meaningless in
        the other.
    preconditioner : LinearPreconditioner or None
        The map, or ``None`` when disabled.
    """

    logdensity: Callable
    init_flat: jnp.ndarray
    enabled: bool
    preconditioner: LinearPreconditioner | None = None

    def restore(self, positions: jnp.ndarray) -> jnp.ndarray:
        """Map sampled draws back to the standardized latent space.

        Call this on the sampler's output before anything interprets it as
        parameters. Safe to call unconditionally — it is the identity when
        preconditioning is off.

        Parameters
        ----------
        positions : array_like, shape (D,) or (n_draw, D)
            Draws in the sampled coordinates.

        Returns
        -------
        ndarray, same shape as ``positions``
            Draws as standardized latents, ``xi = A zeta`` applied row-wise.

        Notes
        -----
        ``positions @ A.T`` is row-wise ``A @ v`` for a stack and reduces to
        ``A @ v`` for a single vector, so one expression covers both ranks.

        **JIT/grad/vmap-safe**: a single matrix product.
        """
        positions = jnp.asarray(positions)
        if self.preconditioner is None:
            return positions
        return positions @ self.preconditioner.matrix.T


def prepare_preconditioning(
    logdensity_fn: Callable,
    init_flat: jnp.ndarray,
    data_args,
    *,
    precondition: bool | None = None,
    floor: float = PRIOR_METRIC_FLOOR,
) -> PreconditionedProblem:
    """Resolve the auto-policy and build the whitened problem in one call.

    The seam every Hamiltonian backend shares. Backends declare support with
    ``register_backend(..., accepts_precondition=True)`` and otherwise do not
    special-case the feature.

    Parameters
    ----------
    logdensity_fn : callable
        ``log_p(xi, data_args) -> scalar`` in the standardized latent space.
    init_flat : array_like, shape (D,)
        Starting position — normally the MAP.
    data_args : pytree
        Observed-data tensors, passed through to ``logdensity_fn``.
    precondition : bool or None, optional
        ``None`` (default) resolves to **off** — the feature is opt-in
        (#1397, :func:`_resolve_precondition`); ``True`` enables, ``False``
        is explicit off.
    floor : float, optional
        Eigenvalue floor for the metric. Default :data:`PRIOR_METRIC_FLOOR`.

    Returns
    -------
    PreconditionedProblem
        Whitened when enabled, a faithful identity when not.

    Notes
    -----
    **Not JIT-safe** when enabled — see :func:`preconditioned_logdensity`. Call once
    per fit, outside any transform.
    """
    init_flat = jnp.asarray(init_flat)
    if not _resolve_precondition(precondition, init_flat.shape[0]):
        return PreconditionedProblem(logdensity=logdensity_fn, init_flat=init_flat, enabled=False)
    wrapped, preconditioner, init_zeta = preconditioned_logdensity(
        logdensity_fn, init_flat, data_args, floor=floor
    )
    return PreconditionedProblem(
        logdensity=wrapped,
        init_flat=init_zeta,
        enabled=True,
        preconditioner=preconditioner,
    )
