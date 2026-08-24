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
recovered exactly by :math:`\\xi = A\\zeta`, no importance weights, no bias.

Notes
-----
This is the same metric NIFTy hands to MGVI/geoVI (``I + J^T N^-1 J``; see
``nifty8/re/evi.py``), the difference being that NIFTy recomputes it at every
iteration while a Hamiltonian sampler needs one fixed metric for the whole chain.

**The stiffness is not a corner case.** Measured across parametric and stochastic
SFHs, photometry / emission lines / spectroscopy, and D = 7 to 73, the raw posterior
condition number ran from :math:`8.5\\times10^4` to :math:`3.1\\times10^8`, every
configuration tested. Preconditioning whitened each to exactly 1.0 at the MAP.

**One fixed metric is a large improvement, not a complete one.** These posteriors are
genuinely non-Gaussian, so the curvature changes over the region a chain explores. One
posterior standard deviation away the whitened stiffness runs 3.7e2 to 1.7e5,
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

import math
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

__all__ = [
    "DEFAULT_WHITENING_STRENGTH",
    "MAX_METRIC_CONDITION",
    "PRECONDITION_MAX_DIM",
    "LinearPreconditioner",
    "PreconditionedProblem",
    "metric_preconditioner",
    "negative_hessian_metric",
    "preconditioned_logdensity",
    "prepare_preconditioning",
    "temper_metric",
]

#: Smallest eigenvalue the metric may carry. The standardized prior contributes
#: exactly ``I`` and the Gauss-Newton likelihood term is positive semi-definite, so
#: a true metric eigenvalue below 1 can only be residual curvature, the term
#: Gauss-Newton drops. Flooring there keeps the metric positive definite without
#: needing the residual Jacobian.
PRIOR_METRIC_FLOOR: float = 1.0

#: Default whitening strength :math:`\alpha` in :math:`A A^\top = G^{-\alpha}`.
#:
#: **Not 1.0, deliberately** (#1442). Write the true precision as ``H`` and the metric
#: actually used as ``G = H^gamma`` (``gamma = 1`` is a perfect metric). The whitened
#: condition number is ``kappa(H) ** |1 - alpha*gamma|``, so whitening is worse than
#: doing nothing exactly when ``gamma > 2/alpha``. Full whitening therefore tolerates
#: only ``gamma <= 2``, and past that it *amplifies* ill-conditioning as
#: ``kappa^(gamma-1)``, unbounded, with no plateau.
#:
#: A single-point Hessian at the MAP is a *modal* curvature estimate. Wherever the
#: posterior is not Gaussian it is not the *bulk* curvature, so ``gamma != 1`` is the
#: normal case. Halving the exponent doubles the tolerated misspecification to
#: ``gamma <= 4`` and costs only a ``kappa ** 0.5`` residual when the metric is exact.
DEFAULT_WHITENING_STRENGTH: float = 0.5

#: Largest condition number the metric may carry into the factorization.
#:
#: A backstop, not the main mechanism, :data:`DEFAULT_WHITENING_STRENGTH` does that
#: work. This bounds the transform when the spectrum is pathological: the smallest
#: eigenvalues are the least reliably estimated and the most damaging when wrong.
#: Measured metrics on tengri field posteriors run ``1e5`` to ``3e8``, so this binds
#: rarely and only on the worst-conditioned directions.
MAX_METRIC_CONDITION: float = 1.0e8


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
    logdensity_fn: callable
        ``log_p(position, data_args) -> scalar`` in the standardized latent space.
    position: array_like, shape (D,)
        Flat latent position to expand around, normally the MAP.
    data_args: pytree
        Observed-data tensors, passed through to ``logdensity_fn``.
    floor: float, optional
        Lower clip applied to the eigenvalue *magnitudes* [dimensionless]. Default
        :data:`PRIOR_METRIC_FLOOR` (1.0). Pass ``0.0`` to disable clipping when
        the curvature is already known to be positive definite.

    Returns
    -------
    metric: ndarray, shape (D, D)
        Symmetric positive-definite metric, eigenvalues ``max(|lambda|, floor)``,
        **provided the curvature at ``position`` is finite**. The floor cannot
        rescue a non-finite input: ``jnp.maximum(nan, floor)`` is ``nan``, so a
        NaN log-density or Hessian propagates through unchanged (#1397).
        Callers that cannot guarantee a finite point should validate it first;
        :func:`preconditioned_logdensity` does.

    Notes
    -----
    Scales by the **magnitude** of the curvature, so a direction of steep *negative*
    curvature is treated as steep rather than flat, the saddle-free Newton choice
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


def temper_metric(
    metric: jnp.ndarray,
    *,
    strength: float,
    max_condition: float = MAX_METRIC_CONDITION,
) -> jnp.ndarray:
    """Raise a metric to a fractional power, capping its condition number.

    The one knob that bounds how much damage a wrong metric can do. Whitening with
    :math:`G^\\alpha` instead of :math:`G` interpolates continuously between no
    preconditioning (:math:`\\alpha = 0`) and full whitening (:math:`\\alpha = 1`).

    Parameters
    ----------
    metric: array_like, shape (D, D)
        Symmetric positive-definite metric ``G``, e.g. from
        :func:`negative_hessian_metric`.
    strength: float
        Exponent :math:`\\alpha \\in [0, 1]` [dimensionless]. ``1.0`` returns the
        metric unchanged, ``0.0`` returns the identity.
    max_condition: float, optional
        Largest permitted ratio of the metric's eigenvalues, applied **before** the
        exponent. Default :data:`MAX_METRIC_CONDITION`. Pass ``inf`` to disable.

    Returns
    -------
    tempered: ndarray, shape (D, D)
        ``G^strength`` with its spectrum clipped, symmetric positive definite.

    Raises
    ------
    ValueError
        If ``strength`` is NaN or outside ``[0, 1]``.

    Notes
    -----
    .. math:: G_\\alpha = V \\, \\mathrm{diag}(\\lambda_i^\\alpha) \\, V^\\top

    for the eigendecomposition :math:`G = V \\Lambda V^\\top`, with
    :math:`\\lambda_i` first clipped below at
    :math:`\\lambda_{\\max} / \\kappa_{\\max}`. The eigenvectors are untouched, so
    tempering rescales the geometry without rotating it.

    Whitening the result gives a transform with
    :math:`A A^\\top = G^{-\\alpha}`. For a true precision ``H`` and a metric
    ``G = H^\\gamma``, the eigenvalues of the whitened precision
    :math:`A^\\top H A` are the generalized eigenvalues of the pencil
    :math:`(H, G^\\alpha)`, hence

    .. math:: \\kappa_{\\rm whitened} = \\kappa(H)^{|1 - \\alpha\\gamma|}

    which exceeds :math:`\\kappa(H)`, worse than no preconditioning at all, exactly
    when :math:`\\gamma > 2/\\alpha`. See :data:`DEFAULT_WHITENING_STRENGTH` (#1442).

    **Not JIT-safe**: validates ``strength`` as a concrete Python float. The
    eigendecomposition itself is traceable; the guard is not.
    """
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError(
            f"whitening strength must be a finite number in [0, 1], got {strength!r}. "
            f"0 is no preconditioning, 1 is full whitening; the default "
            f"{DEFAULT_WHITENING_STRENGTH} bounds the damage from a misspecified metric."
        )
    metric = jnp.asarray(metric)
    eigenvalues, eigenvectors = jnp.linalg.eigh(metric)
    if math.isfinite(max_condition):
        eigenvalues = jnp.maximum(eigenvalues, jnp.max(eigenvalues) / max_condition)
    return (eigenvectors * eigenvalues**strength) @ eigenvectors.T


@dataclass(frozen=True)
class LinearPreconditioner:
    """Linear change of latent variables :math:`\\xi = A\\zeta` that whitens curvature.

    Attributes
    ----------
    matrix: ndarray, shape (D, D)
        The map ``A`` from sampled coordinates to standardized latents.
    inverse: ndarray, shape (D, D)
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
        logdensity_fn: callable
            ``log_p(xi, data_args) -> scalar``.

        Returns
        -------
        wrapped: callable
            ``log_p(A @ zeta, data_args) -> scalar``.

        Notes
        -----
        The constant Jacobian :math:`\\log|\\det A|` is dropped: it shifts the
        log-density by a constant and so leaves the sampled distribution
        untouched. Callers that need a normalized density (evidence estimates)
        must add it back.

        Samplers take ``logdensity_fn`` as a *static* argument because JAX keys
        its compilation cache on function identity, so a wrapped density
        recompiles once per preconditioner, build it once per fit and reuse it.
        """

        def wrapped(zeta, data_args):
            return logdensity_fn(self.matrix @ zeta, data_args)

        return wrapped


def metric_preconditioner(metric: jnp.ndarray) -> LinearPreconditioner:
    """Build the preconditioner that whitens a metric.

    Parameters
    ----------
    metric: array_like, shape (D, D)
        Symmetric positive-definite metric ``G``, e.g. from
        :func:`negative_hessian_metric`.

    Returns
    -------
    preconditioner: LinearPreconditioner
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
    # Distinguish the two ways Cholesky can fail (#1397). A non-finite metric is
    # NOT an indefiniteness problem, and reporting it as one sends the reader to
    # fix curvature when the defect is upstream, the metric was formed at a
    # point where the log-density or its Hessian was already NaN. The eigenvalue
    # floor in ``negative_hessian_metric`` cannot help there: ``jnp.maximum(nan,
    # floor)`` is ``nan``, so the floor is a no-op exactly when it is needed.
    if not bool(jnp.all(jnp.isfinite(metric))):
        raise ValueError(
            "metric is non-finite (NaN or inf), so it cannot be factorized. This "
            "is an upstream failure, not a curvature one: the metric is built at "
            "the expansion point (normally the MAP), so a non-finite value there "
            ", a diverged MAP, or a log-density that is NaN at that point, "
            "propagates straight into the metric. The eigenvalue floor cannot "
            "repair it. Check the initial point is finite, or pass "
            "precondition=False to sample without whitening."
        )
    lower = jnp.linalg.cholesky(metric)
    if not bool(jnp.all(jnp.isfinite(lower))):
        raise ValueError(
            "metric is not positive definite, Cholesky failed on a finite "
            "matrix. Build it with `negative_hessian_metric`, whose eigenvalue "
            "floor guarantees positive definiteness for finite curvature, or "
            "pass precondition=False to sample without whitening."
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
    strength: float = DEFAULT_WHITENING_STRENGTH,
    max_condition: float = MAX_METRIC_CONDITION,
) -> tuple[Callable, LinearPreconditioner, jnp.ndarray]:
    """Re-express a log-density in coordinates where its curvature is white.

    The one call a sampler backend needs: build the metric at ``init_flat``,
    derive the whitening map, and return the wrapped density together with the
    transported starting point.

    Parameters
    ----------
    logdensity_fn: callable
        ``log_p(xi, data_args) -> scalar`` in the standardized latent space.
    init_flat: array_like, shape (D,)
        Starting position, normally the MAP, which is where the metric is most
        representative of the region a chain explores.
    data_args: pytree
        Observed-data tensors, passed through to ``logdensity_fn``.
    floor: float, optional
        Eigenvalue floor for the metric. Default :data:`PRIOR_METRIC_FLOOR`.
    strength: float, optional
        Whitening exponent :math:`\\alpha` in :math:`A A^\\top = G^{-\\alpha}`
        [dimensionless]. Default :data:`DEFAULT_WHITENING_STRENGTH`. See
        :func:`temper_metric` for why the default is not 1.
    max_condition: float, optional
        Condition-number cap on the metric. Default :data:`MAX_METRIC_CONDITION`.

    Returns
    -------
    wrapped: callable
        ``log_p(A @ zeta, data_args) -> scalar``, to be sampled in place of
        ``logdensity_fn``.
    preconditioner: LinearPreconditioner
        Map draws back with ``preconditioner.to_xi(zeta)``.
    init_zeta: ndarray, shape (D,)
        ``init_flat`` expressed in the sampled coordinates.

    Notes
    -----
    **Not JIT-safe**, the positive-definiteness check in
    :func:`metric_preconditioner` reads a concrete boolean and raises
    ``TracerBoolConversionError`` under trace. Call once per fit, outside any
    transform, then pass ``wrapped`` to the sampler as its static log-density.
    The returned ``wrapped`` callable is itself fully traceable.
    """
    init_flat = jnp.asarray(init_flat)
    _reject_nonfinite_expansion_point(init_flat)
    preconditioner, _ = _preconditioner_with_conditioning(
        logdensity_fn,
        init_flat,
        data_args,
        floor=floor,
        strength=strength,
        max_condition=max_condition,
    )
    return (
        preconditioner.wrap(logdensity_fn),
        preconditioner,
        preconditioner.to_latent(init_flat),
    )


def _reject_nonfinite_expansion_point(init_flat: jnp.ndarray) -> None:
    """Refuse a NaN or inf expansion point, naming the actual defect.

    #1397: notebook 01 died reporting "not positive definite" when the real chain was
    ``MAP init done (loss=nan)`` -> NaN Hessian -> NaN Cholesky. The point is upstream
    of everything here and is the only thing the caller can act on.

    Notes
    -----
    Caught here, where the point is still in hand. Downstream all that survives is a
    NaN matrix, and the failure arrives wearing the wrong name three layers later. The
    check lives in this helper rather than in :func:`negative_hessian_metric` because
    that function is documented JIT/grad/vmap-safe and reading a concrete boolean
    inside it would break that contract; every caller of this one is already non-JIT.

    ``bool`` rather than ``int`` on the gate: under trace this raises
    ``TracerBoolConversionError``, the same signal ``metric_preconditioner``'s guard
    gives, so "you traced this" has one failure mode instead of two. The count is only
    computed for the message.
    """
    if bool(jnp.any(~jnp.isfinite(init_flat))):
        n_bad = int(jnp.sum(~jnp.isfinite(init_flat)))
        raise ValueError(
            f"the expansion point is non-finite ({n_bad} of {init_flat.size} "
            "coordinates are NaN or inf), so no metric can be built at it. This "
            "normally means the MAP initialization diverged. Fix the starting "
            "point, pass an explicit init_from=, or adjust the MAP settings, "
            "or pass precondition=False to sample without whitening."
        )


def _preconditioner_with_conditioning(
    logdensity_fn: Callable,
    init_flat: jnp.ndarray,
    data_args,
    *,
    floor: float,
    strength: float,
    max_condition: float,
) -> tuple[LinearPreconditioner, tuple[float, float]]:
    """Build the transform and measure the geometry either side of it.

    Returns
    -------
    preconditioner: LinearPreconditioner
    conditioning: tuple of float
        ``(raw, whitened)`` condition numbers [dimensionless], the metric as built,
        and the curvature the sampler actually faces at the expansion point.

    Notes
    -----
    The whitened value is computed from the product rather than predicted as
    ``raw ** (1 - strength)``, because the two differ whenever ``max_condition``
    binds. One extra ``eigvalsh`` on a ``(D, D)`` matrix is ~0.1 s at D = 521,
    negligible against the fit it is reporting on, and an honest diagnostic is worth
    more than a cheap one.
    """
    metric = negative_hessian_metric(logdensity_fn, init_flat, data_args, floor=floor)
    raw = jnp.linalg.eigvalsh(metric)
    tempered = temper_metric(metric, strength=strength, max_condition=max_condition)
    preconditioner = metric_preconditioner(tempered)
    whitened = preconditioner.matrix.T @ metric @ preconditioner.matrix
    whitened_eigenvalues = jnp.linalg.eigvalsh(0.5 * (whitened + whitened.T))
    return preconditioner, (
        float(jnp.max(raw) / jnp.min(raw)),
        float(jnp.max(whitened_eigenvalues) / jnp.min(whitened_eigenvalues)),
    )


#: Largest ``D`` at which preconditioning has a measured cost profile. Above it the
#: ``O(D^3)`` factorization is untested, so an explicit request is honored but the cost
#: is the caller's to own. This is **advisory**, it does not enable anything.
#:
#: Measured on the field model (CPU, f64): the Hessian is **flat at ~2 s** from D=25 to
#: D=521, ``jax.hessian`` is ``jacfwd(jacrev)``, which vectorizes rather than taking D
#: sequential backward passes, and ``eigh`` + Cholesky is 0.11 s with 2.2 MB of storage
#: at D=521. Only the ``O(D^3)`` factorization grows, so this sits an octave above the
#: largest configuration measured.
PRECONDITION_MAX_DIM: int = 1024


def _resolve_whitening_strength(precondition: bool | float | None, n_dim: int) -> float | None:
    """Resolve ``precondition`` into a whitening strength, or ``None`` for off.

    Carries both the switch and the dial, because they are the same decision: a
    strength of zero *is* "off", and a fit that cannot report how hard it whitened
    cannot be compared against one that whitened differently.

    **Preconditioning is off unless asked for.**

    Note on attribution, since the first version of this docstring got it wrong: the
    notebook failures originally blamed on preconditioning (#1397) were **not** caused
    by it. A sub-band node gradient underflowed to NaN inside the model itself, and
    preconditioning was merely the first thing to notice, it built a metric at the
    poisoned point and refused. With that root cause fixed, ``notebooks/07`` returns to
    the R-hat it had before preconditioning existed.

    What survives, measured on the *fixed* code, is that whitening does not pay at low
    D. On ``recipes.mock_recovery_minimal()`` (D=7), 4 seeds of 4 have the
    unpreconditioned arm converging (R-hat 0.997-1.007) and the preconditioned arm not
    (1.055-2.689), at 4x to 25x worse ESS/s. An earlier sweep on other configurations
    agreed in direction: median 0.84x ESS/s at D=8.

    The conditioning evidence (cond 8.5e4-3.1e8 whitened to 1.0 at the MAP, every
    configuration measured) is real and deterministic. But conditioning is not
    convergence, and only convergence is what a default would be promising.

    Parameters
    ----------
    precondition: bool, float or None
        ``None`` / ``False`` (default, off), ``True`` (on at
        :data:`DEFAULT_WHITENING_STRENGTH`), or a float in ``[0, 1]` naming the
        strength directly. ``1.0`` is full whitening; ``0.0`` is off.
    n_dim: int
        Flat latent dimension. Unused by the policy; retained because callers pass it
        and because a future cost-based warning belongs here.

    Returns
    -------
    float or None
        Whitening strength, or ``None`` when preconditioning is off.

    Raises
    ------
    ValueError
        If a numeric ``precondition`` is NaN or outside ``[0, 1]``.
    """
    if precondition is None or precondition is False:
        return None
    if precondition is True:
        return DEFAULT_WHITENING_STRENGTH
    try:
        strength = float(precondition)
    except (TypeError, ValueError):
        raise ValueError(
            f"precondition must be True, False, None, or a number in [0, 1]; got {precondition!r}"
        ) from None
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError(
            f"precondition must be True, False, None, or a number in [0, 1]; "
            f"got {precondition!r}. The value is the whitening strength alpha in "
            f"A A^T = G^-alpha: 0 is off, 1 is full whitening, and the default "
            f"{DEFAULT_WHITENING_STRENGTH} bounds the damage from a misspecified "
            f"metric (#1442)."
        )
    # Zero strength is the identity transform. Returning it as "on" would build a
    # Hessian, factorize it, and multiply by a matrix that is I -- all cost, no effect.
    return None if strength == 0.0 else strength


@dataclass(frozen=True)
class PreconditionedProblem:
    """A log-density ready to sample, plus the map back, or a faithful identity.

    The single object a sampler backend needs. When preconditioning is off this is
    the identity in every field, so a backend has no ``if`` to write and no branch to
    forget.

    That last point is the reason this type exists. The transform is applied in one
    place and undone in another, hundreds of lines apart, and draws left in the
    whitened coordinates are finite, correctly shaped, and wrong, no exception, no
    warning, a posterior silently reported in the wrong basis. Three backends each
    carried their own copy of ``if preconditioner is not None: positions @ A.T``.

    Attributes
    ----------
    logdensity: callable
        ``log_p(zeta, data_args) -> scalar`` to hand the sampler. The original
        function itself when disabled.
    init_flat: ndarray, shape (D,)
        Starting position in the sampled coordinates.
    enabled: bool
        Whether the coordinates were actually whitened. Fold this into any adaptation
        cache key: a step size or mass matrix tuned in one basis is meaningless in
        the other.
    preconditioner: LinearPreconditioner or None
        The map, or ``None`` when disabled.
    strength: float or None
        The whitening exponent actually applied, or ``None`` when disabled. Report it:
        two fits whitened at different strengths are not comparable, and a number that
        is not recorded will be assumed to have been the default (#1442).
    metric_condition: float or None
        Condition number of the metric as built, before tempering [dimensionless].
        ``None`` when disabled.
    whitened_condition: float or None
        Condition number the sampler actually faces at the expansion point
        [dimensionless]. ``None`` when disabled. The ratio of the two is what the
        transform bought; without them a run cannot say whether the metric was
        excellent or useless, which is how a 58x spread in throughput stayed
        invisible from inside a fit.
    """

    logdensity: Callable
    init_flat: jnp.ndarray
    enabled: bool
    preconditioner: LinearPreconditioner | None = None
    strength: float | None = None
    metric_condition: float | None = None
    whitened_condition: float | None = None

    @property
    def cache_key(self) -> tuple:
        """Hashable summary of the coordinates, for an adaptation cache key.

        Returns
        -------
        tuple
            ``("whiten", strength)``, distinct for every strength and for off.

        Notes
        -----
        Fold this into any cached warmup result. A step size or mass matrix tuned in
        one basis is meaningless in another, and the failure is silent: a stale step
        size is a finite float that samples happily and badly. Keying on a bare
        ``enabled`` boolean let two fits at different strengths share one (#1442).
        """
        return ("whiten", self.strength)

    def restore(self, positions: jnp.ndarray) -> jnp.ndarray:
        """Map sampled draws back to the standardized latent space.

        Call this on the sampler's output before anything interprets it as
        parameters. Safe to call unconditionally, it is the identity when
        preconditioning is off.

        Parameters
        ----------
        positions: array_like, shape (D,) or (n_draw, D)
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
    precondition: bool | float | None = None,
    floor: float = PRIOR_METRIC_FLOOR,
    max_condition: float = MAX_METRIC_CONDITION,
) -> PreconditionedProblem:
    """Resolve the auto-policy and build the whitened problem in one call.

    The seam every Hamiltonian backend shares. Backends declare support with
    ``register_backend(..., accepts_precondition=True)`` and otherwise do not
    special-case the feature.

    Parameters
    ----------
    logdensity_fn: callable
        ``log_p(xi, data_args) -> scalar`` in the standardized latent space.
    init_flat: array_like, shape (D,)
        Starting position, normally the MAP.
    data_args: pytree
        Observed-data tensors, passed through to ``logdensity_fn``.
    precondition: bool, float or None, optional
        ``None`` (default) and ``False`` are off, preconditioning is opt-in
        (#1397, :func:`_resolve_whitening_strength`). ``True`` enables it at
        :data:`DEFAULT_WHITENING_STRENGTH`. A float in ``[0, 1]`` names the
        whitening strength directly; ``1.0`` is full whitening and ``0.0`` is off.
    floor: float, optional
        Eigenvalue floor for the metric. Default :data:`PRIOR_METRIC_FLOOR`.
    max_condition: float, optional
        Condition-number cap on the metric. Default :data:`MAX_METRIC_CONDITION`.

    Returns
    -------
    PreconditionedProblem
        Whitened when enabled, a faithful identity when not.

    Notes
    -----
    **Not JIT-safe** when enabled, see :func:`preconditioned_logdensity`. Call once
    per fit, outside any transform.
    """
    init_flat = jnp.asarray(init_flat)
    strength = _resolve_whitening_strength(precondition, init_flat.shape[0])
    if strength is None:
        return PreconditionedProblem(logdensity=logdensity_fn, init_flat=init_flat, enabled=False)
    _reject_nonfinite_expansion_point(init_flat)
    preconditioner, (raw, whitened) = _preconditioner_with_conditioning(
        logdensity_fn,
        init_flat,
        data_args,
        floor=floor,
        strength=strength,
        max_condition=max_condition,
    )
    return PreconditionedProblem(
        logdensity=preconditioner.wrap(logdensity_fn),
        init_flat=preconditioner.to_latent(init_flat),
        enabled=True,
        preconditioner=preconditioner,
        strength=strength,
        metric_condition=raw,
        whitened_condition=whitened,
    )
