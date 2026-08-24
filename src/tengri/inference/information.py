# SPDX-License-Identifier: BSD-3-Clause
"""How much of a posterior is measurement, and how much is the prior talking back.

A star-formation history with one free parameter per age bin can draw almost
any history, so a fit always *returns* one, whether or not the data said
anything about it. The same is true of any flexible block: reading a posterior
median without knowing which directions the data constrained is how a prior
draw gets reported as a measurement.

The separating quantity is the per-mode **shrinkage**. Every prior in tengri is
standardized, so the prior contributes exactly ``I`` to the posterior precision
(the reasoning recorded at
:data:`~tengri.inference.preconditioning.PRIOR_METRIC_FLOOR`), and the
Gauss-Newton likelihood term is positive semi-definite. An eigenvalue of the
posterior precision therefore decomposes with **no free normalization**:

.. math::

    \\lambda_k = 1 + d_k, \\qquad
    s_k = \\frac{\\lambda_k - 1}{\\lambda_k} = \\frac{d_k}{1 + d_k}, \\qquad
    n_{\\rm eff} = \\sum_k s_k

where :math:`d_k \\ge 0` [dimensionless] is what the data added along mode
:math:`k`, :math:`s_k \\in [0, 1]` is the fraction of that mode's precision the
data supplied, and :math:`n_{\\rm eff}` [dimensionless] counts the directions
actually measured. There is no threshold to choose and no basis to prefer:
:math:`n_{\\rm eff}` is a sum over eigenvalues, hence invariant under any
orthogonal change of parameters.

The practical reading: compare ``n_eff`` to the number of free parameters. A
64-node field with ``n_eff`` near 4 has measured four things and drawn the rest
from the prior, which makes the node count a *smoothness* choice, not a
resolution one.

Notes
-----
The metric is evaluated with ``floor=0.0``. The default
:data:`~tengri.inference.preconditioning.PRIOR_METRIC_FLOOR` of 1.0 exists to
keep the whitening transform positive definite, and it clips away exactly the
prior-carried modes this module is counting. The clip to non-negative shrinkage
therefore happens here instead, eigenvalues below 1 are the residual curvature
term Gauss-Newton drops, not evidence.

References
----------
.. [1] Wang, B., et al. (2025). "Quantifying the Information Content of
       Star Formation Histories." arXiv:2503.06229
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tengri.config.exceptions import warn_measured

__all__ = [
    "MODE_TOLERANCE_NATS",
    "ParameterInformation",
    "information_from_precision",
    "latent_names",
    "parameter_information",
]

#: Remaining log-posterior improvement, in nats, still counted as "at the mode".
#:
#: Estimated by half the squared Newton decrement, :math:`\\tfrac12 g^T H^{-1} g`
#: which is scale-free, so it means the same thing whether a mass is carried in dex or
#: in log10. Half a nat is well inside the noise of any posterior worth
#: summarizing, and far below the gap that matters: a MAP stopped early measured
#: ``n_eff`` = 17.6 where the converged fit gives 6.3.
MODE_TOLERANCE_NATS: float = 0.5


@dataclass(frozen=True)
class ParameterInformation:
    """The measured-versus-prior decomposition of one posterior.

    Attributes
    ----------
    names : tuple of str
        Free-parameter names, in the order of the precision matrix.
    eigenvalues : ndarray, shape (n_free,)
        Posterior precision per mode [dimensionless], ascending. The
        standardized prior contributes 1 to each.
    shrinkage : ndarray, shape (n_free,)
        Fraction of each mode's precision supplied by the data
        [dimensionless], in ``[0, 1]``, aligned with ``eigenvalues``.
    directions : ndarray, shape (n_free, n_free)
        Eigenvectors as **columns**, aligned with ``eigenvalues``.
    """

    names: tuple[str, ...]
    eigenvalues: np.ndarray
    shrinkage: np.ndarray
    directions: np.ndarray
    newton_decrement: float = float("nan")

    @property
    def n_eff(self) -> float:
        """Effective number of directions the data measured [dimensionless]."""
        return float(self.shrinkage.sum())

    @property
    def at_a_mode(self) -> bool:
        """Whether the expansion point is close enough to a mode to trust this.

        Returns
        -------
        bool
            ``True`` when the Newton decrement puts the remaining log-posterior
            improvement below :data:`MODE_TOLERANCE_NATS`. ``True`` also when no
            gradient was supplied, since there is then nothing to contradict.

        Notes
        -----
        Curvature is the posterior precision **only at a stationary point**.
        Evaluated part-way through an optimization it is just the local
        second derivative, and the resulting ``n_eff`` is meaningless while
        looking entirely reasonable, measured at 17.6 against a converged 6.3
        on the same model and data.
        """
        if not np.isfinite(self.newton_decrement):
            return True
        return bool(0.5 * self.newton_decrement**2 < MODE_TOLERANCE_NATS)

    @property
    def n_total(self) -> int:
        """Number of free parameters, i.e. the ceiling on ``n_eff``."""
        return len(self.names)

    def precision(self) -> np.ndarray:
        """Rebuild the precision matrix this decomposition came from.

        Returns
        -------
        ndarray, shape (n_free, n_free)
            :math:`V \\Lambda V^T` [dimensionless]. Exact to floating point,
            the eigendecomposition of a symmetric matrix loses nothing.
        """
        return (self.directions * self.eigenvalues) @ self.directions.T

    def restrict(self, prefix: str) -> ParameterInformation:
        """Information content of one block of parameters, on its own.

        Parameters
        ----------
        prefix : str
            Name prefix selecting the block, e.g. ``"psd_xi"`` for the
            stochastic field or ``"dust_"`` for the attenuation parameters.

        Returns
        -------
        ParameterInformation
            Decomposition of the corresponding **sub-block** of the precision
            matrix.

        Raises
        ------
        ValueError
            If no parameter name starts with ``prefix``.

        Notes
        -----
        This takes the sub-block of the precision matrix and re-diagonalizes it.
        It is **not** a slice of the full decomposition, and the two answer
        different questions:

        * ``n_eff`` of the whole model counts every measured direction,
          including ones that mix the block with everything else;
        * ``restrict(...).n_eff`` counts what the data measured about this block
          **with the rest of the model held fixed**, the conditional, not the
          marginal, so it is an upper bound on the block's own information.

        Comparing the two across papers without saying which was used is how
        "the SFH has 4 modes" and "the fit has 5 modes" turn into an argument.
        For the field the block is ``psd_xi``: components spell the same
        quantity ``sfh_field_xi``, but the *latent* dict, which is what gets
        raveled, keys it ``psd_xi`` (the #1271 two-spelling seam).

        Examples
        --------
        >>> info = posterior.information()  # doctest: +SKIP
        >>> info.n_eff, info.restrict("psd_xi").n_eff  # doctest: +SKIP
        (4.91, 3.21)
        """
        keep = [i for i, name in enumerate(self.names) if name.startswith(prefix)]
        if not keep:
            raise ValueError(
                f"No parameter starts with {prefix!r}. Available: "
                f"{sorted({n.split('[')[0] for n in self.names})}"
            )
        index = np.asarray(keep)
        block = self.precision()[np.ix_(index, index)]
        return information_from_precision(block, names=tuple(self.names[i] for i in keep))

    def by_parameter(self) -> dict[str, float]:
        """Attribute the measured directions back to named parameters.

        Returns
        -------
        dict of str to float
            Each parameter's share of ``n_eff`` [dimensionless].

        Notes
        -----
        Mode ``k`` is spread over parameters by its squared projection
        :math:`v_{ki}^2`. Because the eigenvectors are orthonormal, each mode's
        projections sum to 1, so the shares sum to ``n_eff`` exactly, the
        attribution redistributes the total rather than inventing any of it.

        A mode is a *combination* of parameters, so a large share means the data
        constrained something that parameter participates in, not that the
        parameter is individually identified. Two perfectly degenerate
        parameters split one measured direction evenly.
        """
        shares = (self.directions**2) @ self.shrinkage
        return {name: float(v) for name, v in zip(self.names, shares)}

    def summary(self) -> str:
        """One-screen report: the total, then the worst-measured parameters.

        Returns
        -------
        str
            Formatted table. Printing it is the intended use.
        """
        shares = self.by_parameter()
        order = sorted(shares, key=lambda k: shares[k])
        lines = [
            f"Information content: n_eff = {self.n_eff:.2f} of "
            f"{self.n_total} free parameters "
            f"({self.n_eff / max(self.n_total, 1):.0%} of the model measured)",
        ]
        if not self.at_a_mode:
            lines.append(
                f"  ** NOT AT A MODE ({0.5 * self.newton_decrement**2:.1f} nats "
                f"unclaimed), this number is not meaningful. **"
            )
        lines += [
            "",
            f"  {'parameter':<34s} {'share of n_eff':>14s}",
            f"  {'-' * 34} {'-' * 14}",
        ]
        lines += [f"  {name:<34s} {shares[name]:>14.3f}" for name in order]
        lines += [
            "",
            "  A share near 0 means that parameter's posterior is the prior.",
        ]
        return "\n".join(lines)


def information_from_precision(
    precision: np.ndarray,
    names: tuple[str, ...],
    gradient: np.ndarray | None = None,
) -> ParameterInformation:
    """Decompose a posterior precision matrix into measured and prior-carried modes.

    Parameters
    ----------
    precision : array_like, shape (n_free, n_free)
        Posterior precision :math:`-\\nabla^2 \\log p` in the **standardized**
        latent space [dimensionless]. Symmetry is assumed; the symmetric part is
        used.
    names : tuple of str
        Free-parameter names, one per row.
    gradient : array_like, shape (n_free,), optional
        Gradient of the **negative** log-posterior at the same point
        [dimensionless]. Supplied, it sets ``newton_decrement`` and so lets
        :attr:`ParameterInformation.at_a_mode` report whether the expansion
        point was actually a mode. Default ``None`` leaves that unchecked.

    Returns
    -------
    ParameterInformation
        The decomposition.

    Raises
    ------
    ValueError
        If ``precision`` is not square, contains non-finite entries, or
        ``names`` has the wrong length.

    Notes
    -----
    Not JIT-compatible: eigendecomposition of a materialized matrix, called once
    per fit rather than in any inner loop. Uses NumPy deliberately, the result
    is a diagnostic, never a term in a gradient.

    Shrinkage is clipped into ``[0, 1]``. The lower clip catches eigenvalues
    below 1, which are the residual curvature Gauss-Newton drops rather than
    negative information; see the module docstring.
    """
    precision = np.asarray(precision, dtype=np.float64)
    if precision.ndim != 2 or precision.shape[0] != precision.shape[1]:
        raise ValueError(f"precision must be a square matrix, got shape {precision.shape}")
    if not np.all(np.isfinite(precision)):
        n_bad = int((~np.isfinite(precision)).sum())
        raise ValueError(
            f"precision contains {n_bad} non-finite entries. A NaN or Inf Hessian "
            "means the log-posterior is not twice-differentiable at this point, "
            "usually a parameter railed against a bound, or a NaN in the forward "
            "model. Averaging over it would return a confident-looking n_eff for "
            "a posterior that has none."
        )
    if len(names) != precision.shape[0]:
        raise ValueError(
            f"names has {len(names)} entries but precision is "
            f"{precision.shape[0]}x{precision.shape[0]}"
        )

    symmetric = 0.5 * (precision + precision.T)
    eigenvalues, directions = np.linalg.eigh(symmetric)
    with np.errstate(divide="ignore", invalid="ignore"):
        shrinkage = np.where(eigenvalues > 0.0, (eigenvalues - 1.0) / eigenvalues, 0.0)
    shrinkage = np.clip(shrinkage, 0.0, 1.0)

    decrement = float("nan")
    if gradient is not None:
        gradient = np.asarray(gradient, dtype=np.float64).ravel()
        if gradient.size != symmetric.shape[0]:
            raise ValueError(
                f"gradient has {gradient.size} entries but precision is "
                f"{symmetric.shape[0]}x{symmetric.shape[0]}"
            )
        # g^T H^-1 g through the eigenbasis, so the near-null directions that make
        # a direct solve unstable are simply skipped rather than amplified.
        projected = directions.T @ gradient
        usable = eigenvalues > 0.0
        decrement = float(np.sqrt(np.sum(projected[usable] ** 2 / eigenvalues[usable])))

    return ParameterInformation(
        names=tuple(names),
        eigenvalues=eigenvalues,
        shrinkage=shrinkage,
        directions=directions,
        newton_decrement=decrement,
    )


def parameter_information(target, params=None, *, key=None) -> ParameterInformation:
    """Measure how much of a posterior the data determined, mode by mode.

    Parameters
    ----------
    target : Posterior or InferenceContext or Fitter
        A completed fit, or a context to expand around ``params``.
    params : dict of str to float, optional
        Point to expand the posterior around, in **physical** units. Default
        ``None`` uses the posterior's own point estimate, which is where the
        quadratic approximation is tightest.
    key : jax.Array, optional
        PRNG key, used only when ``target`` supplies no point estimate and an
        initial position must be drawn.

    Returns
    -------
    ParameterInformation
        ``n_eff``, per-mode shrinkage, and the per-parameter attribution.

    Raises
    ------
    ValueError
        If the Hessian at the expansion point is not finite.

    Notes
    -----
    Not JIT-compatible, and not cheap: one dense Hessian of the log-posterior,
    which costs :math:`O(D)` gradient evaluations. Call it once on a finished
    fit, never inside a sampling loop.

    The expansion is a Laplace approximation, so the answer describes the
    posterior *near* the given point. For a strongly multimodal posterior it
    characterizes the mode it was handed, which is the same caveat that applies
    to any curvature-based error bar.

    Examples
    --------
    >>> post = model.fit(data, noise, method="map")  # doctest: +SKIP
    >>> info = parameter_information(post)  # doctest: +SKIP
    >>> print(info.summary())  # doctest: +SKIP
    >>> info.n_eff  # doctest: +SKIP
    4.17
    """
    import jax

    from tengri.inference.backends.mcmc._shared import _get_flat_logdensity
    from tengri.inference.context import InferenceContext
    from tengri.inference.preconditioning import negative_hessian_metric

    posterior = None
    if hasattr(target, "params") and hasattr(target, "method"):  # a finished fit
        posterior = target
        target = posterior._fitter
        if target is None:
            raise ValueError(
                "This Posterior carries no fit context, so the log-posterior it came "
                "from cannot be re-evaluated. Information content needs the data and "
                "the noise model, not just the samples, pass the object returned "
                "directly by model.fit(...)."
            )

    if not hasattr(target, "_engine_cache_key"):
        # InferenceContext.from_target() wraps anything at all as a Fitter, so an
        # SEDModel would sail through and fail much later with an unrelated
        # AttributeError. A model on its own genuinely cannot answer this: with no
        # data there is no likelihood, and every shrinkage would be zero.
        raise TypeError(
            f"parameter_information() needs a fit, not a {type(target).__name__}. "
            "Information content is a property of a posterior, how much the data "
            "moved the prior, so there is nothing to measure before fitting. "
            "Pass the Posterior from model.fit(data, noise, ...)."
        )

    context = InferenceContext.from_target(target)

    if key is None:
        key = jax.random.PRNGKey(0)
    # Always go through initial_params, even when the expansion point is known.
    # It is what defines the latent pytree the backends actually sample, and that
    # is *not* the physical parameter dict: the free scalars are joined by the
    # field's latent vector (``psd_xi``), and the fixed parameters are absent.
    # Ravelling posterior.params instead gives a different, larger vector whose
    # Hessian is not the posterior precision.
    latent = context.initial_params(key, init_from=params if params is not None else posterior)

    logdensity_flat, _unravel_fn, init_flat, data_args = _get_flat_logdensity(
        context.fitter,
        latent,
    )
    # floor=0.0: the default floor of 1.0 keeps the whitening transform positive
    # definite and in doing so clips away exactly the prior-carried modes this
    # function exists to count.
    metric = negative_hessian_metric(logdensity_flat, init_flat, data_args, floor=0.0)
    gradient = -jax.grad(logdensity_flat)(init_flat, data_args)

    info = information_from_precision(
        np.asarray(metric),
        names=latent_names(latent),
        gradient=np.asarray(gradient),
    )
    if not info.at_a_mode:
        warn_measured(
            f"Expansion point is not a mode: the Newton decrement leaves "
            f"{0.5 * info.newton_decrement**2:.1f} nats of log-posterior "
            f"unclaimed (tolerance {MODE_TOLERANCE_NATS}). Curvature away from a "
            f"stationary point is not the posterior precision, so n_eff = "
            f"{info.n_eff:.2f} is not meaningful. Re-run the fit to convergence "
            f"(more restarts or more steps) or pass params= at a converged "
            f"point. Measured on one model: an under-converged MAP reported "
            f"n_eff 17.6 where the converged fit gives 6.3.",
            RuntimeWarning,
            stacklevel=2,
            newton_decrement=float(info.newton_decrement),
            unclaimed_nats=float(0.5 * info.newton_decrement**2),
            n_eff=float(info.n_eff),
        )
    return info


def latent_names(latent: dict) -> tuple[str, ...]:
    """Names for each entry of the raveled latent vector, in ravel order.

    Parameters
    ----------
    latent : dict
        The latent parameter pytree, as returned by
        ``InferenceContext.initial_params``.

    Returns
    -------
    tuple of str
        One name per scalar degree of freedom. Vector-valued entries, the
        stochastic field's ``psd_xi`` in particular, expand to indexed names
        ``psd_xi[0] ... psd_xi[n-1]``.

    Notes
    -----
    ``jax.flatten_util.ravel_pytree`` visits dict keys in sorted order, so this
    walks them the same way. Deriving the names from the pytree rather than from
    ``spec.free_params`` is deliberate: the latter counts the field as a single
    name, which is off by ``n_grid - 1`` and silently mislabels every mode.
    """
    names: list[str] = []
    for key_name in sorted(latent):
        value = np.asarray(latent[key_name])
        if value.ndim == 0:
            names.append(key_name)
        else:
            names.extend(f"{key_name}[{i}]" for i in range(value.size))
    return tuple(names)
