# SPDX-License-Identifier: BSD-3-Clause
"""Reusable Gaussian likelihood primitives.

These tiny pure-JAX helpers are the single source of truth for the
diagonal-Gaussian χ² that appears throughout
:mod:`tengri.inference.loss_functions` (16 call-sites as of 2026-05)
and the new :class:`tengri.pipeline.PhotometryLikelihood` Protocol
adapter. Both the legacy fitter and the component pipeline
delegate here so any change (e.g. adding a systematic floor) lands in
one place.

Rationale
---------
The legacy fitter inlined ``jnp.sum(((d - μ) / σ) ** 2)`` in 16
distinct branches of ``build_loglikelihood_fn``. Two consequences:

1. **Drift risk**, adding a systematic floor or robustification to
   the photometric branch would silently miss the spectroscopy
   branch.
2. **Adapter duplication**, the new ``Likelihood`` Protocol adapter
   would have had to re-write the same line, doubling the surface
   that needs to evolve together.

Extracting this module makes the math the contract: every χ²
computation in tengri goes through here.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.tree_util import Partial

# The whitening primitive lives in utils/ so that ``observation/`` can reach it
# without importing ``inference/``, the layering runs
# observation -> inference, never back (#1588). Re-exported here because every
# likelihood in this module is written in terms of it.
from tengri.utils.scale import whiten

__all__ = [
    "diag_gaussian_chi2",
    "diag_gaussian_log_prob",
    "diag_noise_operators",
    "inv_noise_std",
    "standardized_residual",
    "whiten",
]


def inv_noise_std(noise: jnp.ndarray) -> jnp.ndarray:
    r""":math:`\sqrt{N^{-1}} = 1/\sigma`, the float32-representable spelling.

    ``jnp.sqrt(1.0 / noise**2)`` is the *same number* by a route float32 cannot
    travel: at :math:`\sigma = 3\times10^{-30}` the destination
    :math:`1/\sigma = 3.3\times10^{29}` is comfortably inside the float32 range
    while the intermediate :math:`1/\sigma^2 = 1.1\times10^{59}` is not, so the
    result arrives as ``sqrt(inf) = inf`` and every geoVI/MGVI sqrt-metric
    primitive downstream returns ``inf`` or ``NaN`` (#1588).

    Parameters
    ----------
    noise : array_like
        Per-point 1-σ uncertainty. Must be > 0.

    Returns
    -------
    ndarray
        ``1 / noise``, same shape as the input.

    Notes
    -----
    **JIT-compatible**: yes, a single reciprocal, no barrier needed (there is
    no grouping for XLA to re-associate).

    The variable-noise branch has always used this spelling
    (:func:`tengri.observation.noise.compute_std_inv` returns
    :math:`\tau = 1/\sigma_{\rm eff}`); only the fixed-noise branch routed
    through the square.
    """
    return 1.0 / noise


def _apply_noise_cov_inv(noise: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Apply :math:`N^{-1}` as two divisions. ``noise`` first, for ``Partial``."""
    return whiten(whiten(x, noise), noise)


def _apply_noise_std_inv(noise: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Apply :math:`\\sqrt{N^{-1}}` as one division. ``noise`` first, for ``Partial``."""
    return whiten(x, noise)


def diag_noise_operators(noise: jnp.ndarray) -> tuple[Partial, Partial]:
    r"""Return :math:`(N^{-1},\; \sqrt{N^{-1}})` as operators, not as arrays.

    A diagonal Gaussian needs both, and *as arrays* one of them is always
    unrepresentable in float32: :math:`1/\sigma^2 \sim 10^{59}` at a real flux
    uncertainty. As *operators* both are safe, because each application is a
    division and no intermediate ever leaves the representable range.

    This is what NIFTy's :class:`nifty8.re.Gaussian` wants, it accepts
    callables for ``noise_cov_inv`` and ``noise_std_inv``, and derives whichever
    one it is not given from the other (``sqrt`` of the first, or the square of
    the second). Passing an array for either therefore reintroduces the
    overflow no matter which one is chosen; passing both as operators is the
    only spelling that avoids it.

    Parameters
    ----------
    noise : array_like
        Per-point 1-σ uncertainty. Must be > 0.

    Returns
    -------
    cov_inv : jax.tree_util.Partial
        ``x -> (x/sigma)/sigma``.
    std_inv : jax.tree_util.Partial
        ``x -> x/sigma``.

    Notes
    -----
    **JIT-compatible**: yes. ``Partial`` is a pytree, so ``noise`` stays a
    traced leaf rather than a baked constant, matching the array spelling it
    replaces, so a cached likelihood does not gain a recompile.
    """
    return Partial(_apply_noise_cov_inv, noise), Partial(_apply_noise_std_inv, noise)


def standardized_residual(
    observed: jnp.ndarray,
    predicted: jnp.ndarray,
    sigma_eff: jnp.ndarray,
) -> jnp.ndarray:
    r""":math:`(d - \mu) / \sigma`, with the grouping made binding on the compiler.

    Every χ² in tengri divides *before* squaring, because at real photometric
    scales :math:`\sigma \sim 10^{-31}` both :math:`\sigma^2 \sim 10^{-62}` and
    :math:`(d-\mu)^2` underflow float32 to zero while the ratio :math:`r` is
    O(1) and perfectly representable.

    Parameters
    ----------
    observed : array_like
        Data :math:`d`.
    predicted : array_like
        Model prediction :math:`\mu`.
    sigma_eff : array_like
        Effective 1-σ uncertainty, already combined with any floor.

    Returns
    -------
    ndarray
        The standardized residual, same shape as the inputs.

    Notes
    -----
    **JIT-compatible**: yes. Delegates to :func:`whiten`, which carries the
    ``optimization_barrier`` that makes the divide-before-square ordering
    binding on XLA rather than merely suggested.

    Costs nothing measurable: -1.6% against a 2.2% A/A noise floor on a 30-band
    χ², i.e. inside the measurement error.
    """
    return whiten(observed - predicted, sigma_eff)


def diag_gaussian_chi2(
    predicted: jnp.ndarray,
    observed: jnp.ndarray,
    sigma: jnp.ndarray,
    sigma_floor: float | jnp.ndarray = 0.0,
    presence: jnp.ndarray | None = None,
) -> jnp.ndarray:
    r"""Diagonal-Gaussian χ²: :math:`\sum_i p_i \frac{(d_i - \mu_i)^2}{\sigma_i^2 + (f \, d_i)^2}`.

    Parameters
    ----------
    predicted : array_like
        Model prediction :math:`\mu`.
    observed : array_like
        Data :math:`d`.
    sigma : array_like
        Per-point 1-σ uncertainty. Must be > 0.
    sigma_floor : float, optional
        Fractional systematic floor :math:`f` added in quadrature
        relative to ``observed``: total variance becomes
        ``sigma**2 + (sigma_floor * observed)**2``. Default ``0``
        (pure measurement noise).
    presence : array_like, optional
        Per-band presence mask (0.0 or 1.0 float). Absent bands
        (presence=0) contribute exactly zero to χ² and its gradient.
        Default ``None`` (all-ones; all bands present).

    Returns
    -------
    scalar jnp.ndarray
        χ². Multiply by ``-0.5`` for the data-term log-likelihood.

    Notes
    -----
    **JIT-compatible**: yes, pure JAX.

    **Numerical stability (#1206)**: the standardized residual
    :math:`r = (d - \mu)/\sigma_{\rm eff}` is formed *before* squaring, rather
    than evaluating :math:`(d-\mu)^2/\sigma^2` directly. Flux uncertainties are
    ~1e-30, so :math:`\sigma^2` (~1e-60) and :math:`(d-\mu)^2` (~1e-56) both
    underflow float32 to zero and the ratio becomes ``0/0 = NaN``, even though
    :math:`r` itself is O(1) and representable. :func:`jnp.hypot` combines the
    measurement noise and the fractional floor without squaring either. This is
    what lets a pure-float32 fit produce a finite likelihood; identical in
    float64 to the last bit.
    """
    sigma_eff = jnp.hypot(sigma, sigma_floor * observed)
    r = standardized_residual(observed, predicted, sigma_eff)
    chi2_summand = r * r
    if presence is not None:
        chi2_summand = presence * chi2_summand
    return jnp.sum(chi2_summand)


def diag_gaussian_log_prob(
    predicted: jnp.ndarray,
    observed: jnp.ndarray,
    sigma: jnp.ndarray,
    sigma_floor: float | jnp.ndarray = 0.0,
    presence: jnp.ndarray | None = None,
) -> jnp.ndarray:
    r"""Data-term log-probability of a diagonal Gaussian: :math:`-\tfrac12 \chi^2`.

    The Gaussian normalization constant
    :math:`-\tfrac{1}{2} n \log(2\pi) - \sum_i \log\sigma_i` is dropped
    most inference engines treat it as an additive constant. Add it
    back explicitly if you need a true log-evidence term.

    Parameters
    ----------
    presence : array_like, optional
        Per-band presence mask (0.0 or 1.0 float). Absent bands
        contribute exactly zero to the log-probability and its gradient.
        Default ``None`` (all-ones; all bands present).

    Notes
    -----
    **JIT-compatible**: yes, pure JAX.
    """
    return -0.5 * diag_gaussian_chi2(predicted, observed, sigma, sigma_floor, presence)
