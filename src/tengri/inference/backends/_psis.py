# SPDX-License-Identifier: BSD-3-Clause
r"""Pareto-smoothed importance sampling (PSIS) utilities.

Importance sampling reweights draws from a proposal distribution by the
ratios of target to proposal density. When the upper tail of those ratios
is heavy, a handful of draws dominate the estimate. PSIS [1]_ stabilizes
the weights by fitting a generalized Pareto distribution (GPD) to the
upper tail of the log ratios and replacing the largest raw weights with
the expected order statistics of that fit. The fitted tail shape
``k-hat`` is also a diagnostic: when ``k-hat`` exceeds
:func:`pareto_k_threshold` the importance sampling estimate is unreliable.

Let :math:`w_i` be the raw importance ratios, shifted so that
:math:`\max_i w_i = 1`, let :math:`M` be the tail length, and let
:math:`w_c` be the largest ratio outside the tail (the cutoff). The GPD
quantile function with shape :math:`k` and scale :math:`\sigma > 0` is

.. math::

    Q(p; k, \sigma) = \frac{\sigma}{k} \left[ (1 - p)^{-k} - 1 \right],
    \qquad 0 < p < 1,

with the limit :math:`\sigma (-\log(1 - p))` as :math:`k \to 0`. The
exceedances :math:`w_i - w_c` of the tail are fitted by
:func:`fit_generalized_pareto`, and the :math:`M` tail weights are replaced,
in order, by

.. math::

    \tilde{w}_{(i)} = \min\left( w_c + Q\left(\frac{i - 1/2}{M}; k, \sigma
    \right), 1 \right), \qquad i = 1, \ldots, M,

where :math:`w_{(i)}` are the tail weights sorted in ascending order. The
truncation at 1 keeps every smoothed weight at or below the largest raw
weight. The smoothed weights are then normalized to sum to one.

The implementation uses NumPy and SciPy only. These are NumPy
post-processing utilities, not JIT-traceable: do not call them inside
``jax.jit`` or ``jax.vmap``-traced code.

References
----------
.. [1] Vehtari, A., Simpson, D., Gelman, A., Yao, Y., & Gabry, J. (2024).
   Pareto Smoothed Importance Sampling. Journal of Machine Learning
   Research, 25(72), 1-58. arXiv:1507.02646.
.. [2] Zhang, J., & Stephens, M. A. (2009). A New and Efficient Estimation
   Method for the Generalized Pareto Distribution. Technometrics, 51(3),
   316-325. doi:10.1198/tech.2009.08017.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

_MIN_SAMPLES = 10
_MIN_TAIL_LENGTH = 5
_TAIL_FRACTION = 0.2
_TAIL_SCALE = 3.0
_K_UPPER_LIMIT = 0.7
_K_ZERO_TOL = 1e-12
_ZS_PRIOR_SCALE = 3.0
_K_PRIOR_MEAN = 0.5
_K_PRIOR_PSEUDO_COUNT = 10.0


@dataclass(frozen=True)
class PSISResult:
    """Output of :func:`psis`.

    Attributes
    ----------
    log_weights : numpy.ndarray
        Smoothed, normalized log weights, shape ``(n,)``. ``logsumexp`` of
        this array is zero. Entries equal to ``-inf`` have zero weight.
    pareto_k : float
        Estimated tail shape :math:`\\hat{k}`. ``nan`` when the tail was too
        short or degenerate to fit, in which case no smoothing was applied.
    ess : float
        Effective sample size ``1 / sum(w**2)`` of the normalized smoothed
        weights.
    log_mean_raw_weight : float
        ``logsumexp(log_ratios) - log(n)``. When the ratios are
        target over proposal densities, this is the importance sampling
        estimate of the log normalizing constant of the target.
    """

    log_weights: np.ndarray
    pareto_k: float
    ess: float
    log_mean_raw_weight: float


def pareto_k_threshold(n: int) -> float:
    """Return the Pareto-:math:`k` reliability threshold for ``n`` draws.

    Parameters
    ----------
    n : int
        Number of draws, at least 10.

    Returns
    -------
    float
        ``min(1 - 1 / log10(n), 0.7)``. A :math:`\\hat{k}` above this value
        means the importance sampling estimate is unreliable
        (Vehtari et al. 2024, Section 3).

    Raises
    ------
    ValueError
        If ``n`` is smaller than 10.
    """
    if n < _MIN_SAMPLES:
        raise ValueError(f"n must be at least {_MIN_SAMPLES}; got {n}")
    return min(1.0 - 1.0 / math.log10(n), _K_UPPER_LIMIT)


def gpd_quantile(p: np.ndarray, k: float, sigma: float) -> np.ndarray:
    """Quantile function of a generalized Pareto distribution.

    Parameters
    ----------
    p : numpy.ndarray
        Probabilities in ``(0, 1)``, shape ``(m,)``.
    k : float
        Shape parameter. Positive values give a heavy upper tail.
    sigma : float
        Scale parameter, positive.

    Returns
    -------
    numpy.ndarray
        Quantiles of shape ``(m,)``, computed as
        ``sigma * expm1(-k * log1p(-p)) / k``, with the ``k -> 0`` limit
        ``-sigma * log1p(-p)`` used when ``|k| < 1e-12``.
    """
    p = np.asarray(p, dtype=float)
    if abs(k) < _K_ZERO_TOL:
        return -sigma * np.log1p(-p)
    return sigma * np.expm1(-k * np.log1p(-p)) / k


def fit_generalized_pareto(x: np.ndarray) -> tuple[float, float]:
    """Fit a generalized Pareto distribution to positive exceedances.

    Implements the profile-likelihood estimator of Zhang and Stephens [2]_,
    with the weakly informative prior on the shape from Vehtari et al. [1]_.

    Parameters
    ----------
    x : numpy.ndarray
        Exceedances over a threshold, finite and strictly positive, shape
        ``(n,)`` with ``n >= 2``. The candidate grid has ``m = 30 +
        int(sqrt(n))`` points, so the cost is ``O(m * n)`` in time and
        memory.

    Returns
    -------
    tuple of float
        ``(k, sigma)``: shape (positive for heavy tails) and scale. The
        shape is shrunk toward 0.5 with ``a = 10`` pseudo-observations,
        ``k = (n * k + 10 * 0.5) / (n + 10)``. The scale uses the unshrunk
        shape.

    Raises
    ------
    ValueError
        If ``x`` is not 1-D with at least two entries, or contains
        non-finite or non-positive values.
    """
    values = np.sort(np.asarray(x, dtype=float))
    if values.ndim != 1 or values.size < 2:
        raise ValueError("x must be a 1-D array with at least 2 entries")
    if not np.all(np.isfinite(values)) or values[0] <= 0.0:
        raise ValueError("exceedances must be finite and strictly positive")

    n = values.size
    m = 30 + int(math.sqrt(n))
    x_star = values[int(n / 4 + 0.5) - 1]
    i = np.arange(1, m + 1, dtype=float)
    b = 1.0 / values[-1] + (1.0 - np.sqrt(m / (i - 0.5))) / (_ZS_PRIOR_SCALE * x_star)

    # k_b[j] is the profile shape at candidate b[j]; log_profile[j] is the
    # profile log likelihood. The weights below equal
    # 1 / sum_j exp(log_profile_j - log_profile_i), computed in log space.
    k_b = np.mean(np.log1p(-b[:, None] * values[None, :]), axis=1)
    log_profile = n * (np.log(-b / k_b) - k_b - 1.0)
    weights = np.exp(log_profile - logsumexp(log_profile))
    b_post = float(np.sum(b * weights))

    k_raw = float(np.mean(np.log1p(-b_post * values)))
    sigma = -k_raw / b_post
    k = (n * k_raw + _K_PRIOR_PSEUDO_COUNT * _K_PRIOR_MEAN) / (n + _K_PRIOR_PSEUDO_COUNT)
    return k, sigma


def _validate_log_ratios(log_ratios: np.ndarray) -> np.ndarray:
    """Return ``log_ratios`` as a float array after checking its entries.

    Raises
    ------
    ValueError
        If the input is not 1-D, has fewer than 10 entries, contains NaN or
        ``+inf`` entries, or has no finite entries. ``-inf`` is allowed.
    """
    lr = np.asarray(log_ratios, dtype=float)
    if lr.ndim != 1:
        raise ValueError(f"log_ratios must be 1-D; got shape {lr.shape}")
    if lr.size < _MIN_SAMPLES:
        raise ValueError(f"log_ratios must have at least {_MIN_SAMPLES} entries; got {lr.size}")
    n_nan = int(np.count_nonzero(np.isnan(lr)))
    if n_nan:
        raise ValueError(f"log_ratios contains {n_nan} NaN entries")
    n_pos_inf = int(np.count_nonzero(np.isposinf(lr)))
    if n_pos_inf:
        raise ValueError(f"log_ratios contains {n_pos_inf} +inf entries")
    if not np.any(np.isfinite(lr)):
        raise ValueError("log_ratios has no finite entries")
    return lr


def _smooth_tail(shifted: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the tail-smoothed log weights and the fitted shape.

    ``shifted`` has its maximum at zero. When the tail cannot be fitted,
    the input is returned unchanged with a shape of ``nan``.
    """
    n = shifted.size
    tail_len = int(min(_TAIL_FRACTION * n, _TAIL_SCALE * math.sqrt(n)))
    if tail_len < _MIN_TAIL_LENGTH:
        return shifted.copy(), math.nan

    order = np.argsort(shifted, kind="stable")
    cutoff = shifted[order[-tail_len - 1]]
    if not np.isfinite(cutoff):
        return shifted.copy(), math.nan

    tail_idx = order[-tail_len:]
    exceedances = np.exp(shifted[tail_idx]) - np.exp(cutoff)
    positive = exceedances[exceedances > 0.0]
    if positive.size < _MIN_TAIL_LENGTH:
        return shifted.copy(), math.nan

    k_hat, sigma = fit_generalized_pareto(positive)
    if not (math.isfinite(k_hat) and math.isfinite(sigma) and sigma > 0.0):
        return shifted.copy(), math.nan

    probs = (np.arange(1, tail_len + 1) - 0.5) / tail_len
    quantiles = gpd_quantile(probs, k_hat, sigma)
    smoothed = np.log(np.exp(cutoff) + quantiles)
    smoothed_full = shifted.copy()
    smoothed_full[tail_idx] = np.minimum(smoothed, 0.0)
    return smoothed_full, k_hat


def psis(log_ratios: np.ndarray) -> PSISResult:
    """Apply Pareto-smoothed importance sampling to log importance ratios.

    Parameters
    ----------
    log_ratios : numpy.ndarray
        Log importance ratios (log target minus log proposal density) for
        ``n`` draws, shape ``(n,)``, with ``n >= 10``. Entries may be
        ``-inf`` (zero weight) but not NaN or ``+inf``.

    Returns
    -------
    PSISResult
        Smoothed normalized log weights, the tail shape ``pareto_k``
        (``nan`` if smoothing was skipped), the effective sample size, and
        the log of the mean raw weight.

    Raises
    ------
    ValueError
        If ``log_ratios`` is not 1-D, has fewer than 10 entries, contains
        NaN or ``+inf``, or has no finite entries.
    """
    lr = _validate_log_ratios(log_ratios)
    n = lr.size
    log_mean_raw_weight = float(logsumexp(lr) - math.log(n))

    shifted = lr - np.max(lr)
    smoothed, k_hat = _smooth_tail(shifted)
    log_weights = smoothed - logsumexp(smoothed)
    ess = float(1.0 / np.sum(np.exp(2.0 * log_weights)))
    return PSISResult(
        log_weights=log_weights,
        pareto_k=float(k_hat),
        ess=ess,
        log_mean_raw_weight=log_mean_raw_weight,
    )
