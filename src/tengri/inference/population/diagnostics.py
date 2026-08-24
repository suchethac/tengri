# SPDX-License-Identifier: BSD-3-Clause
"""Diagnostics module for hierarchical PSD recovery."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["credible_interval", "interval_width_scaling", "report"]


def interval_width_scaling(widths: np.ndarray, n_values: np.ndarray) -> dict[str, Any]:
    r"""Test whether credible interval widths scale as 1/sqrt(N).

    Regresses ``log(width)`` on ``log(N)`` using ordinary least squares.
    If widths scale as ``1/sqrt(N)``, the slope should be approximately -0.5.
    A flat width series (slope ≈ 0) fails the scaling test.

    Parameters
    ----------
    widths : array_like, shape (M,)
        Credible interval widths [dimensionless].
    n_values : array_like, shape (M,)
        Sample sizes [dimensionless].

    Returns
    -------
    scaling : dict
        Dictionary with keys:

        - ``"slope"`` : float
            Slope of ``log(width)`` vs ``log(N)`` [dimensionless].
        - ``"slope_err"`` : float
            Standard error of the slope [dimensionless].
        - ``"excludes_zero_3sigma"`` : bool
            True if ``abs(slope) > 3 * slope_err``, i.e., the slope is
            significantly different from zero at the 3-sigma level. Flat
            widths return False.

    Raises
    ------
    ValueError
        If fewer than 3 data points are provided (minimum needed to estimate
        slope_err = sqrt(mse / var), with mse dividing by n-2).
    ValueError
        If the input contains NaN values.

    Notes
    -----
    **Ordinary least squares fit.** Regresses ``log(width)`` on ``log(N)``
    to estimate whether widths scale as a power law:

    .. math::

        \log(\text{width}) = \text{slope} \cdot \log(N) + \text{intercept}

    The slope and its standard error are:

    .. math::

        \text{slope} = \frac{\sum_i (\log N_i - \overline{\log N})
                             (\log w_i - \overline{\log w})}
                            {\sum_i (\log N_i - \overline{\log N})^2}

        \text{slope\_err} = \sqrt{\frac{\text{MSE}}{\sum_i (\log N_i - \overline{\log N})^2}}

        \text{MSE} = \frac{\sum_i (\log w_i - \hat{\log w}_i)^2}{n - 2}

    where :math:`\hat{\log w}_i = \text{slope} \cdot \log N_i + \text{intercept}`
    and :math:`n` is the number of data points [dimensionless].

    **The operative half of the criterion is excluding zero**, not matching
    -0.5. A previously published claim in this project said the shared-PSD
    credible intervals "shrink approximately as 1/sqrt(N)". Later measurements
    found they did not shrink at all, the intervals stayed roughly constant
    across an 8192-fold increase in data, which is the signature of a
    posterior dominated by its prior rather than by the likelihood. This
    function must be able to FAIL on flat widths, so a version that reports
    success on flat widths would let the same wrong claim through again.
    """
    widths = np.asarray(widths)
    n_values = np.asarray(n_values)

    # Input validation
    if len(n_values) < 3:
        raise ValueError(
            f"At least 3 data points required to estimate slope_err "
            f"(divides by n-2), got {len(n_values)}"
        )

    log_widths = np.log(widths)
    log_n = np.log(n_values)

    if np.any(np.isnan(log_widths)) or np.any(np.isnan(log_n)):
        raise ValueError("NaN in input widths or n_values")

    # Ordinary least squares fit: log(width) = slope * log(N) + intercept
    # Using the normal equations: slope = cov(log_N, log_width) / var(log_N)
    n = len(n_values)
    mean_log_n = np.mean(log_n)
    mean_log_widths = np.mean(log_widths)

    # Calculate slope and residuals
    numerator = np.sum((log_n - mean_log_n) * (log_widths - mean_log_widths))
    denominator = np.sum((log_n - mean_log_n) ** 2)
    slope = numerator / denominator

    # Calculate intercept and residuals
    intercept = mean_log_widths - slope * mean_log_n
    y_pred = slope * log_n + intercept
    residuals = log_widths - y_pred

    # Standard error of the slope
    mse = np.sum(residuals**2) / (n - 2)
    slope_err = np.sqrt(mse / denominator)

    excludes_zero_3sigma = abs(slope) > 3 * slope_err

    return {
        "slope": float(slope),
        "slope_err": float(slope_err),
        "excludes_zero_3sigma": bool(excludes_zero_3sigma),
    }


def credible_interval(log_posterior: np.ndarray, grid: Any, level: float = 0.68) -> dict[str, Any]:
    """Compute credible intervals from a log-posterior on a 2D grid.

    Parameters
    ----------
    log_posterior : array_like, shape (A * B,)
        Unnormalized log-posterior [nats] on the grid (C-ordered as node
        ``a*B + b`` is ``(sigma[a], tau_yr[b])``).
    grid : SharedGrid
        Quadrature grid with attributes ``sigma`` (A,) and ``tau_yr`` (B,).
    level : float, optional
        Credible level (default 0.68 for 1-sigma intervals) [dimensionless].

    Returns
    -------
    intervals : dict
        Dictionary with keys:

        - ``"sigma_lower"`` : float
            Lower bound of sigma interval [dex].
        - ``"sigma_upper"`` : float
            Upper bound of sigma interval [dex].
        - ``"tau_lower_yr"`` : float
            Lower bound of tau interval [yr].
        - ``"tau_upper_yr"`` : float
            Upper bound of tau interval [yr].
        - ``"credible_levels"`` : tuple of float
            Pair of (level, level) for the marginal probabilities [dimensionless].
    """
    log_posterior = np.asarray(log_posterior)
    grid_sigma = np.asarray(grid.sigma)
    grid_tau = np.asarray(grid.tau_yr)

    n_sigma = len(grid_sigma)
    n_tau = len(grid_tau)

    # Reshape to (A, B) and normalize to get posterior probabilities.
    #
    # Subtract the max BEFORE exponentiating. ``log_posterior`` is unnormalized
    # and sums one term per galaxy, so its magnitude grows with N: at N = 64 it
    # is routinely in the hundreds or thousands. ``np.exp`` of that underflows
    # every node to 0.0, and the normalization then divides 0 by 0 and returns
    # NaN for the whole grid, silently, and only once the population is large
    # enough to matter.
    lp = log_posterior.reshape(n_sigma, n_tau)
    posterior_grid = np.exp(lp - np.max(lp))
    total = np.sum(posterior_grid)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(
            "Posterior grid did not normalize: sum is "
            f"{total!r}. The log-posterior may contain NaN or -inf at every "
            "node, check that the tau bounds do not reach the regime where "
            "ou_logpdf underflows, and that the interim samples are finite."
        )
    posterior_grid = posterior_grid / total

    # Marginal posteriors
    sigma_marginal = np.sum(posterior_grid, axis=1)  # (A,)
    tau_marginal = np.sum(posterior_grid, axis=0)  # (B,)

    # Cumulative distributions
    sigma_cdf = np.cumsum(sigma_marginal)
    tau_cdf = np.cumsum(tau_marginal)

    # Credible intervals (symmetric in probability)
    lower_percentile = (1.0 - level) / 2.0
    upper_percentile = 1.0 - lower_percentile

    # Interpolate the CDF rather than snapping to a node.
    #
    # ``searchsorted`` returns a grid INDEX, so the interval endpoints could only
    # ever be grid values and the WIDTH was quantized to whole cells. On a
    # 60-node sigma grid over (0.01, 1.0) the spacing is 0.01678, and two
    # different population sizes both reported a width of exactly 0.117, seven
    # cells, twice. That is fatal for ``interval_width_scaling``, whose entire
    # job is to detect how the width changes: quantization both adds a
    # +/- one-cell error and floors the width at one cell, so a genuinely
    # shrinking interval stops shrinking as soon as it reaches the resolution.
    # Snapping also biases the width upward, since searchsorted returns the
    # first index at or past the target mass.
    sigma_lower = float(np.interp(lower_percentile, sigma_cdf, grid_sigma))
    sigma_upper = float(np.interp(upper_percentile, sigma_cdf, grid_sigma))
    tau_lower = float(np.interp(lower_percentile, tau_cdf, grid_tau))
    tau_upper = float(np.interp(upper_percentile, tau_cdf, grid_tau))

    return {
        "sigma_lower": sigma_lower,
        "sigma_upper": sigma_upper,
        "tau_lower_yr": tau_lower,
        "tau_upper_yr": tau_upper,
        "credible_levels": (level, level),
    }


def report(interim_result: dict[str, Any], shared_posterior: tuple[Any, Any]) -> dict[str, Any]:
    """Bundle diagnostics from an interim result and shared posterior.

    Parameters
    ----------
    interim_result : dict
        Result from a single-galaxy interim fit with keys like
        ``"n_divergent"``, etc.
    shared_posterior : tuple
        A 2-tuple from :func:`shared_log_posterior`, where the second
        element is an ESSSummary with ``.at_mode`` and ``.min_high_mass``.

    Returns
    -------
    diagnostics : dict
        Dictionary with keys:

        - ``"ess_at_mode"`` : ndarray, shape (N,)
            ESS at the posterior mode [dimensionless]; primary diagnostic.
        - ``"ess_min_high_mass"`` : ndarray, shape (N,)
            Minimum ESS over nodes carrying top 99% of posterior mass
            [dimensionless]; use to detect tail degeneracy.
        - ``"zero_divergence_flag"`` : bool
            True if the population reported zero divergences across all
            galaxies (a red flag, not a pass).

    Notes
    -----
    Issues a `UserWarning` if the population reports zero divergences across
    all galaxies. This is a red flag, not a pass: a chain that traverses hard
    geometry reports divergences, while chains frozen in separate basins have
    nothing to report.
    """
    import warnings

    _, ess_summary = shared_posterior

    # Check for zero divergences
    n_divergent = interim_result.get("n_divergent", np.array([]))
    if int(np.sum(n_divergent)) == 0:
        warnings.warn(
            "Zero divergences across the whole population. Treat as a red flag only "
            "for NUTS, where energy error is checked at every tree doubling and bad "
            "geometry diverges readily. For STATIC HMC the blackjax default "
            "divergence_threshold is 1000, so zero means the integrator never "
            "exploded, NOT that the chain mixed. There, judge on R-hat and ESS. "
            "Where it does apply, it is a red flag, not "
            "a clean bill of health: a chain that traverses hard geometry reports "
            "it, while chains frozen in separate basins have nothing to report. "
            "Check R-hat including psd_xi before trusting these intervals.",
            UserWarning,
            stacklevel=2,
        )

    return {
        "ess_at_mode": ess_summary.at_mode,
        "ess_min_high_mass": ess_summary.min_high_mass,
        "zero_divergence_flag": int(np.sum(n_divergent)) == 0,
    }
