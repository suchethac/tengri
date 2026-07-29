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

    Notes
    -----
    The operative half of the criterion is **excluding zero**, not matching
    -0.5. A previously published claim in this project said the shared-PSD
    credible intervals "shrink approximately as 1/sqrt(N)". Later measurements
    found they did not shrink at all — the intervals stayed roughly constant
    across an 8192-fold increase in data, which is the signature of a
    posterior dominated by its prior rather than by the likelihood. This
    function must be able to FAIL on flat widths, so a version that reports
    success on flat widths would let the same wrong claim through again.
    """
    widths = np.asarray(widths)
    n_values = np.asarray(n_values)

    log_widths = np.log(widths)
    log_n = np.log(n_values)

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


def credible_interval(log_posterior: np.ndarray, grid, level: float = 0.68) -> dict:
    """Compute credible intervals from a log-posterior on a 2D grid.

    Parameters
    ----------
    log_posterior : array_like, shape (A * B,)
        Unnormalized log-posterior [nats] on the grid.
    grid : SharedGrid
        Quadrature grid with attributes ``sigma`` (A,) and ``tau_yr`` (B,).
    level : float, optional
        Credible level (default 0.68 for 1-sigma intervals) [dimensionless].

    Returns
    -------
    intervals : dict
        Dictionary with keys ``"sigma_lower"``, ``"sigma_upper"``,
        ``"tau_lower_yr"``, ``"tau_upper_yr"``, and ``"credible_levels"``
        (the two marginal levels [dimensionless]).
    """
    log_posterior = np.asarray(log_posterior)
    grid_sigma = np.asarray(grid.sigma)
    grid_tau = np.asarray(grid.tau_yr)

    n_sigma = len(grid_sigma)
    n_tau = len(grid_tau)

    # Reshape to (A, B) and normalize to get posterior probabilities
    posterior_grid = np.exp(log_posterior.reshape(n_sigma, n_tau))
    posterior_grid = posterior_grid / np.sum(posterior_grid)

    # Marginal posteriors
    sigma_marginal = np.sum(posterior_grid, axis=1)  # (A,)
    tau_marginal = np.sum(posterior_grid, axis=0)  # (B,)

    # Cumulative distributions
    sigma_cdf = np.cumsum(sigma_marginal)
    tau_cdf = np.cumsum(tau_marginal)

    # Credible intervals (symmetric in probability)
    lower_percentile = (1.0 - level) / 2.0
    upper_percentile = 1.0 - lower_percentile

    sigma_lower_idx = np.searchsorted(sigma_cdf, lower_percentile)
    sigma_upper_idx = np.searchsorted(sigma_cdf, upper_percentile)
    tau_lower_idx = np.searchsorted(tau_cdf, lower_percentile)
    tau_upper_idx = np.searchsorted(tau_cdf, upper_percentile)

    sigma_lower = float(grid_sigma[sigma_lower_idx])
    sigma_upper = float(grid_sigma[sigma_upper_idx])
    tau_lower = float(grid_tau[tau_lower_idx])
    tau_upper = float(grid_tau[tau_upper_idx])

    return {
        "sigma_lower": sigma_lower,
        "sigma_upper": sigma_upper,
        "tau_lower_yr": tau_lower,
        "tau_upper_yr": tau_upper,
        "credible_levels": (level, level),
    }


def report(interim_result, shared_posterior) -> dict:
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
        Bundled diagnostics including ESS checks and a zero-divergence
        warning flag.

    Notes
    -----
    Warns if the population reports zero divergences across all galaxies.
    This is a red flag, not a pass: a chain that traverses hard geometry
    reports divergences, while chains frozen in separate basins have
    nothing to report.
    """
    import warnings

    _, ess_summary = shared_posterior

    # Check for zero divergences
    n_divergent = interim_result.get("n_divergent", np.array([]))
    if int(np.sum(n_divergent)) == 0:
        warnings.warn(
            "Zero divergences across the whole population. This is a red flag, not "
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
