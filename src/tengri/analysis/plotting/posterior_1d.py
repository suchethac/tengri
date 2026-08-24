# SPDX-License-Identifier: BSD-3-Clause
"""Single-parameter posterior histograms and Chebyshev calibration plots.

Adds two BAGPIPES-parity helpers (closes #509):

- :func:`plot_1d_posterior`: histogram of one free parameter with the
  median + 16/84 percentile bars; optionally overlay the prior.
- :func:`plot_calibration`: plot the Chebyshev calibration polynomial
  :math:`C(\\lambda)` with its 16/84 band, useful for sanity-checking
  spectroscopy fits.

"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .styles import COLORS

__all__ = ["plot_1d_posterior", "plot_calibration"]


def plot_1d_posterior(
    posterior,
    param_name: str,
    *,
    ax=None,
    bins: int = 40,
    color: str | None = None,
    prior: bool = False,
    show_summary: bool = True,
):
    """Plot the marginal posterior of a single parameter.

    BAGPIPES-style: a normalized histogram with the median and 16/84
    percentiles marked, and (optionally) the prior overlaid.

    Parameters
    ----------
    posterior: Posterior
        Output of :meth:`Fitter.run`. Must expose ``.samples`` as a
        dict ``{name: array, ...}`` of MCMC/VI draws.
    param_name: str
        Free-parameter name (e.g. ``"redshift"``, ``"dust_tau_diff"``).
    ax: matplotlib.axes.Axes, optional
        Axes to draw on. Creates a new figure if ``None``.
    bins: int, optional
        Number of histogram bins. Default ``40``.
    color: str, optional
        Histogram color. Defaults to the project's primary color.
    prior: bool, optional
        If ``True`` and ``posterior`` exposes a ``.model.spec.distributions``
        registry for this parameter, sample the prior 10000× and overlay
        it as a dashed black curve. Default ``False``.
    show_summary: bool, optional
        If ``True`` (default), annotate the panel with
        ``median +Δ_upper / -Δ_lower``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 3.5))

    if posterior.samples is None or param_name not in posterior.samples:
        raise KeyError(
            f"posterior.samples does not contain {param_name!r}. "
            f"Available: {sorted((posterior.samples or {}).keys())}"
        )
    samples = np.asarray(posterior.samples[param_name])

    if color is None:
        color = COLORS.get("rt", "C0")

    ax.hist(
        samples,
        bins=bins,
        density=True,
        color=color,
        alpha=0.55,
        edgecolor=color,
        linewidth=1.0,
    )

    p16, p50, p84 = np.percentile(samples, [16, 50, 84])
    for x, ls in ((p50, "-"), (p16, "--"), (p84, "--")):
        ax.axvline(x, color=color, linestyle=ls, linewidth=1.0, alpha=0.85)

    if prior:
        try:
            dist = posterior.model.spec.get_distribution(param_name)
        except Exception:
            dist = None
        if dist is not None and hasattr(dist, "sample"):
            import jax.random as jr

            prior_draws = np.asarray(dist.sample(jr.PRNGKey(0), shape=(10_000,)))
            counts, edges = np.histogram(prior_draws, bins=bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(centers, counts, "k--", linewidth=1.0, label="prior")

    if show_summary:
        ax.text(
            0.02,
            0.97,
            f"{p50:.3g}$^{{+{p84 - p50:.2g}}}_{{-{p50 - p16:.2g}}}$",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )

    ax.set_xlabel(param_name)
    ax.set_ylabel("density")
    return ax


def plot_calibration(
    posterior,
    *,
    ax=None,
    wave_aa: np.ndarray | None = None,
    color: str | None = None,
    ci_levels: tuple[float, float] = (16, 84),
    show_median: bool = True,
):
    """Plot the Chebyshev calibration polynomial with its credible band.

    The calibration polynomial :math:`C(\\lambda)` rescales the observed
    spectrum to absorb instrumental flux-calibration drifts. When the
    posterior includes Chebyshev coefficients (``cal_c1``, ``cal_c2``, ...),
    we evaluate the polynomial for every draw and plot the median + the
    16/84 percentile band.

    Parameters
    ----------
    posterior: Posterior
        Posterior with calibration coefficient samples.
    ax: matplotlib.axes.Axes, optional
        Axes to draw on. Creates a new figure if ``None``.
    wave_aa: ndarray, optional
        Observed-frame wavelength grid [Å]. Defaults to
        ``posterior.observation.spectroscopy.wave_obs`` if available,
        otherwise raises.
    color: str, optional
        Line/band color.
    ci_levels: tuple of float, optional
        Percentiles for the filled credible band. Default ``(16, 84)``.
    show_median: bool, optional
        If ``True`` (default), overlay the median polynomial as a solid line.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 3.5))

    if posterior.samples is None:
        raise ValueError("posterior has no samples: nothing to plot.")

    coeff_names = sorted(
        k for k in posterior.samples if k.startswith("cal_") and k[4:].lstrip("c").isdigit()
    )
    if not coeff_names:
        raise KeyError(
            "No Chebyshev calibration coefficients (cal_c1, cal_c2, ...) "
            "found in posterior.samples. plot_calibration requires a fit "
            "with calibration enabled."
        )

    if wave_aa is None:
        try:
            wave_aa = np.asarray(posterior.observation.spectroscopy.wave_obs)
        except Exception as err:
            raise ValueError(
                "wave_aa not given and posterior does not carry an "
                "observation.spectroscopy.wave_obs grid."
            ) from err

    coeffs = np.stack([np.asarray(posterior.samples[n]) for n in coeff_names], axis=1)
    # Prepend the fixed constant term (c_0 = 1) to the free coefficients.
    # The Chebyshev polynomial is C(λ) = c_0*T_0(x) + c_1*T_1(x) + c_2*T_2(x) + ...
    # where c_0 = 1 is fixed (degenerate with overall normalization).
    # numpy.polynomial.chebyshev.chebval expects coefficients in this order.
    coeffs_with_const = np.concatenate([np.ones((coeffs.shape[0], 1)), coeffs], axis=1)
    # Chebyshev domain mapped to [-1, +1] over the observed wavelength range.
    lam_min, lam_max = wave_aa.min(), wave_aa.max()
    x = 2.0 * (wave_aa - lam_min) / (lam_max - lam_min) - 1.0
    poly_draws = np.polynomial.chebyshev.chebval(x, coeffs_with_const.T)  # (n_draws, n_wave)

    lo, hi = np.percentile(poly_draws, ci_levels, axis=0)
    if color is None:
        color = COLORS.get("rt", "C0")
    ax.fill_between(
        wave_aa, lo, hi, color=color, alpha=0.3, label=f"{ci_levels[1] - ci_levels[0]:.0f}% band"
    )
    if show_median:
        ax.plot(wave_aa, np.median(poly_draws, axis=0), color=color, linewidth=1.5, label="median")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel(r"observed $\lambda$ [Å]")
    ax.set_ylabel(r"calibration $C(\lambda)$")
    ax.legend(frameon=False, loc="best")
    return ax
