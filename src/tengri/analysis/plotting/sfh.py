# SPDX-License-Identifier: BSD-3-Clause
"""Star formation history plotting utilities.

Publication-quality SFH visualization for posterior recovery and model comparison.
Inspired by BAGPIPES (Carnall+2018) and Prospector (Johnson+2021).
"""

import matplotlib.pyplot as plt
import numpy as np

from .styles import COLORS, SAMPLER_STYLE

# ═══════════════════════════════════════════════════════════════════
# SFH plotting (BAGPIPES-inspired)
# ═══════════════════════════════════════════════════════════════════


def plot_sfh(
    model,
    posterior,
    true_params=None,
    ax=None,
    color=None,
    label="Posterior",
    method="RT",
    show_draws=True,
    n_draws=30,
    ci_levels=(16, 84),
    xlim=(0, 13.5),
    show_mean_sfh=True,
    xscale="linear",
):
    """Plot SFH posterior with uncertainty band: BAGPIPES/Prospector style.

    Features:

    - Filled 68% credible interval (not sample spaghetti)
    - Optional faint sample draws underneath
    - Truth as solid black line
    - Mean SFH backbone as dashed line
    - Clean axis labels with proper units

    Parameters
    ----------
    model: SEDModel
        Fitted model instance.
    posterior: Posterior
        Posterior from :meth:`Fitter.run`.
    true_params: dict, optional
        Ground truth parameters for mock recovery plots.
    ax: matplotlib Axes, optional
        Axes to plot on. Creates new figure if None.
    color: str, optional
        Line color. Defaults to sampler-specific color from style sheet.
    label: str
        Legend label. Default ``"Posterior"``.
    method: str
        Sampler name for auto-coloring (``"RT"``, ``"geoVI"``, ``"NUTS"``, ``"MAP"``).
    show_draws: bool
        If True, draw faint individual SFH samples. Default True.
    n_draws: int
        Number of sample draws to plot. Default 30.
    ci_levels: tuple of float
        Percentile levels for the filled credible interval. Default ``(16, 84)``.
    xlim: tuple of float
        x-axis limits in lookback time. [Gyr]
    show_mean_sfh: bool
        If True, overplot the smooth mean SFH backbone as a dashed line. Default True.
    xscale: {"linear", "log"}
        Lookback-time axis scale. Default ``"linear"`` (BAGPIPES convention:
        present at right). ``"log"`` resolves the recent, bursty SFH that a
        linear axis compresses into a sliver near the present; the lower limit
        is clamped off zero to ``max(xlim[0], 1e-3)`` Gyr (~1 Myr, the youngest
        SFH grid point) and the axis runs ascending (recent at left).

    Returns
    -------
    ax: matplotlib Axes

    Raises
    ------
    ValueError
        If ``xscale`` is not ``"linear"`` or ``"log"``.

    Examples
    --------
    .. code-block:: python

        from tengri import plot_sfh

        ax = plot_sfh(model, posterior, true_params=true_params, method="NUTS")
        ax.figure.savefig("sfh_recovery.pdf")
    """
    if xscale not in ("linear", "log"):
        raise ValueError(f"xscale must be 'linear' or 'log', got {xscale!r}")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    if color is None:
        color = SAMPLER_STYLE.get(method, {}).get("color", COLORS["rt"])

    if posterior.samples is not None:
        # Compute SFH draws
        n_total = len(next(iter(posterior.samples.values())))
        sfh_draws = []
        for i in range(n_total):
            s_i = {k: posterior.samples[k][i] for k in posterior.samples}
            sfh_i = model.predict_sfh(s_i)
            key = "sfr_full" if model.spec.stochastic else "sfr_mean"
            sfh_draws.append(sfh_i[key])

        sfh_arr = np.array(sfh_draws)
        t_gyr = np.array(model.predict_sfh(posterior.params)["t_gyr"])

        # Credible interval fill
        lo = np.percentile(sfh_arr, ci_levels[0], axis=0)
        hi = np.percentile(sfh_arr, ci_levels[1], axis=0)
        median = np.median(sfh_arr, axis=0)

        ax.fill_between(
            t_gyr, lo, hi, color=color, alpha=0.25, edgecolor="none", label=f"{label} (68% CI)"
        )

        # Faint sample draws (Prospector style)
        if show_draws:
            idx = np.linspace(0, n_total - 1, min(n_draws, n_total), dtype=int)
            for j in idx:
                ax.plot(t_gyr, sfh_arr[j], color=color, alpha=0.06, lw=0.5)

        # Median line
        ax.plot(t_gyr, median, color=color, lw=2.0, zorder=4)

    else:
        # MAP: point estimate
        sfh = model.predict_sfh(posterior.params)
        t_gyr = np.array(sfh["t_gyr"])
        key = "sfr_full" if model.spec.stochastic else "sfr_mean"
        ax.plot(t_gyr, sfh[key], color=COLORS["map"], lw=2.0, ls="--", label="MAP", zorder=3)

    # Truth overlay (BAGPIPES convention: solid black)
    if true_params is not None:
        sfh_true = model.predict_sfh(true_params)
        key = "sfr_full" if model.spec.stochastic else "sfr_mean"
        ax.plot(
            sfh_true["t_gyr"],
            sfh_true[key],
            color=COLORS["truth"],
            lw=2.0,
            label="Truth",
            zorder=10,
        )
        # Show mean SFH backbone for stochastic models
        if model.spec.stochastic and show_mean_sfh:
            ax.plot(
                sfh_true["t_gyr"],
                sfh_true["sfr_mean"],
                color=COLORS["truth"],
                lw=1.0,
                ls=":",
                alpha=0.4,
            )

    ax.set_xlabel(r"$\mathrm{Lookback\ time\ /\ Gyr}$")
    ax.set_ylabel(r"$\mathrm{SFR\ /\ M_\odot\ yr^{-1}}$")
    if xscale == "log":
        # Log lookback time gives the recent, bursty SFH the resolution a linear
        # axis denies it. Clamp off zero (youngest SFH grid point ~1 Myr) and run
        # ascending: recent at left, oldest at right.
        ax.set_xscale("log")
        ax.set_xlim(max(xlim[0], 1e-3), xlim[1])
    else:
        # BAGPIPES convention: lookback time with present at right.
        ax.set_xlim(xlim[1], xlim[0])  # reversed: high lookback at left, present at right
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper left")

    return ax


def add_sfh_inset(ax, t_gyr, sfr, inset_range_myr=200, **kwargs):
    """Add a zoom inset showing recent SFH (last 200 Myr by default).

    Parameters
    ----------
    ax: matplotlib Axes
        Parent axes to attach the inset to.
    t_gyr: array_like
        Lookback time in Gyr. [Gyr]
    sfr: array_like
        Star formation rate. [Msun/yr]
    inset_range_myr: float, optional
        Maximum lookback time to show in the inset. [Myr] Default 200 Myr.
    **kwargs
        Passed to ax_in.plot() (e.g., ``color``, ``lw``).

    Returns
    -------
    ax_in: matplotlib Axes
        The inset axes (for further customization).

    Examples
    --------
    .. code-block:: python

        from tengri import add_sfh_inset

        ax_in = add_sfh_inset(ax, t_gyr, sfr_median, color="blue", lw=1.5)
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    ax_in = inset_axes(ax, width="35%", height="40%", loc="upper right", borderpad=1.5)
    t_myr = np.asarray(t_gyr) * 1e3
    mask = t_myr <= inset_range_myr
    if mask.sum() > 2:
        ax_in.plot(t_myr[mask], np.asarray(sfr)[mask], **kwargs)
    ax_in.set_xlim(0, inset_range_myr)
    ax_in.set_xlabel("Lookback (Myr)", fontsize=6)
    ax_in.set_ylabel("SFR", fontsize=6)
    ax_in.tick_params(labelsize=5)
    return ax_in


def plot_sfh_comparison(model, results, true_params=None, methods=None, figsize=(15, 4)):
    """Side-by-side SFH recovery for multiple methods.

    BAGPIPES-style multi-panel layout.

    Parameters
    ----------
    model: SEDModel
        Fitted model instance.
    results: dict
        Mapping from method name to :class:`Posterior` (e.g. ``{"NUTS": post, "VI": post2}``).
    true_params: dict, optional
        Ground truth parameters for mock recovery plots.
    methods: list of str, optional
        Panel order. Defaults to ``list(results.keys())``.
    figsize: tuple of float
        Figure size in inches. Default ``(15, 4)``.

    Returns
    -------
    fig: matplotlib Figure
    axes: list of matplotlib Axes

    Examples
    --------
    .. code-block:: python

        from tengri import plot_sfh_comparison

        results = {"NUTS": posterior_nuts, "VI": posterior_vi}
        fig, axes = plot_sfh_comparison(model, results, true_params=true_params)
        fig.savefig("sfh_comparison.pdf")
    """
    if methods is None:
        methods = list(results.keys())
    n = len(methods)

    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        plot_sfh(
            model, results[method], true_params=true_params, ax=ax, method=method, label=method
        )
        ax.set_title(method, fontsize=12, fontweight="bold")
        if ax != axes[0]:
            ax.set_ylabel("")

    fig.tight_layout()
    return fig, axes
