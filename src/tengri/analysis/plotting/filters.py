# SPDX-License-Identifier: BSD-3-Clause
"""Filter transmission curve visualization.

Plot, compare, and inspect photometric filter transmission curves using
data from the SVO Filter Profile Service (loaded via
:mod:`tengri.observation.filters`).
"""

from __future__ import annotations

import numpy as np

from tengri.analysis.plotting.styles import setup_style
from tengri.observation.filters import (
    compute_effective_wavelength,
    filter_info,
    load_filter,
)


def plot_filter_curves(
    names: list[str],
    *,
    ax=None,
    show_eff_wave: bool = True,
    normalize: bool = False,
    label_filters: bool = True,
    alpha: float = 0.7,
):
    """Plot one or more filter transmission curves.

    Parameters
    ----------
    names : list of str
        Short filter names from ``FILTER_REGISTRY``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  Created if ``None``.
    show_eff_wave : bool
        Mark each filter's effective wavelength with a vertical line.
    normalize : bool
        Normalize each curve to unit peak before plotting.
    label_filters : bool
        Add filter name labels to the legend.
    alpha : float
        Curve transparency.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    setup_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    cmap = plt.cm.tab20
    for idx, name in enumerate(names):
        fc = load_filter(name)
        wave = np.asarray(fc.wave)
        trans = np.asarray(fc.trans)
        if normalize and np.max(trans) > 0:
            trans = trans / np.max(trans)
        color = cmap(idx / max(len(names) - 1, 1))
        label = name if label_filters else None
        ax.plot(wave, trans, color=color, alpha=alpha, label=label, lw=1.2)
        if show_eff_wave:
            lam_eff = compute_effective_wavelength(wave, trans)
            ax.axvline(lam_eff, color=color, ls=":", alpha=0.4, lw=0.8)

    ax.set_xlabel(r"Wavelength [$\AA$]")
    ylabel = "Normalized transmission" if normalize else "Transmission"
    ax.set_ylabel(ylabel)
    if label_filters and len(names) <= 20:
        ax.legend(fontsize=7, ncol=max(1, len(names) // 8), loc="best")
    return ax


def plot_filter_coverage(
    names: list[str],
    *,
    ax=None,
    color_by_facility: bool = True,
    show_labels: bool = True,
):
    """Horizontal bar chart of wavelength coverage per filter.

    Each filter is a horizontal bar spanning its FWHM, centered on its
    effective wavelength.  Useful for visualizing spectral coverage of a
    photometric survey.

    Parameters
    ----------
    names : list of str
        Short filter names from ``FILTER_REGISTRY``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  Created if ``None``.
    color_by_facility : bool
        Color bars by facility.
    show_labels : bool
        Annotate bars with filter name.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    setup_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(12, max(3, len(names) * 0.3)))

    infos = [filter_info(n) for n in names]
    infos.sort(key=lambda d: d["lambda_eff_aa"])

    facilities = list({d["facility"] for d in infos})
    facilities.sort()
    fac_cmap = plt.cm.Set2
    fac_colors = {f: fac_cmap(i / max(len(facilities) - 1, 1)) for i, f in enumerate(facilities)}

    for y_pos, info in enumerate(infos):
        lam = info["lambda_eff_aa"]
        fw = info["fwhm_aa"]
        color = fac_colors[info["facility"]] if color_by_facility else "#1f77b4"
        ax.barh(
            y_pos,
            fw,
            left=lam - fw / 2,
            height=0.7,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
        )
        if show_labels:
            ax.text(
                lam,
                y_pos,
                info["name"],
                ha="center",
                va="center",
                fontsize=6,
                color="black",
            )

    ax.set_yticks(range(len(infos)))
    ax.set_yticklabels([d["name"] for d in infos], fontsize=7)
    ax.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_xscale("log")
    ax.set_title("Filter wavelength coverage")

    if color_by_facility and len(facilities) <= 15:
        from matplotlib.patches import Patch

        handles = [Patch(facecolor=fac_colors[f], label=f) for f in facilities if f != "Other"]
        if handles:
            ax.legend(handles=handles, fontsize=7, loc="lower right")

    ax.invert_yaxis()
    return ax


def compare_filter_sets(
    set_a: list[str],
    set_b: list[str],
    *,
    ax=None,
    labels: tuple[str, str] = ("Set A", "Set B"),
    normalize: bool = True,
):
    """Overlay two filter sets for visual comparison.

    Useful for comparing, e.g., original vs. recalibrated JWST curves,
    or CANDELS vs. JADES filter coverage.

    Parameters
    ----------
    set_a, set_b : list of str
        Short filter names for each set.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  Created if ``None``.
    labels : tuple of str
        Legend labels for each set.
    normalize : bool
        Normalize each curve to unit peak.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    setup_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    for name in set_a:
        fc = load_filter(name)
        wave, trans = np.asarray(fc.wave), np.asarray(fc.trans)
        if normalize and np.max(trans) > 0:
            trans = trans / np.max(trans)
        ax.fill_between(wave, trans, alpha=0.25, color="#1f77b4")
        ax.plot(wave, trans, color="#1f77b4", alpha=0.6, lw=0.8)

    for name in set_b:
        fc = load_filter(name)
        wave, trans = np.asarray(fc.wave), np.asarray(fc.trans)
        if normalize and np.max(trans) > 0:
            trans = trans / np.max(trans)
        ax.fill_between(wave, trans, alpha=0.25, color="#d62728")
        ax.plot(wave, trans, color="#d62728", alpha=0.6, lw=0.8)

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#1f77b4", alpha=0.4, label=labels[0]),
            Patch(facecolor="#d62728", alpha=0.4, label=labels[1]),
        ],
        fontsize=8,
    )
    ylabel = "Normalized transmission" if normalize else "Transmission"
    ax.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_ylabel(ylabel)
    return ax
