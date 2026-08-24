# SPDX-License-Identifier: BSD-3-Clause
"""Corner plot utilities for posterior visualization and comparison.

Safe wrappers around corner-plot generation that handle degenerate posteriors
and enable overlaying multiple posteriors on a single figure.
"""

import numpy as np

from .styles import COLORS

# ═══════════════════════════════════════════════════════════════════
# Corner plot wrapper (safe against degenerate posteriors)
# ═══════════════════════════════════════════════════════════════════


def safe_corner(posterior, **kwargs):
    """Wrapper around Posterior.plot_corner that handles degenerate posteriors.

    Parameters
    ----------
    posterior : Posterior
        Fitted posterior object with a ``plot_corner`` method.
    **kwargs
        Additional keyword arguments passed to ``posterior.plot_corner``.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Corner plot figure, or None if generation failed.

    Examples
    --------
    .. code-block:: python

        from tengri import safe_corner

        fig = safe_corner(posterior, params=["sfh_dpl_alpha", "dust_tau_bc"])
        if fig is not None:
            fig.savefig("corner.pdf")
    """
    try:
        return posterior.plot_corner(**kwargs)
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"Corner plot skipped: {e}")
        return None


def plot_corner_comparison(posteriors, labels, colors=None, truths=None, params=None):
    """Overlay multiple posteriors on a single corner plot.

    Parameters
    ----------
    posteriors : list of Posterior
        Posterior objects to overlay.
    labels : list of str
        Legend labels for each posterior (e.g. ``["NUTS", "VI", "MAP"]``).
    colors : list of str, optional
        Colors for each posterior. Defaults to sampler palette.
    truths : dict, optional
        True parameter values for recovery plots.
    params : list of str, optional
        Parameter names to include. Defaults to all free parameters.

    Returns
    -------
    fig : matplotlib Figure or None
        Combined corner plot, or None if all posteriors failed.

    Examples
    --------
    .. code-block:: python

        from tengri import plot_corner_comparison

        fig = plot_corner_comparison(
            [post_nuts, post_vi],
            labels=["NUTS", "VI"],
            truths=true_params,
            params=["sfh_dpl_alpha", "dust_tau_bc"],
        )
        if fig is not None:
            fig.savefig("corner_comparison.pdf")
    """
    if colors is None:
        default_colors = [
            COLORS["rt"],
            COLORS["geovi"],
            COLORS["nuts"],
            COLORS["mgvi"],
            COLORS["map"],
        ]
        colors = default_colors[: len(posteriors)]

    fig = None
    for post, label, color in zip(posteriors, labels, colors):
        try:
            fig = post.plot_corner(params=params, truths=truths, color=color, label=label, fig=fig)
        except (ValueError, np.linalg.LinAlgError):
            print(f"Corner plot skipped for {label}")
            continue

    return fig
