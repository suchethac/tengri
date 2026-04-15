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

    Returns fig on success, None on failure.
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
    labels : list of str
    colors : list of str, optional
    truths : dict, optional
    params : list of str, optional

    Returns
    -------
    fig or None
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
