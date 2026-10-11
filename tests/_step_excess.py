# SPDX-License-Identifier: BSD-3-Clause
"""Shared statistic for detecting a staircase in a swept SFH time parameter.

A hard mask at a support boundary that moves with a free parameter makes the
integrated photometry a step function of that parameter: a jump each time an
age-grid node crosses the moving boundary. :func:`step_excess` is the largest
single step in a fine sweep, in units of the sweep's median step -- a smooth
curve sits near 1, and a staircase reads far above it (see
``tests/physics/gradients/test_lognormal_onset_smoothness.py`` and
``tests/physics/gradients/test_sfh_time_parameter_smoothness.py``).
"""

from __future__ import annotations

import numpy as np


def step_excess(values: np.ndarray) -> float:
    """Largest single step in ``values``, in units of the median step.

    Parameters
    ----------
    values : array_like, shape (n,)
        A quantity swept finely over one parameter (e.g. summed photometry
        over an age sweep), ``n >= 2``.

    Returns
    -------
    float
        ``max(|diff|) / median(|diff|)``, relative to each step's own
        starting value. A smooth curve sits near 1; a staircase reads far
        above it. ``nan`` (via a zero median) signals the sweep did not move
        the quantity enough to judge -- the caller should treat that as
        "undetermined", not as a pass.
    """
    steps = np.abs(np.diff(values)) / np.abs(values[:-1])
    return float(steps.max() / np.median(steps))


#: Half-width, in sweep steps, of the neighborhood :func:`local_step_excess`
#: compares each step against: 10 steps is 5 % of a 201-point sweep, wide
#: enough that the median is set by the smooth trend between jumps and narrow
#: enough that the trend's own variation over the window stays small.
LOCAL_HALF_WIDTH = 10


def local_step_excess(values: np.ndarray, half_width: int = LOCAL_HALF_WIDTH) -> float:
    """Largest single step in ``values``, in units of the median step around it.

    :func:`step_excess` divides by the median over the whole sweep, so a smooth
    curve whose slope varies by more than the threshold across the window reads
    as a staircase: the periodic SFH's ``tau_bursts_gyr`` sweep is smooth (its
    finite differences match the autodiff gradient to 6e-5 at every point, and
    its largest step is 1.13 of its neighbors') but its slope at
    ``tau = 50`` Myr is 4.3 times the sweep's median slope, converged under a
    4x finer integrand. A staircase is a step large against its *neighbors*,
    which this measures; it also reads higher than :func:`step_excess` for a
    genuine jump on a curve whose slope varies.

    Parameters
    ----------
    values : array_like, shape (n,)
        A quantity swept finely over one parameter, ``n >= 2``.
    half_width : int, optional
        Neighborhood half-width in steps (default :data:`LOCAL_HALF_WIDTH`).

    Returns
    -------
    float
        ``max_i |d_i| / median(|d_{i-h..i+h}|)`` for the relative steps
        ``d_i``. A smooth curve sits near 1. A non-zero step whose neighborhood
        median is zero (a jump between exactly flat stretches) reads ``inf``.
    """
    steps = np.abs(np.diff(values)) / np.abs(values[:-1])
    n = steps.shape[0]
    med = np.array(
        [np.median(steps[max(0, i - half_width) : i + half_width + 1]) for i in range(n)]
    )
    # A step among exactly flat neighbors is a jump (inf); a flat step there is none.
    flat = np.where(steps > 0.0, np.inf, 0.0)
    ratio = np.divide(steps, med, out=flat, where=med > 0.0)
    return float(np.max(ratio))
