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
