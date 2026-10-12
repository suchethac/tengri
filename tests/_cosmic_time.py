# SPDX-License-Identifier: BSD-3-Clause
"""Assertions for the published SFH grid, which is bounded by cosmic time (#2640)."""

from __future__ import annotations

import numpy as np


def assert_grid_ends_at_age_of_universe(t_yr, sfr, z):
    """The published history is bounded by cosmic time with a node AT age(z).

    Nodes older than the universe collapse onto age(z) (zero-width cells carrying
    the history's value there), so no node lies beyond age(z) and the trapezoid
    area over cells starting at age(z) is exactly zero. The node sits at tengri's
    own age(z), which agrees with the independent cosmology to ~1e-7, so the
    bound is checked against tengri's value to rounding.
    """
    from tengri.cosmology import age_at_z

    t_yr = np.asarray(t_yr, dtype=float)
    sfr = np.asarray(sfr, dtype=float)
    age_yr = float(age_at_z(z)) * 1e9
    assert t_yr.max() <= age_yr * (1.0 + 1e-12), "a node lies beyond age(z)"
    assert t_yr.max() >= age_yr * (1.0 - 1e-12), "the grid does not reach age(z)"
    assert np.all(np.diff(t_yr) >= 0.0)
    starts_at_age = t_yr[:-1] >= age_yr * (1.0 - 1e-12)
    assert starts_at_age.any(), "no collapsed node at age(z)"
    area = 0.5 * (sfr[1:] + sfr[:-1]) * np.diff(t_yr)
    assert float(np.sum(area[starts_at_age])) == 0.0, "area beyond age(z)"
    return age_yr
