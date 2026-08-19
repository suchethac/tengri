# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #1978: continuity kept forming stars past its last bin edge.

``continuity`` assigned each age to a bin with ``searchsorted`` and then clamped
the index into ``[0, n_bins - 1]``. Every age older than the last edge therefore
landed in the oldest bin and kept its SFR, so the history ran on to the end of
the age grid (13.8 Gyr) outside any declared bin, while the mass normalization
summed bin widths only and never accounted for that tail.

Invisible with the default ladder, which already ends at 13.7 Gyr. It became
visible as soon as #1975 let a ladder be bounded to the age of the universe: a
correctly bounded model still reported most of its mass forming before the Big
Bang, because the warning integrates the clamped tail.

The low-end clamp is deliberate and stays: ``psb_suess2022`` uses edges starting
at 0.3 Gyr and relies on younger ages taking the youngest bin's rate.
"""

import numpy as np
import pytest

from tengri.components.stellar.sfh.nonparametric import (
    DEFAULT_BIN_EDGES_GYR,
    continuity,
    psb_continuity,
)

pytestmark = pytest.mark.regression_bug

_RATIOS = {f"ratio_{i}": 0.4 for i in range(6)}


class TestNoStarFormationPastTheLastEdge:
    def test_sfr_is_zero_beyond_the_last_edge(self):
        """A bounded ladder must form no stars older than its last edge."""
        edges = np.array([0.0, 0.03, 0.1, 0.3, 1.0, 2.0, 3.0, 4.28])
        age_yr = np.geomspace(1e6, 13.7e9, 400)
        sfr = np.asarray(continuity(age_yr, log_total_mass=10.0, bin_edges_gyr=edges, **_RATIOS))

        beyond = age_yr > edges[-1] * 1e9
        assert beyond.any(), "test grid must extend past the last edge"
        assert np.all(sfr[beyond] == 0.0), (
            "SFR is non-zero past the last bin edge: the oldest bin is still being "
            "extended to the end of the age grid (#1978)"
        )
        assert np.any(sfr[~beyond] > 0.0), "the bounded region must still form stars"

    def test_mass_integral_matches_the_declared_total(self):
        """With no tail, integrating the history returns the declared mass."""
        edges = np.array([0.0, 0.03, 0.1, 0.3, 1.0, 2.0, 3.0, 4.28])
        age_yr = np.linspace(0.0, 13.7e9, 200_001)
        sfr = np.asarray(continuity(age_yr, log_total_mass=10.0, bin_edges_gyr=edges, **_RATIOS))
        formed = np.trapezoid(sfr, age_yr)
        assert formed == pytest.approx(1e10, rel=1e-3), (
            f"integrated mass {formed:.4g} != declared 1e10; the clamped tail is "
            "adding mass the normalization never counted (#1978)"
        )

    def test_default_ladder_is_effectively_unchanged(self):
        """The default ends at 13.7 Gyr, so only the last sliver may change."""
        age_yr = np.geomspace(1e6, 13.6e9, 300)  # strictly inside the default ladder
        sfr = np.asarray(continuity(age_yr, log_total_mass=10.0, **_RATIOS))
        assert np.all(sfr > 0.0)
        assert np.all(np.isfinite(sfr))

    def test_youngest_bin_still_covers_ages_below_the_first_edge(self):
        """The low-end clamp is load-bearing for psb_suess2022 and must survive."""
        old_edges = DEFAULT_BIN_EDGES_GYR[2:]  # starts at 0.3 Gyr
        age_yr = np.array([1e6, 1e7, 1e8])  # all below the first edge
        sfr = np.asarray(
            psb_continuity(age_yr, log_total_mass=10.0, bin_edges_gyr=np.asarray(old_edges))
        )
        assert np.all(sfr > 0.0), (
            "ages below the first edge lost their SFR: the low-end clamp was "
            "removed along with the high-end one (#1978)"
        )
