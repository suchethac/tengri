# SPDX-License-Identifier: BSD-3-Clause
"""Tests for Prospector-β redshift-aware age bin construction.

Tests verify monotonicity, coverage, and numpy array output type.
"""

import numpy as np
import pytest

from tengri.components.stellar.sfh.nonparametric import make_agebins_from_zred

pytestmark = pytest.mark.bounds


class TestMakeAgebinsFromZred:
    """Tests for Prospector-β redshift-aware age bin construction."""

    def test_edges_monotone(self):
        edges = make_agebins_from_zred(1.0)
        assert np.all(np.diff(edges) >= 0.0), "bin edges must be monotonically non-decreasing"

    def test_starts_at_zero(self):
        edges = make_agebins_from_zred(2.0)
        assert edges[0] == 0.0

    def test_capped_at_tuniv_z2(self):
        edges = make_agebins_from_zred(2.0)
        # Age of universe at z=2 is ~3.3 Gyr; edges must not exceed it
        assert edges[-1] <= 3.5, f"edges exceed tuniv at z=2: {edges[-1]:.2f} Gyr"

    def test_capped_at_tuniv_z4(self):
        edges = make_agebins_from_zred(4.0)
        assert edges[-1] <= 1.8, f"edges exceed tuniv at z=4: {edges[-1]:.2f} Gyr"

    def test_capped_at_tuniv_z6(self):
        edges = make_agebins_from_zred(6.0)
        assert edges[-1] <= 1.0, f"edges exceed tuniv at z=6: {edges[-1]:.2f} Gyr"

    def test_returns_numpy_not_jax(self):
        edges = make_agebins_from_zred(1.0)
        assert isinstance(edges, np.ndarray), "should return numpy array (setup-time utility)"

    def test_n_bins_argument(self):
        edges = make_agebins_from_zred(1.0, n_bins=5)
        assert len(edges) == 6, f"n_bins=5 → 6 edges, got {len(edges)}"

    def test_low_zred_has_young_bins(self):
        edges = make_agebins_from_zred(0.5)
        # Should include ~30 Myr and ~100 Myr young edges
        assert any(0.02 < e < 0.05 for e in edges), "missing ~30 Myr young bin edge"
        assert any(0.08 < e < 0.15 for e in edges), "missing ~100 Myr young bin edge"
