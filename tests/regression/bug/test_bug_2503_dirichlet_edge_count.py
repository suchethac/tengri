# SPDX-License-Identifier: BSD-3-Clause
r"""Regression test for #2503: dirichlet SFH with wrong bin-edge count.

Dirichlet SFH ladders declare n internal z_frac_* parameters (where n is one
fewer than the number of bins, because the stick-breaking sums the last two
into one final bin). A dirichlet model with n declared z_frac_* parameters
requires n+2 bin edges. validate_bin_edges_gyr must reject a 7-edge ladder
for a 6-parameter dirichlet. resolve_sfh must also call validate_bin_edges_gyr
so direct calls (tests, tools) enforce the same rule as the build path does.

With correct edges (8 for a 6-parameter dirichlet), resolve_sfh returns a
param_map with exactly six z_frac_* parameters, the returned function accepts
z_frac_0..z_frac_5, and z_frac_6 raises the #2479 TypeError (unknown keyword).
"""

import numpy as np
import pytest

from tengri.components.stellar.sfh.registry import (
    resolve_sfh,
    validate_bin_edges_gyr,
)


@pytest.mark.regression_bug
class TestDirichletBinEdgeCount:
    """Test dirichlet bin-edge count validation (#2503)."""

    def test_validate_bin_edges_gyr_rejects_7_edges_for_6param_dirichlet(self):
        """validate_bin_edges_gyr raises on 7-edge ladder for 6-parameter dirichlet."""
        edges_7 = np.linspace(0.1, 13.0, 7)
        with pytest.raises(ValueError, match=r"declares 6 z_frac parameters.*needs 8 bin edges"):
            validate_bin_edges_gyr("dirichlet", edges_7)

    def test_resolve_sfh_rejects_7_edges_for_6param_dirichlet(self):
        """resolve_sfh validates bin_edges_gyr for dirichlet, rejecting 7 edges."""
        edges_7 = np.linspace(0.1, 13.0, 7)
        with pytest.raises(ValueError, match=r"declares 6 z_frac parameters.*needs 8 bin edges"):
            resolve_sfh("dirichlet", bin_edges_gyr=edges_7)

    def test_resolve_sfh_accepts_8_edges_for_6param_dirichlet(self):
        """resolve_sfh accepts 8-edge ladder for 6-parameter dirichlet."""
        edges_8 = np.linspace(0.1, 13.0, 8)
        _, _, param_map, _ = resolve_sfh("dirichlet", bin_edges_gyr=edges_8)

        # Check that exactly six z_frac_* parameters are present
        z_frac_params = [
            name for name, (internal, _, _) in param_map.items() if internal.startswith("z_frac_")
        ]
        assert len(z_frac_params) == 6, f"Expected 6 z_frac params, got {len(z_frac_params)}"
        assert set(z_frac_params) == {f"sfh_dir_z_{i}" for i in range(6)}

    def test_dirichlet_fn_accepts_z_frac_0_to_5(self):
        """Returned dirichlet function accepts z_frac_0..z_frac_5."""
        edges_8 = np.linspace(0.1, 13.0, 8)
        fn, _, _, _ = resolve_sfh("dirichlet", bin_edges_gyr=edges_8)

        # Build kwargs with valid z_frac_0..z_frac_5
        kw = {f"sfh_dir_z_{i}": 0.5 for i in range(6)}
        kw["sfh_dir_log_total_mass"] = 10.0

        # Should not raise
        t_lookback = np.array([0.0, 1.0, 5.0])
        sfr = fn(t_lookback, **kw)
        assert sfr.shape == t_lookback.shape

    def test_dirichlet_fn_rejects_z_frac_6(self):
        """dirichlet function raises TypeError on unknown z_frac_6 (#2479)."""
        from tengri.components.stellar.sfh.nonparametric import dirichlet

        edges_8 = np.linspace(0.1, 13.0, 8)

        kw = {
            f"z_frac_{i}": 0.5
            for i in range(7)  # Include z_frac_6 (one too many)
        }
        kw["log_total_mass"] = 10.0

        t_lookback = np.array([0.0, 1.0, 5.0])

        # Should raise TypeError for unknown keyword z_frac_6 when passed to dirichlet
        with pytest.raises(TypeError, match=r"z_frac_6"):
            dirichlet(t_lookback, bin_edges_gyr=edges_8, **kw)
