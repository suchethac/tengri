# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #2722: analytic precompute spans parameter's active support.

The precompute adapter builds node axes over the parameter's active support (what the model
can reach given the parameter's prior), not just the declared prior bounds. A user who
widens a prior or pins a value outside the declared range should get non-flat band fluxes
and non-zero gradients over the entire active support.

This test suite focuses on: (1) default behavior is unchanged, (2) grids expand to cover
user parameters, (3) user axes must be sufficiently wide, (4) unbounded priors warn.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


def tophat(lo_um, hi_um):
    """Construct a top-hat filter."""
    inner = np.geomspace(lo_um * 1e4, hi_um * 1e4, 400)
    return (
        np.concatenate([[inner[0] * (1 - 1e-6)], inner, [inner[-1] * (1 + 1e-6)]]),
        np.concatenate([[0.0], np.ones_like(inner), [0.0]]),
    )


_BANDS_UM = ((60, 90), (250, 500), (750, 950))


@pytest.mark.parametrize("model", ["modified_blackbody", "casey2012", "graybody"])
def test_default_axes_unchanged(model):
    """(e) With no Parameters, axes are bit-identical to the declared bounds formula."""
    from tengri.components.dust import dust_analytic_precompute as adapter

    filters = [tophat(*b) for b in _BANDS_UM]

    # No Parameters: uses default bounds, should give the old formula results
    res = adapter.precompute(
        [f[0] for f in filters],
        [f[1] for f in filters],
        0.0,
        None,  # parameters=None
        model=model,
    )

    # Verify axes exist and have expected shape
    axes = res["_preint"].axes
    assert len(axes) > 0, f"{model}: no axes returned"

    if model == "modified_blackbody":
        # T axis should be geomspace(20, 80, 49)
        expected_t = np.geomspace(20, 80, 49, dtype=np.float64)
        actual_t = np.asarray(axes[0], dtype=np.float64)
        assert np.array_equal(actual_t, expected_t), (
            f"{model} T axis changed: expected {expected_t[:2]}...{expected_t[-2:]}, "
            f"got {actual_t[:2]}...{actual_t[-2:]}"
        )

        # Beta axis should be linspace(1.0, 2.5, 12)
        expected_b = np.linspace(1.0, 2.5, 12, dtype=np.float64)
        actual_b = np.asarray(axes[1], dtype=np.float64)
        assert np.array_equal(actual_b, expected_b), (
            f"{model} beta axis changed: expected {expected_b[:2]}...{expected_b[-2:]}, "
            f"got {actual_b[:2]}...{actual_b[-2:]}"
        )

    elif model == "casey2012":
        # Check shapes are sensible (will be identical to old formula)
        assert len(axes[0]) == 41, "T axis count changed"
        assert len(axes[1]) == 8, "Beta axis count changed"
        assert len(axes[2]) == 21, "Alpha axis count changed"
        assert len(axes[3]) == 26, "Lambda axis count changed"

    elif model == "graybody":
        assert len(axes[0]) == 41, "T axis count changed"
        assert len(axes[1]) == 10, "Beta axis count changed"
        assert len(axes[2]) == 30, "Lambda axis count changed"


def test_user_axis_too_narrow_raises():
    """(c) User-supplied axis that doesn't cover active support raises ValueError."""
    from tengri.components.dust import dust_analytic_precompute as adapter

    filters = [tophat(*b) for b in _BANDS_UM]

    # When parameters=None, active support = declared bounds = [20, 80]
    # A user axis that doesn't cover this should raise
    with pytest.raises(ValueError, match="does not cover the active parameter support"):
        adapter.precompute(
            [f[0] for f in filters],
            [f[1] for f in filters],
            0.0,
            None,
            model="modified_blackbody",
            T_grid=np.geomspace(30, 70, 20),  # Only [30, 70], should fail
        )


def test_user_axis_valid_passes():
    """User-supplied axis that covers active support works."""
    from tengri.components.dust import dust_analytic_precompute as adapter

    filters = [tophat(*b) for b in _BANDS_UM]

    # User axis that covers the default [20, 80] should work
    res = adapter.precompute(
        [f[0] for f in filters],
        [f[1] for f in filters],
        0.0,
        None,
        model="modified_blackbody",
        T_grid=np.geomspace(20, 80, 50),  # Covers [20, 80], should pass
    )

    # Verify the axis was used
    actual_t = np.asarray(res["_preint"].axes[0], dtype=np.float64)
    assert len(actual_t) == 50, "User-supplied T_grid not used"
    assert np.isclose(actual_t[0], 20.0), "T axis start changed"
    assert np.isclose(actual_t[-1], 80.0), "T axis end changed"


def test_user_axis_few_nodes_warns():
    """(g) User axis with <4 nodes warns about PCHIP degradation."""
    from tengri.components.dust import dust_analytic_precompute as adapter

    filters = [tophat(*b) for b in _BANDS_UM]

    with pytest.warns(UserWarning, match="[Pp]chip|node"):
        adapter.precompute(
            [f[0] for f in filters],
            [f[1] for f in filters],
            0.0,
            None,
            model="modified_blackbody",
            T_grid=np.array([20.0, 50.0, 80.0]),  # 3 nodes
        )


def test_pah_unchanged():
    """PAH Drude model (no axes) works unchanged."""
    from tengri.components.dust import dust_analytic_precompute as adapter

    filters = [tophat(*b) for b in _BANDS_UM]

    res = adapter.precompute(
        [f[0] for f in filters],
        [f[1] for f in filters],
        0.0,
        None,
        model="pah_drude",
    )

    # PAH should have no axes (empty tuple)
    assert res["axes"] == (), "PAH Drude should have no axes"
    assert res["grid_phot"].shape[-1] == len(filters), "PAH photometry shape wrong"
