# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for composable precompute parity checks (#2288)."""

from __future__ import annotations

import numpy as np
import pytest

from tengri.components.agn.blocks import Recipe, composable_precompute
from tests._data_skip import requires_grahsp

pytestmark = pytest.mark.regression_bug


def _toy_filter():
    """Return a toy filter for testing."""
    wave = np.linspace(4500.0, 6500.0, 200)
    trans = np.exp(-0.5 * ((wave - 5500.0) / 300.0) ** 2)
    return wave, trans


@requires_grahsp
def test_precompute_succeeds_with_21_nodes():
    """Precompute should succeed with 21-node axis (current standard)."""
    fw, ft = _toy_filter()
    recipe = Recipe.from_selectors(
        disc="grahsp_sbpl",
        torus="grahsp",
        attenuation="none",
        axis_params=("agn_grahsp_log_l5100",),
    )

    axis_grid = np.linspace(43, 46, 21)
    out = composable_precompute.precompute(
        filter_waves=[fw],
        filter_trans=[ft],
        redshift=0.0,
        parameters=None,
        recipe=recipe,
        axis_grids={"agn_grahsp_log_l5100": axis_grid},
        wave_rest=np.logspace(2.0, 6.0, 1500, dtype=np.float64),
    )
    assert "grid_phot" in out
    assert "axes" in out
    fn = composable_precompute.build_lookup(out)
    assert fn is not None


@requires_grahsp
def test_parity_check_runs_on_precompute():
    """Test (1): Parity check runs successfully during precompute."""
    fw, ft = _toy_filter()
    recipe = Recipe.from_selectors(
        disc="grahsp_sbpl",
        torus="grahsp",
        attenuation="none",
        axis_params=("agn_grahsp_log_l5100",),
    )

    axis_grid = np.linspace(43, 46, 5)  # Small grid for testing
    out = composable_precompute.precompute(
        filter_waves=[fw],
        filter_trans=[ft],
        redshift=0.0,
        parameters=None,
        recipe=recipe,
        axis_grids={"agn_grahsp_log_l5100": axis_grid},
        wave_rest=np.logspace(2.0, 6.0, 1500, dtype=np.float64),
    )

    # The parity check should have stored the interior error.
    assert "_interior_max_rel_err" in out
    assert isinstance(out["_interior_max_rel_err"], float)
    assert out["_interior_max_rel_err"] >= 0.0


@requires_grahsp
def test_interior_accuracy_with_21_nodes():
    """Test (2): Interior accuracy measurement with 21-node axis.

    The 21-node axis (np.linspace(43, 46, 21)) is measured to have
    a maximum interior relative error of approximately 3.1e-3.
    """
    fw, ft = _toy_filter()
    recipe = Recipe.from_selectors(
        disc="grahsp_sbpl",
        torus="grahsp",
        attenuation="none",
        axis_params=("agn_grahsp_log_l5100",),
    )

    axis_grid = np.linspace(43, 46, 21)
    out = composable_precompute.precompute(
        filter_waves=[fw],
        filter_trans=[ft],
        redshift=0.0,
        parameters=None,
        recipe=recipe,
        axis_grids={"agn_grahsp_log_l5100": axis_grid},
        wave_rest=np.logspace(2.0, 6.0, 1500, dtype=np.float64),
    )

    # Check that the preint dict has the interior error stored
    assert "_interior_max_rel_err" in out
    assert isinstance(out["_interior_max_rel_err"], float)
    assert 0.0 <= out["_interior_max_rel_err"] <= 0.5
