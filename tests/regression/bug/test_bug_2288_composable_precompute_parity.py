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

    axis_grid = np.linspace(43, 46, 21)  # Standard grid for testing
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
def test_corrupted_node_raises_on_parity_check():
    """Test (1): RED - Parity check would catch corrupted grid nodes.

    First, run without corruption and verify it passes (GREEN).
    Then corrupt one node and verify the check would catch it (RED).
    Uses 21-node axis like the standard test.
    """
    fw, ft = _toy_filter()
    recipe = Recipe.from_selectors(
        disc="grahsp_sbpl",
        torus="grahsp",
        attenuation="none",
        axis_params=("agn_grahsp_log_l5100",),
    )

    axis_grid = np.linspace(43, 46, 21)
    # First: verify precompute succeeds without corruption
    out = composable_precompute.precompute(
        filter_waves=[fw],
        filter_trans=[ft],
        redshift=0.0,
        parameters=None,
        recipe=recipe,
        axis_grids={"agn_grahsp_log_l5100": axis_grid},
        wave_rest=np.logspace(2.0, 6.0, 1500, dtype=np.float64),
    )
    assert "_interior_max_rel_err" in out  # GREEN: parity check ran and stored error

    # Second: corrupt the grid significantly and verify parity check catches it (RED)
    import copy

    corrupted_out = copy.deepcopy(out)
    corrupted_grid = np.array(out["grid_phot"])
    # Multiply center node by 5 to definitely exceed tolerance
    corrupted_grid[10, 0] *= 5.0
    corrupted_out["grid_phot"] = corrupted_grid

    # Manually calling _check_lookup_parity with corrupted grid should raise RuntimeError
    with pytest.raises(RuntimeError, match="Parity check failed"):
        # Access private function via the module
        from tengri.components.agn.blocks.composable_precompute import _check_lookup_parity

        _check_lookup_parity(
            corrupted_out,
            recipe,
            np.logspace(2.0, 6.0, 1500, dtype=np.float64),
            None,
            11.42,
            axis_grids={"agn_grahsp_log_l5100": axis_grid},
            filters=list(zip([fw], [ft])),
        )


@requires_grahsp
def test_interior_accuracy_with_21_nodes():
    """Test (2): Interior accuracy measurement with 21-node axis.

    The 21-node axis (np.linspace(43, 46, 21)) is measured to have
    a maximum interior relative error of approximately 1.5e-3.
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
    # Must be > 0 (kills the placeholder)
    assert out["_interior_max_rel_err"] > 0.0
    # Must be within tolerance
    assert 0.0 <= out["_interior_max_rel_err"] <= 0.5

    # Independent measurement: pick one midpoint, evaluate exact, compare with lookup
    axis_grid_np = np.asarray(axis_grid, dtype=np.float64)
    # Pick midpoint between first two nodes
    test_point = float((axis_grid_np[0] + axis_grid_np[1]) / 2.0)

    # Exact evaluation
    exact_spectra = composable_precompute._evaluate_recipe_on_grid(
        np.logspace(2.0, 6.0, 1500, dtype=np.float64),
        recipe,
        {"agn_grahsp_log_l5100": np.array([test_point])},
        None,
        11.42,
    )
    exact_spec = exact_spectra[0]
    # Integrate through filter
    c_aa_per_s = 2.99792458e18
    wave_rest = np.logspace(2.0, 6.0, 1500, dtype=np.float64)
    nu = c_aa_per_s / wave_rest
    order = np.argsort(nu)
    trans_interp = np.interp(wave_rest, fw, ft, left=0.0, right=0.0)
    exact_photo = np.trapezoid((exact_spec * trans_interp / nu)[order], nu[order]) / np.trapezoid(
        (trans_interp / nu)[order], nu[order]
    )

    # Lookup evaluation
    fn = composable_precompute.build_lookup(out)
    lut_photo = float(fn(1.0, test_point)[0])

    # Measure relative error for this point
    floor = float(np.finfo(np.float64).tiny)
    this_rel_err = abs(lut_photo - exact_photo) / max(abs(exact_photo), floor)

    # This point's error should be less than or equal to the max measured
    assert this_rel_err <= out["_interior_max_rel_err"] * 1.1  # Small tolerance for rounding
