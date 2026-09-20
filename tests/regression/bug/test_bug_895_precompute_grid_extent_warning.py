# SPDX-License-Identifier: BSD-3-Clause
r"""Test build-time warning when a FREE prior exceeds precompute grid axis.

When a FREE parameter's prior bounds extend beyond the axis grid bounds, a
build-time warning alerts the user that queries outside the grid will clamp
silently to the edge (#1953), producing a flat likelihood plateau there.
This mirrors the stellar metallicity guard (_validate_metallicity_bounds, #442)
extended to AGN template-grid axes in the composable precompute.

The clamping behavior itself is tested in test_agn_composable_precompute.py;
this file asserts the warning presence and content.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tengri import Fixed, SEDModel, Uniform
from tests._data_skip import requires_grahsp

pytestmark = [pytest.mark.regression_bug, requires_grahsp]


def _minimal_observation():
    """Minimal photometry-only observation for testing."""
    from tengri.observation import Photometry

    return Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"])


def test_precompute_grid_extent_warning_free_prior_exceeds_axis(ssp_data_wne):
    """FREE prior wider than axis → exactly ONE PrecomputeGridExtentWarning."""
    # Use a recipe with axis_params (e.g., agn_log_lbol on a custom grid)
    # The prior is [43, 46] but the grid is [43.5, 45.5], so the prior exceeds it
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = SEDModel.build(
            ssp_data=ssp_data_wne,
            observation=_minimal_observation(),
            redshift=Fixed(0.0),
            agn={
                "type": "composable",
                "disc": {"type": "powerlaw"},
                "torus": {"type": "none"},
                "attenuation": {"type": "none"},
                "agn_log_lbol": Uniform(43.0, 46.0),  # FREE: [43, 46] in log_Lsun
            },
            approx={
                "type": "composable_precompute",
                "axis_grids": {
                    "agn_log_lbol": np.linspace(43.5, 45.5, 5)  # Narrower: [43.5, 45.5]
                },
            },
        )

        # Should have exactly ONE PrecomputeGridExtentWarning
        precompute_warnings = [
            warning
            for warning in w
            if hasattr(warning.category, "__name__")
            and "PrecomputeGridExtentWarning" in warning.category.__name__
        ]
        assert len(precompute_warnings) == 1
        # Check message contains parameter name and axis extent
        msg = str(precompute_warnings[0].message)
        assert "agn_log_lbol" in msg
        assert "43.5" in msg or "45.5" in msg


def test_precompute_grid_extent_free_prior_inside_axis(ssp_data_wne):
    """FREE prior inside axis bounds → zero PrecomputeGridExtentWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = SEDModel.build(
            ssp_data=ssp_data_wne,
            observation=_minimal_observation(),
            redshift=Fixed(0.0),
            agn={
                "type": "composable",
                "disc": {"type": "powerlaw"},
                "torus": {"type": "none"},
                "attenuation": {"type": "none"},
                "agn_log_lbol": Uniform(43.5, 45.5),  # FREE: [43.5, 45.5]
            },
            approx={
                "type": "composable_precompute",
                "axis_grids": {
                    "agn_log_lbol": np.linspace(43.0, 46.0, 5)  # Wider: [43, 46]
                },
            },
        )

        # Zero PrecomputeGridExtentWarning when prior is inside grid
        precompute_warnings = [
            warning
            for warning in w
            if hasattr(warning.category, "__name__")
            and "PrecomputeGridExtentWarning" in warning.category.__name__
        ]
        assert len(precompute_warnings) == 0


def test_precompute_fixed_param_beyond_axis_no_warning(ssp_data_wne):
    """FIXED param beyond axis bounds → zero PrecomputeGridExtentWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = SEDModel.build(
            ssp_data=ssp_data_wne,
            observation=_minimal_observation(),
            redshift=Fixed(0.0),
            agn={
                "type": "composable",
                "disc": {"type": "powerlaw"},
                "torus": {"type": "none"},
                "attenuation": {"type": "none"},
                "agn_log_lbol": Fixed(45.0),  # FIXED within grid [43, 46]
            },
            approx={
                "type": "composable_precompute",
                "axis_grids": {"agn_log_lbol": np.linspace(43.0, 46.0, 5)},
            },
        )

        # Fixed params should not trigger the warning (they are handled separately)
        precompute_warnings = [
            warning
            for warning in w
            if hasattr(warning.category, "__name__")
            and "PrecomputeGridExtentWarning" in warning.category.__name__
        ]
        assert len(precompute_warnings) == 0


def test_precompute_in_grid_unaffected_by_warning(ssp_data_wne):
    """In-grid queries are bit-identical whether warning runs or not."""
    # Build once with warnings suppressed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_quiet = SEDModel.build(
            ssp_data=ssp_data_wne,
            observation=_minimal_observation(),
            redshift=Fixed(0.0),
            agn={
                "type": "composable",
                "disc": {"type": "powerlaw"},
                "torus": {"type": "none"},
                "attenuation": {"type": "none"},
                "agn_log_lbol": Uniform(43.0, 46.0),
            },
            approx={
                "type": "composable_precompute",
                "axis_grids": {"agn_log_lbol": np.linspace(43.5, 45.5, 5)},
            },
        )

    # Build again with warnings active
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        model_loud = SEDModel.build(
            ssp_data=ssp_data_wne,
            observation=_minimal_observation(),
            redshift=Fixed(0.0),
            agn={
                "type": "composable",
                "disc": {"type": "powerlaw"},
                "torus": {"type": "none"},
                "attenuation": {"type": "none"},
                "agn_log_lbol": Uniform(43.0, 46.0),
            },
            approx={
                "type": "composable_precompute",
                "axis_grids": {"agn_log_lbol": np.linspace(43.5, 45.5, 5)},
            },
        )

    # Predictions on in-grid values should be identical
    params_in_grid = {"agn_log_lbol": 44.5}
    phot_quiet = model_quiet.predict_photometry(params_in_grid)
    phot_loud = model_loud.predict_photometry(params_in_grid)
    np.testing.assert_array_equal(phot_quiet, phot_loud)
