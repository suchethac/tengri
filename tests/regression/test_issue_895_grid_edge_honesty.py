# SPDX-License-Identifier: BSD-3-Clause
"""Tests for issue #895: grid edge honesty in triweight interpolation.

When a query point lies entirely outside a grid axis (beyond the triweight
kernel's support), compute_grid_weights falls back to nearest-bin clamping.
The clamped value is the edge of the grid with zero gradient — a silent
plateau that can trap MCMC/optimization chains (issue #895, class with #442).

Fixes:
  - Add `on_out_of_grid` parameter to choose between 'clamp' (default) and 'nan'
  - Document gradient consequence (zero gradient outside grid)
  - Build-time guard warns when free parameter prior exceeds grid bounds
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.utils.interpolation import compute_grid_weights, compute_grid_window

# Regression test for issue #895: silent clamping to nearest bin (grid edge honesty).
# Taxonomy marker per tests/TESTING.md: regression_bug = frozen output for a previously-fixed bug.
pytestmark = pytest.mark.regression_bug


class TestComputeGridWeightsOnOutOfGridParameter:
    """Test on_out_of_grid parameter behavior."""

    def test_on_out_of_grid_clamp_is_default(self):
        """Clamp mode is the default, preserving backward compatibility."""
        grid = jnp.linspace(0, 1, 5)
        w_default = compute_grid_weights(2.0, grid)  # out-of-grid
        w_clamp = compute_grid_weights(2.0, grid, on_out_of_grid="clamp")
        np.testing.assert_array_equal(w_default, w_clamp)

    def test_on_out_of_grid_clamp_returns_edge_value(self):
        """Clamp mode returns nearest-bin value with zero gradient outside grid."""
        grid = jnp.linspace(0, 1, 5)
        # Query far outside grid: should clamp to edge (last bin)
        w = compute_grid_weights(2.0, grid, on_out_of_grid="clamp")
        # Expect all weight on the last node (nearest to x=2.0)
        expected = np.zeros(5)
        expected[-1] = 1.0
        np.testing.assert_array_almost_equal(w, expected)

    def test_on_out_of_grid_nan_returns_nan(self):
        """NaN mode returns NaN when query is entirely outside grid."""
        grid = jnp.linspace(0, 1, 5)
        # Query far outside grid
        w = compute_grid_weights(2.0, grid, on_out_of_grid="nan")
        assert jnp.all(jnp.isnan(w)), "Expected all NaN values out-of-grid"

    def test_on_out_of_grid_invalid_mode_raises(self):
        """Invalid mode string raises ValueError at trace time."""
        grid = jnp.linspace(0, 1, 5)
        with pytest.raises(ValueError, match="on_out_of_grid must be"):
            compute_grid_weights(0.5, grid, on_out_of_grid="invalid")

    def test_in_grid_query_unaffected_by_mode(self):
        """In-grid behavior is bit-identical regardless of on_out_of_grid mode."""
        grid = jnp.linspace(0, 1, 5)
        x = 0.5  # in-grid query
        w_clamp = compute_grid_weights(x, grid, on_out_of_grid="clamp")
        w_nan = compute_grid_weights(x, grid, on_out_of_grid="nan")
        # Both should be identical for in-grid queries (total > 0)
        np.testing.assert_array_equal(w_clamp, w_nan)

    def test_in_grid_reference_value_bit_identical(self):
        """
        In-grid interpolation is bit-identical before/after on_out_of_grid change.

        Regression test pinning the value computed on the unmodified code.
        Reference: commit before issue #895 fix was applied.
        """
        grid = jnp.linspace(0.0, 1.0, 5)
        x = 0.5
        w = compute_grid_weights(x, grid, scatter=0.2)
        # Expected values: pre-#895 computation
        # Computed with: w = compute_grid_weights(0.5, jnp.linspace(0, 1, 5))
        expected = np.array(
            [
                0.026670952,
                0.255099544,
                0.436459040,
                0.255099544,
                0.026670952,
            ]
        )
        np.testing.assert_array_almost_equal(w, expected, decimal=6)


class TestComputeGridWindowOnOutOfGridParameter:
    """Test on_out_of_grid parameter for compute_grid_window."""

    def test_window_clamp_is_default(self):
        """Clamp mode is default for compute_grid_window."""
        grid = jnp.linspace(0, 1, 10)
        start_default, w_default = compute_grid_window(0.5, grid)
        start_clamp, w_clamp = compute_grid_window(0.5, grid, on_out_of_grid="clamp")
        np.testing.assert_array_equal(w_default, w_clamp)
        assert int(start_default) == int(start_clamp)

    def test_window_invalid_mode_raises(self):
        """Invalid mode raises ValueError."""
        grid = jnp.linspace(0, 1, 10)
        with pytest.raises(ValueError, match="on_out_of_grid must be"):
            compute_grid_window(0.5, grid, on_out_of_grid="bogus")

    def test_window_out_of_grid_nan_mode(self):
        """Window NaN mode returns NaN when query is outside grid."""
        grid = jnp.linspace(0, 1, 10)
        x = 2.0  # far outside
        _, w = compute_grid_window(x, grid, on_out_of_grid="nan")
        assert jnp.any(jnp.isnan(w)), "Expected NaN in window for out-of-grid query"


class TestSpsBackendDeprecation:
    """Test sps_backend deprecation (issue #1470)."""

    def test_sps_backend_default_no_warning(self):
        """Default sps_backend='dsps' does not emit warning."""
        from tengri.components.stellar.component import StellarSEDComponentConfig

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = StellarSEDComponentConfig()  # default sps_backend="dsps"
            # Filter to only DeprecationWarning about sps_backend
            sps_warnings = [warning for warning in w if "sps_backend" in str(warning.message)]
            assert len(sps_warnings) == 0, "No warning for default sps_backend"

    def test_sps_backend_nondefault_emits_warning(self):
        """Non-default sps_backend emits DeprecationWarning."""
        from tengri.components.stellar.component import StellarSEDComponentConfig

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = StellarSEDComponentConfig(sps_backend="custom")
            sps_warnings = [
                warning
                for warning in w
                if "sps_backend" in str(warning.message)
                and issubclass(warning.category, DeprecationWarning)
            ]
            assert len(sps_warnings) == 1, "Expected one deprecation warning"
            msg = str(sps_warnings[0].message)
            assert "deprecated" in msg.lower()
            assert "age_kernel" in msg


# These are regression tests; the taxonomy marker is applied via pytestmark
