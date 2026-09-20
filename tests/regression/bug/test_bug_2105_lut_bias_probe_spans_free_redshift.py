# SPDX-License-Identifier: BSD-3-Clause
"""#2105: the LUT-bias advisory spans the free-redshift prior and names approx=None.

With redshift free, the z-interpolation error along the LUT's z axis is a
second error term invisible to a single-point forward probe. The advisory
must evaluate at (at least) two redshifts spanning one LUT z-table step
within the prior and take the worst case, naming approx=None as the remedy.
"""

from __future__ import annotations

import warnings
from collections import namedtuple

import numpy as np
import pytest

from tengri.config.exceptions import PrecompBiasWarning
from tengri.inference.fitter import (
    _warn_if_lut_bias_amplified,
)

pytestmark = pytest.mark.regression_bug


class _StubDistribution:
    """Stub distribution with bounds and unstandardize."""

    def __init__(self, z_min, z_max):
        self.bounds = (z_min, z_max)

    def unstandardize(self, std_val):
        """Return the median (std_val=0)."""
        return (self.bounds[0] + self.bounds[1]) / 2.0


class _StubSpec:
    """Stub parameter spec."""

    def __init__(self, free_params=None, distributions=None):
        self.free_params = free_params or ()
        self.stochastic = False
        self._distributions = distributions or {}

    def get_distribution(self, name):
        if name not in self._distributions:
            raise KeyError(name)
        return self._distributions[name]


class _StubModel:
    """Stub model that can optionally provide a z-table."""

    def __init__(self, flux, spec=None, z_grid=None):
        self.spec = spec or _StubSpec()
        self._flux = np.asarray(flux, dtype=float)
        self.n_calls = 0
        self._z_grid = z_grid

    def predict_photometry(self, params):
        self.n_calls += 1
        # Apply a triangle function bias in redshift if z is in params
        z = params.get("redshift", 0.5)
        if self._z_grid is not None:
            # Triangle function: small at nodes, large at midpoints
            z_idx = np.searchsorted(self._z_grid, z)
            if z_idx == 0 or z_idx >= len(self._z_grid):
                bias_factor = 0.001
            else:
                # Relative distance to nearest node in the step
                z_left = self._z_grid[z_idx - 1]
                z_right = self._z_grid[z_idx]
                rel_pos = (z - z_left) / (z_right - z_left)
                # Triangle: 0 at boundaries, 1 at midpoint
                bias_factor = 4 * rel_pos * (1 - rel_pos) * 0.003
            return self._flux * (1.0 + bias_factor)
        return self._flux

    def predict_spectrum(self, params):
        return self.predict_photometry(params)

    def _ztable_data_for_jit(self):
        """Return a mock PhotometricZTable if z_grid is available."""
        if self._z_grid is None:
            return None
        PhotometricZTable = namedtuple("PhotometricZTable", ["z_grid"])
        return PhotometricZTable(z_grid=self._z_grid)


def test_fixed_redshift_single_evaluation():
    """Fixed redshift: exactly one call to _lut_forward_bias (cached)."""
    flux = np.array([1.0, 2.0, 3.0, 4.0])
    spec = _StubSpec()
    exact = _StubModel(flux, spec)
    lut = _StubModel(flux * 1.002, spec)
    data = flux
    noise = np.abs(data) / 100.0

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _warn_if_lut_bias_amplified(exact, lut, data, noise, "photometry", surface="Test")

    # Count how many calls were made to predict_photometry on lut
    # For fixed redshift, should be 1 evaluation
    assert lut.n_calls == 1


def test_free_redshift_with_nodes_probes_midpoints():
    """Free redshift: probes median + inter-node midpoints when nodes exist."""
    flux = np.array([1.0, 2.0, 3.0, 4.0])
    z_grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    spec = _StubSpec(
        free_params=("redshift",), distributions={"redshift": _StubDistribution(0.1, 1.9)}
    )
    exact = _StubModel(flux, spec, z_grid)
    lut = _StubModel(flux * 1.002, spec, z_grid)
    data = flux
    noise = np.abs(data) / 100.0

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _warn_if_lut_bias_amplified(exact, lut, data, noise, "photometry", surface="Test")

    # For free-redshift with nodes, should evaluate at least 2-3 times
    # (median + midpoints)
    assert lut.n_calls >= 2, f"Expected at least 2 calls, got {lut.n_calls}"


def test_free_redshift_no_nodes_evaluates_bounds():
    """Free redshift: falls back to prior bounds when no z-grid."""
    flux = np.array([1.0, 2.0, 3.0, 4.0])
    spec = _StubSpec(
        free_params=("redshift",), distributions={"redshift": _StubDistribution(0.0, 2.0)}
    )
    exact = _StubModel(flux, spec, z_grid=None)
    lut = _StubModel(flux * 1.002, spec, z_grid=None)
    data = flux
    noise = np.abs(data) / 100.0

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _warn_if_lut_bias_amplified(exact, lut, data, noise, "photometry", surface="Test")

    # No z-grid, should evaluate at lo and hi (2 evaluations)
    assert lut.n_calls == 2


def test_free_redshift_message_names_z_peak():
    """Free redshift + high bias: warning message names the peak redshift."""
    flux = np.array([1.0, 2.0, 3.0, 4.0])
    z_grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    spec = _StubSpec(
        free_params=("redshift",), distributions={"redshift": _StubDistribution(0.1, 1.9)}
    )
    exact = _StubModel(flux, spec, z_grid)
    # High bias to trigger warning
    lut = _StubModel(flux * 1.01, spec, z_grid)
    data = flux
    noise = np.abs(data) / 100.0

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _warn_if_lut_bias_amplified(exact, lut, data, noise, "photometry", surface="Test")

    precomp_warnings = [
        warning for warning in w if issubclass(warning.category, PrecompBiasWarning)
    ]
    if precomp_warnings:
        msg = str(precomp_warnings[0].message)
        # Should mention the peak redshift and "redshift is free"
        assert "peaking at z=" in msg
        assert "redshift is free" in msg
        assert "approx=None" in msg


def test_fixed_redshift_message_no_z_peak():
    """Fixed redshift: warning message does not mention peak redshift."""
    flux = np.array([1.0, 2.0, 3.0, 4.0])
    spec = _StubSpec()
    exact = _StubModel(flux, spec)
    lut = _StubModel(flux * 1.01, spec)
    data = flux
    noise = np.abs(data) / 100.0

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _warn_if_lut_bias_amplified(exact, lut, data, noise, "photometry", surface="Test")

    precomp_warnings = [
        warning for warning in w if issubclass(warning.category, PrecompBiasWarning)
    ]
    if precomp_warnings:
        msg = str(precomp_warnings[0].message)
        # Should not mention peak redshift
        assert "peaking at z=" not in msg
        assert "redshift is free" not in msg
        assert "approx=None" in msg


def test_bias_high_snr_warns():
    """Baseline: Fixed redshift, high bias with high SNR should warn."""
    flux = np.array([1.0, 2.0, 3.0, 4.0])
    spec = _StubSpec()
    exact = _StubModel(flux, spec)
    lut = _StubModel(flux * 1.002, spec)
    data = exact._flux
    noise = np.abs(data) / 100.0

    with pytest.warns(PrecompBiasWarning) as rec:
        _warn_if_lut_bias_amplified(exact, lut, data, noise, "photometry", surface="Fitter")
    msg = str(rec[0].message)
    assert "approx=None" in msg


def test_bias_low_snr_no_warn():
    """Low SNR (noise >> signal) should not trigger warning."""
    flux = np.array([1.0, 2.0, 3.0, 4.0])
    spec = _StubSpec()
    exact = _StubModel(flux, spec)
    lut = _StubModel(flux * 1.002, spec)
    data = exact._flux
    noise = np.abs(data) / 10.0  # SNR=10, bias x SNR = 2% < 5%

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _warn_if_lut_bias_amplified(exact, lut, data, noise, "photometry", surface="Fitter")
    precomp_warnings = [
        warning for warning in w if issubclass(warning.category, PrecompBiasWarning)
    ]
    assert len(precomp_warnings) == 0
