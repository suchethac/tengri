# SPDX-License-Identifier: BSD-3-Clause
"""Diagnostics module for hierarchical PSD recovery."""

import numpy as np
import pytest

pytestmark = pytest.mark.contract


def test_flat_widths_do_not_pass_the_scaling_criterion():
    """June's failure signature: 8192x more data, unchanged intervals."""
    from tengri.inference.population.diagnostics import interval_width_scaling

    n_values = np.array([50, 100, 200, 500])
    flat = np.array([1.80, 1.79, 1.81, 1.80])
    out = interval_width_scaling(flat, n_values)
    assert not out["excludes_zero_3sigma"], "a flat width must not pass"


def test_sqrt_n_widths_pass_the_scaling_criterion():
    from tengri.inference.population.diagnostics import interval_width_scaling

    n_values = np.array([50, 100, 200, 500])
    scaling = 12.0 / np.sqrt(n_values)
    out = interval_width_scaling(scaling, n_values)
    assert out["excludes_zero_3sigma"]
    assert abs(out["slope"] + 0.5) < 0.05, f"slope {out['slope']:.3f} should be -0.5"
