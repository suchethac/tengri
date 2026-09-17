# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for issue #901: tbabs is actually wabs (Morrison & McCammon 1983).

The function tbabs_transmission implements the Morrison & McCammon (1983) wabs
cross-section, not the Wilms+2000 tbabs cross-section. This test ensures:

1. wabs_transmission is importable and bit-identical to tbabs_transmission
2. tbabs_transmission as a deprecated alias emits a DeprecationWarning
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest


def test_wabs_transmission_importable():
    """wabs_transmission can be imported from xray module."""
    from tengri.components.xray.xray import wabs_transmission

    assert callable(wabs_transmission)


def test_wabs_transmission_bit_identical_to_tbabs():
    """wabs_transmission is bit-identical to tbabs_transmission on test grid.

    Tests bit-identity using np.array_equal (not allclose) to ensure
    they take the same code path.
    """
    from tengri.components.xray.xray import wabs_transmission, tbabs_transmission

    # Suppress deprecation warning for this test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        E_grid = jnp.array([0.5, 1.0, 2.0, 5.0])

        # Test at log_nh = 21
        result_wabs_21 = wabs_transmission(E_grid, 21.0)
        result_tbabs_21 = tbabs_transmission(E_grid, 21.0)
        assert np.array_equal(result_wabs_21, result_tbabs_21), \
            "wabs_transmission and tbabs_transmission differ at log_nh=21"

        # Test at log_nh = 23
        result_wabs_23 = wabs_transmission(E_grid, 23.0)
        result_tbabs_23 = tbabs_transmission(E_grid, 23.0)
        assert np.array_equal(result_wabs_23, result_tbabs_23), \
            "wabs_transmission and tbabs_transmission differ at log_nh=23"


def test_tbabs_transmission_emits_deprecation_warning():
    """Calling tbabs_transmission emits a DeprecationWarning with correct message."""
    from tengri.components.xray.xray import tbabs_transmission

    E_grid = jnp.array([0.5, 1.0])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = tbabs_transmission(E_grid, 22.0)

        # Check that exactly one warning was raised
        assert len(w) == 1, f"Expected 1 warning, got {len(w)}"

        # Check it's a DeprecationWarning
        assert issubclass(w[0].category, DeprecationWarning), \
            f"Expected DeprecationWarning, got {w[0].category}"

        # Check message contains 'wabs' and 'Wilms' (mentioning the convention)
        msg = str(w[0].message)
        assert "wabs" in msg.lower(), \
            f"Warning message should mention 'wabs': {msg}"
        assert "wilms" in msg.lower(), \
            f"Warning message should mention 'Wilms': {msg}"
        assert "wabs_transmission" in msg, \
            f"Warning message should name the new function: {msg}"
