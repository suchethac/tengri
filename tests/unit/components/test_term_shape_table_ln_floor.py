# SPDX-License-Identifier: BSD-3-Clause
"""The term-shape table's log floor equals the old ``ln(max(band, 1e-300))`` bit for bit."""

from __future__ import annotations

import numpy as np

from tengri.components._term_shape_table import _LN_FLOOR_LOG, _ln_band


def _old_ln(band: np.ndarray) -> np.ndarray:
    """The pre-#1568 expression, kept here as the reference the new path must match."""
    return np.log(np.maximum(band, 1.0e-300))


def test_floor_literal_is_log_of_1e_minus_300_in_float64():
    assert float(np.log(1.0e-300)) == _LN_FLOOR_LOG
    assert _LN_FLOOR_LOG.hex() == np.log(1.0e-300).hex()


def test_bit_identical_above_the_floor():
    rng = np.random.default_rng(0)
    band = np.concatenate(
        [
            10.0 ** rng.uniform(-299.0, 30.0, size=2000),
            np.array([1.0e-299, 2.0e-300, 1.0, 1.0e-12]),
        ]
    )
    assert np.array_equal(_ln_band(band).view(np.uint64), _old_ln(band).view(np.uint64))


def test_zero_and_below_floor_give_the_floor():
    band = np.array([0.0, -0.0, 1.0e-301, 1.0e-320, 5.0e-324])
    expected = np.full(band.shape, np.log(1.0e-300))
    assert np.array_equal(_ln_band(band).view(np.uint64), expected.view(np.uint64))
    assert np.array_equal(_old_ln(band).view(np.uint64), expected.view(np.uint64))


def test_nan_propagates_as_before():
    band = np.array([np.nan, 1.0])
    assert np.isnan(_ln_band(band)[0])
    assert np.isnan(_old_ln(band)[0])
