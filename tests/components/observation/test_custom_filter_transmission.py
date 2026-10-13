# SPDX-License-Identifier: BSD-3-Clause
"""Custom filter transmission is validated at registration, before any division by it.

An all-zero or non-finite transmission curve used to register silently and later divide
by zero when a band was normalized. Small negative noise in a measured curve is legitimate
and is clipped to zero without complaint.

Markers: bounds (CI-enforced per tests/TESTING.md)
"""

import warnings

import numpy as np
import pytest

from tengri.observation.filters import FILTER_REGISTRY, load_filter, register_filter
from tengri.observation.filters.custom import _USER_FILTER_REGISTRY, _sanitize_filter_curve

pytestmark = pytest.mark.bounds


@pytest.fixture
def clean_registry():
    """Clear the user registry before and after each test."""
    initial_state = _USER_FILTER_REGISTRY.copy()
    _USER_FILTER_REGISTRY.clear()
    yield _USER_FILTER_REGISTRY
    _USER_FILTER_REGISTRY.clear()
    _USER_FILTER_REGISTRY.update(initial_state)


@pytest.fixture
def wave():
    """Optical grid [Angstrom]."""
    return np.linspace(5000.0, 7000.0, 100)


def test_all_zero_transmission_is_rejected_at_registration(clean_registry, wave):
    """A curve with max transmission 0 has nothing to normalize against."""
    with pytest.raises(ValueError, match="transmission"):
        register_filter("all_zero_band", wave, np.zeros_like(wave))
    assert "all_zero_band" not in FILTER_REGISTRY
    assert "all_zero_band" not in _USER_FILTER_REGISTRY


def test_all_negative_transmission_is_rejected(clean_registry, wave):
    """A curve whose peak is non-positive is rejected, not clipped to zero."""
    with pytest.raises(ValueError, match="transmission"):
        register_filter("all_negative_band", wave, -np.ones_like(wave))


def test_non_finite_transmission_is_rejected(clean_registry, wave):
    """NaN or inf transmission is rejected at registration."""
    trans = np.exp(-0.5 * ((wave - 6000.0) / 300.0) ** 2)
    trans[10] = np.nan
    with pytest.raises(ValueError, match="transmission"):
        register_filter("nan_band", wave, trans)


def test_small_negative_noise_still_registers(clean_registry, wave):
    """Measured curves carry small negative noise; it is clipped and the band registers."""
    trans = np.exp(-0.5 * ((wave - 6000.0) / 300.0) ** 2)
    trans[:5] = -1e-4
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        register_filter("noisy_band", wave, trans)
    fc = load_filter("noisy_band")
    assert float(np.min(np.asarray(fc.trans))) >= 0.0


def test_large_negative_clip_warns(wave):
    """Clipping a negative value above 1% of the peak emits a UserWarning."""
    trans = np.exp(-0.5 * ((wave - 6000.0) / 300.0) ** 2)
    trans[3] = -0.2
    with pytest.warns(UserWarning, match="negative"):
        _sanitize_filter_curve(wave, trans)


def test_sanitize_keeps_sort_and_duplicate_averaging(wave):
    """Sorting and duplicate averaging are unchanged by the new validation."""
    w = np.array([6000.0, 5000.0, 6000.0])
    t = np.array([0.6, 0.1, 0.4])
    w_out, t_out = _sanitize_filter_curve(w, t)
    np.testing.assert_allclose(w_out, [5000.0, 6000.0])
    np.testing.assert_allclose(t_out, [0.1, 0.5])
