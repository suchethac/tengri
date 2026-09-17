# SPDX-License-Identifier: BSD-3-Clause
"""Regression: ``Spectroscopy(wave_obs=...)`` silently accepts invalid grids (#2172).

A wavelength grid containing NaN, negative, or unsorted values is accepted at
construction, builds into a model without complaint, and produces a prediction
silently wrong: one NaN wavelength destroys all 50 pixels (all go non-finite),
while negative and unsorted grids produce a finite, plausible-looking spectrum
whose peak is identical to the clean control. A user who has a single bad pixel
in an instrument's wavelength solution gets a wholly NaN spectrum with no
indication of why.

Measured on main@7b3276004: one NaN → 50/50 pixels non-finite; one -500.0 →
finite spectrum, peak 2.765e-27 (identical to clean); two swapped values →
finite, peak 2.765e-27 (identical to clean).

Fix: validate ``wave_obs`` in ``Spectroscopy.__post_init__``:
- all finite (no NaN/inf);
- all strictly positive;
- strictly increasing (descending grids are REFUSED with a "reverse it" hint);
- ``calibration_order`` ≥ 0.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.observation.spectroscopy import Spectroscopy

pytestmark = pytest.mark.regression_bug


def test_wave_obs_with_nan():
    """Single NaN in wave_obs raises ValueError with the offending index."""
    wave_obs = jnp.linspace(4000.0, 7000.0, 50)
    wave_obs = wave_obs.at[25].set(jnp.nan)

    with pytest.raises(ValueError, match=r"(NaN|nan|index 25|non-finite)"):
        Spectroscopy(wave_obs=wave_obs)


def test_wave_obs_with_negative():
    """Single negative value in wave_obs raises ValueError with index and 'positive'."""
    wave_obs = jnp.linspace(4000.0, 7000.0, 50)
    wave_obs = wave_obs.at[10].set(-500.0)

    with pytest.raises(ValueError, match=r"(negative|positive|index 10|non-positive)"):
        Spectroscopy(wave_obs=wave_obs)


def test_wave_obs_unsorted():
    """Non-increasing values in wave_obs raises ValueError with first bad index."""
    wave_obs = jnp.linspace(4000.0, 7000.0, 50)
    # Swap two adjacent values to make it non-increasing
    swapped = jnp.array([wave_obs[26], wave_obs[25]])
    wave_obs = jnp.concatenate([wave_obs[:25], swapped, wave_obs[27:]])

    with pytest.raises(ValueError, match=r"(increasing|sorted|index|order)"):
        Spectroscopy(wave_obs=wave_obs)


def test_wave_obs_descending():
    """Fully descending wavelength grid raises ValueError with hint to reverse."""
    wave_obs = jnp.linspace(7000.0, 4000.0, 50)  # Descending

    with pytest.raises(ValueError, match=r"(reverse|ascending|increasing)"):
        Spectroscopy(wave_obs=wave_obs)


def test_calibration_order_negative():
    """calibration_order < 0 raises ValueError."""
    wave_obs = jnp.linspace(4000.0, 7000.0, 50)

    with pytest.raises(ValueError, match=r"(calibration_order|non-negative)"):
        Spectroscopy(wave_obs=wave_obs, calibration_order=-1)


def test_clean_wave_obs():
    """Clean wavelength grid constructs successfully."""
    wave_obs = jnp.linspace(4000.0, 7000.0, 50)
    spec = Spectroscopy(wave_obs=wave_obs)
    assert spec.n_pixels == 50
