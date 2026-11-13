# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #3433: zero flux gives +inf AB magnitude, not a floor value.

``fnu_to_ab_mag`` clamped F_nu to the dtype's smallest normal before the log,
so a band with zero flux came out at 46.22 mag in float32 and 701.4 mag in
float64, which reads as a faint detection. A non-positive flux now maps to
+inf, the mathematical limit of -2.5 log10(F_nu). The same rule applies to the
absolute-magnitude and magnitude-index siblings. Gradients stay finite at
positive flux (double-where, so the untaken branch never takes log of 0).

The float32 cases run with ``jax_enable_x64`` off, because the floor is
resolved from the x64 setting: with x64 on, a float32 zero flux already
reaches the floor's underflow to 0 and gives +inf, which hid the bug in the
suite's default configuration.
"""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.spectral_indices import _index_from_ratio
from tengri.utils.magnitudes import (
    LOG10_MAGGIES_ZP_CGS,
    fnu_to_ab_mag,
    lnu_to_absolute_ab_mag,
)
from tengri.utils.scale import representable_floor

pytestmark = pytest.mark.regression_bug

_DTYPES = [jnp.float32, jnp.float64]


@contextlib.contextmanager
def _precision(dtype):
    """float32 runs with x64 off (the configuration in which the floor underflows)."""
    if dtype == jnp.float32:
        with jax.enable_x64(False):
            yield
    else:
        with jax.enable_x64(True):
            yield


@pytest.mark.parametrize("dtype", _DTYPES)
def test_zero_flux_is_plus_inf(dtype):
    """A zero-flux band has no magnitude: +inf in both precisions."""
    with _precision(dtype):
        mag = fnu_to_ab_mag(jnp.asarray(0.0, dtype=dtype))
        assert mag.dtype == dtype
        assert bool(jnp.isposinf(mag))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_negative_flux_is_plus_inf(dtype):
    """Negative flux keeps the convention of the clamp it replaces: no detection, +inf."""
    with _precision(dtype):
        mag = fnu_to_ab_mag(jnp.asarray([-1e-30, -0.0], dtype=dtype))
        assert bool(jnp.all(jnp.isposinf(mag)))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_nan_flux_stays_nan(dtype):
    """NaN is not a zero flux; it must not come out as a finite or infinite magnitude."""
    with _precision(dtype):
        mag = fnu_to_ab_mag(jnp.asarray(jnp.nan, dtype=dtype))
        assert bool(jnp.isnan(mag))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_zero_luminosity_is_plus_inf(dtype):
    """The absolute-magnitude sibling follows the same rule."""
    with _precision(dtype):
        mag = lnu_to_absolute_ab_mag(jnp.asarray(0.0, dtype=dtype))
        assert bool(jnp.isposinf(mag))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_zero_ratio_magnitude_index_is_plus_inf(dtype):
    """The magnitude index -2.5 log10(F/F_C) at F = 0 is +inf, not a floor value."""
    with _precision(dtype):
        ratio = jnp.asarray([0.0, 1.0], dtype=dtype)
        idx = _index_from_ratio(ratio, feat_width=10.0, units="mag")
        assert bool(jnp.isposinf(idx[0]))
        assert float(idx[1]) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_positive_flux_above_floor_unchanged_bit_for_bit(dtype):
    """Positive flux gives the same bits as the pre-#3433 expression (clamp, then log)."""
    with _precision(dtype):
        fnu = jnp.asarray([1e-30, 3.631e-20, 1.5e-19, 2.0], dtype=dtype)

        @jax.jit
        def previous(f):
            # The pre-#3433 body, verbatim.
            return -2.5 * (
                jnp.log10(jnp.maximum(f, representable_floor(1e-300))) - LOG10_MAGGIES_ZP_CGS
            )

        np.testing.assert_array_equal(np.asarray(fnu_to_ab_mag(fnu)), np.asarray(previous(fnu)))


def test_grad_finite_at_positive_flux():
    """d m / d F_nu is finite where the flux is positive."""
    grad = jax.grad(lambda f: fnu_to_ab_mag(f))(jnp.asarray(1.5e-19, dtype=jnp.float64))
    assert bool(jnp.isfinite(grad))


def test_grad_finite_at_zero_flux():
    """The double-where keeps the gradient finite even where the magnitude is +inf.

    A single where would send the untaken log(0) branch a 0 * inf cotangent, i.e. NaN.
    """
    grad = jax.grad(lambda f: jnp.sum(fnu_to_ab_mag(f)))(
        jnp.asarray([0.0, 1.5e-19], dtype=jnp.float64)
    )
    assert bool(jnp.all(jnp.isfinite(grad)))
