# SPDX-License-Identifier: BSD-3-Clause
"""Float32 guard floors must keep a zero input finite (#1492).

``1e-300`` and ``1e-50`` are below float32's smallest subnormal, so a literal
``jnp.maximum(x, 1e-300)`` is exactly ``0.0`` in float32 and ``log10(0)`` returns
``-inf``. Each site below is driven at a zero input in float32 (the precision
#1206 delivers) and must return a finite value. The expected values are the ones
the floor implies: ``log10(finfo(float32).tiny)`` replaces ``log10(0)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

_LOG10_TINY_F32 = float(np.log10(np.finfo(np.float32).tiny))


def test_mag_to_surface_brightness_zero_area_is_finite_float32():
    """A zero area is floored, not sent to ``log10(0)``, in float32."""
    from tengri.utils.magnitudes import mag_to_surface_brightness

    with jax.enable_x64(False):
        mu = mag_to_surface_brightness(jnp.float32(20.0), jnp.float32(0.0))
        assert bool(jnp.isfinite(mu))
        assert float(mu) != 0.0
        assert float(mu) == pytest.approx(20.0 + 2.5 * _LOG10_TINY_F32, rel=1e-5)


def test_surface_brightness_to_mag_zero_area_is_finite_float32():
    """The inverse conversion at a zero area is floored the same way."""
    from tengri.utils.magnitudes import surface_brightness_to_mag

    with jax.enable_x64(False):
        mag = surface_brightness_to_mag(jnp.float32(20.0), jnp.float32(0.0))
        assert bool(jnp.isfinite(mag))
        assert float(mag) != 0.0
        assert float(mag) == pytest.approx(20.0 - 2.5 * _LOG10_TINY_F32, rel=1e-5)


def test_fnu_to_ab_mag_zero_flux_is_finite_float32():
    """A zero flux density is floored to the smallest normal, not ``-inf`` dex."""
    from tengri.utils.magnitudes import fnu_to_ab_mag

    with jax.enable_x64(False):
        mag = fnu_to_ab_mag(jnp.float32(0.0))
        assert bool(jnp.isfinite(mag))
        assert float(mag) != 0.0


def test_compute_uv_slope_zero_luminosity_is_finite_float32():
    """Zero L_nu inside the FUV window gives a finite, non-zero slope, not NaN.

    The blue half of the window is zero; the red half carries a power law so the
    fitted slope is genuinely non-zero.
    """
    from tengri.components.nebular.fesc_model import compute_uv_slope

    with jax.enable_x64(False):
        wave = jnp.linspace(1000.0, 3000.0, 64, dtype=jnp.float32)
        l_nu = jnp.where(wave < 2000.0, 0.0, wave ** (-2.0))
        beta = compute_uv_slope(wave, l_nu)
        assert bool(jnp.isfinite(beta))
        assert float(beta) != 0.0


def test_nebular_log_stack_zero_is_finite_float32():
    """The nebular table log stack maps zero luminosity to a finite dex value.

    ``precompute_nebular_grid`` is too large to call in isolation, so this checks
    the exact expression its ``_stack_log`` evaluates, with the floor it now uses.
    """
    from tengri.utils.scale import representable_floor

    with jax.enable_x64(False):
        arr = jnp.zeros((4, 3), dtype=jnp.result_type(float))
        log = jnp.log10(jnp.maximum(arr, representable_floor(1e-300)))
        assert bool(jnp.all(jnp.isfinite(log)))
        assert bool(jnp.all(log != 0.0))
