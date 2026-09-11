# SPDX-License-Identifier: BSD-3-Clause
"""Parity test for lnu_to_fnu and fnu_to_lnu conversions.

Validates that the log-scale refactoring (issue #1186) preserves float64
accuracy (rtol ≤ 1e-12 vs old formula) and produces finite float32 results
(rtol ≤ 2e-3 from f64).

Float32 tests run inside jax.enable_x64(False) context to ensure no silent
upcast to float64 masks underflow bugs; this matches test_float32_parity.py.

See: src/tengri/utils/conversions.py:438, :472.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.cosmology import luminosity_distance
from tengri.utils.conversions import fnu_to_lnu, lnu_to_fnu

pytestmark = pytest.mark.regression_bug


@pytest.mark.parametrize("redshift", [0.01, 1.0, 6.0])
def test_lnu_to_fnu_parity_f64(redshift):
    """lnu_to_fnu matches old formula in float64 (rtol ≤ 1e-12)."""
    lnu = jnp.asarray(2.0e34)  # typical stellar luminosity
    z = jnp.asarray(redshift)

    # Compute luminosity distance (returns cm)
    dl_cm = jnp.asarray(luminosity_distance(float(z)))

    # New implementation
    result = lnu_to_fnu(lnu, dl_cm, z)

    # Old formula: lnu * (1+z) / (4π d_L²)
    old_result = lnu * (1.0 + z) / (4.0 * jnp.pi * dl_cm**2)

    # Verify match
    np.testing.assert_allclose(result, old_result, rtol=1e-12)


@pytest.mark.parametrize("redshift", [0.01, 1.0, 6.0])
def test_lnu_to_fnu_finite_f32(redshift):
    """lnu_to_fnu produces finite float32 results (rtol ≤ 2e-3 from f64).

    Runs inside jax.enable_x64(False) to prevent silent upcast to float64.
    """
    lnu = jnp.asarray(2.0e34)
    z = jnp.asarray(redshift)
    dl_cm = jnp.asarray(luminosity_distance(float(z)))

    # f64 result (reference)
    result_f64 = lnu_to_fnu(lnu, dl_cm, z)

    # f32 result: build inputs and call inside x64=False context
    with jax.enable_x64(False):
        lnu_f32 = jnp.asarray(2.0e34, dtype=jnp.float32)
        z_f32 = jnp.asarray(redshift, dtype=jnp.float32)
        dl_cm_f32 = jnp.asarray(luminosity_distance(float(z)), dtype=jnp.float32)
        result_f32 = lnu_to_fnu(lnu_f32, dl_cm_f32, z_f32)

    # Check finite
    assert jnp.isfinite(result_f32).all(), f"f32 result has NaN/Inf: {result_f32}"
    assert jnp.any(result_f32 != 0.0), (
        "`result_f32` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )

    # Check match (relaxed tolerance for float32)
    np.testing.assert_allclose(float(result_f32), float(result_f64), rtol=2e-3)


@pytest.mark.parametrize("redshift", [0.01, 1.0, 6.0])
def test_fnu_to_lnu_parity_f64(redshift):
    """fnu_to_lnu matches old formula in float64 (rtol ≤ 1e-12)."""
    fnu = jnp.asarray(1e-29)  # typical flux
    z = jnp.asarray(redshift)
    dl_cm = jnp.asarray(luminosity_distance(float(z)))

    # New implementation
    result = fnu_to_lnu(fnu, dl_cm, z)

    # Old formula: fnu * 4π d_L² / (1+z)
    old_result = fnu * 4.0 * jnp.pi * dl_cm**2 / (1.0 + z)

    # Verify match
    np.testing.assert_allclose(result, old_result, rtol=1e-12)


@pytest.mark.parametrize("redshift", [0.01, 1.0, 6.0])
def test_fnu_to_lnu_finite_f32(redshift):
    """fnu_to_lnu produces finite float32 results (rtol ≤ 2e-3 from f64).

    Runs inside jax.enable_x64(False) to prevent silent upcast to float64.
    """
    fnu = jnp.asarray(1e-29)
    z = jnp.asarray(redshift)
    dl_cm = jnp.asarray(luminosity_distance(float(z)))

    # f64 result (reference)
    result_f64 = fnu_to_lnu(fnu, dl_cm, z)

    # f32 result: build inputs and call inside x64=False context
    with jax.enable_x64(False):
        fnu_f32 = jnp.asarray(1e-29, dtype=jnp.float32)
        z_f32 = jnp.asarray(redshift, dtype=jnp.float32)
        dl_cm_f32 = jnp.asarray(luminosity_distance(float(z)), dtype=jnp.float32)
        result_f32 = fnu_to_lnu(fnu_f32, dl_cm_f32, z_f32)

    # Check finite
    assert jnp.isfinite(result_f32).all(), f"f32 result has NaN/Inf: {result_f32}"
    assert jnp.any(result_f32 != 0.0), (
        "`result_f32` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )

    # Check match (relaxed tolerance for float32)
    np.testing.assert_allclose(float(result_f32), float(result_f64), rtol=2e-3)
