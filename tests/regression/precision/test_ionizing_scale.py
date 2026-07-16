"""Test ionizing-SED mass-scale float32 safety (Balmer decrement guard).

Validates that reparametrizing the ionizing-SED scale as a log offset
prevents float32 overflow in the Cue nebular ionizing flux.
Balmer decrement H-alpha/H-beta ≈ 2.86 (Case B, independent of total mass scale).
"""
import jax
import numpy as np
import pytest
from numpy.testing import assert_allclose

from .conftest import build_model

pytestmark = pytest.mark.regression_bug


def test_balmer_decrement_mixed_precision_f32(ssp_bare):
    """(A) Mixed-precision: forward_dtype=float32 under x64=True (Tier A guarantee).

    Build the model with forward_dtype="float32" while JAX x64 is enabled (default).
    The ionizing SED is computed in float32, but scalars like stellar_mass_scale
    remain f64. This tests that the log-offset reparametrization keeps the
    Balmer decrement finite and correct.
    """
    # Build f64 reference (x64=True, forward_dtype="float64")
    m64 = build_model(ssp_bare, "float64")
    p = dict(m64.spec.sample(jax.random.PRNGKey(0)))
    p["redshift"] = 1.0
    dec64 = float(m64.predict(p).properties["balmer_decrement"])

    # Build f32 with x64=True (mixed precision)
    m32 = build_model(ssp_bare, "float32")
    dec32 = float(m32.predict(p).properties["balmer_decrement"])

    # Decrement should be finite and close to the f64 reference
    assert np.isfinite(dec32), f"f32 Balmer decrement is non-finite: {dec32}"
    assert 2.7 < dec32 < 3.1, f"f32 Balmer decrement {dec32} off Case B range"
    assert_allclose(dec32, dec64, rtol=5e-3)


@pytest.mark.xfail(
    reason="Pure-f32 fails in AGN SKIRTOR interpolation (dtype mismatch in interp_nd_triweight),"
    " not due to the ionizing-SED scale. The Tier A guarantee (mixed-precision) is test A."
    " Tier B: published stellar_mass_scale scalar (~1e42) is f32-unrepresentable;"
    " consumers (dust energy balance, nebular backends) must use log-scaled variants.",
    strict=False,
)
def test_balmer_decrement_pure_float32(ssp_bare):
    """(B) Pure-float32: forward_dtype=float32 under jax.enable_x64(False).

    Disable JAX x64 globally, forcing all scalars and arrays to float32.
    If the published stellar_mass_scale (~1e42) still causes issues, mark this
    test as xfail and trace the consumer chain.

    Current failure: dtype mismatch in AGN SKIRTOR interpolation (triweight grid
    weights, reduce operands float64 vs initial float32) — separate from the
    ionizing-SED mass scale fix. The Tier A guarantee is test A (mixed-precision).
    """
    # Build f64 reference
    m64 = build_model(ssp_bare, "float64")
    p = dict(m64.spec.sample(jax.random.PRNGKey(0)))
    p["redshift"] = 1.0
    dec64 = float(m64.predict(p).properties["balmer_decrement"])

    # Build and predict under pure float32
    with jax.enable_x64(False):
        m32 = build_model(ssp_bare, "float32")
        dec32 = float(m32.predict(p).properties["balmer_decrement"])

    # In pure f32, if balmer_decrement is still non-finite / wrong, the
    # published stellar_mass_scale scalar (~1e42) is the culprit (Tier B fix).
    # For now, just check it's finite and in range.
    assert np.isfinite(dec32), f"pure-f32 Balmer decrement is non-finite: {dec32}"
    assert 2.7 < dec32 < 3.1, f"pure-f32 Balmer decrement {dec32} off Case B range"
    assert_allclose(dec32, dec64, rtol=5e-3)
