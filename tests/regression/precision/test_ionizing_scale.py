"""Test ionizing-SED mass-scale float32 safety (Balmer decrement guard).

Validates that reparametrizing the ionizing-SED scale as a log offset
prevents float32 overflow in the Cue nebular ionizing flux.
Balmer decrement H-alpha/H-beta ≈ 2.86 (Case B, independent of total mass scale).

TIER A GUARANTEE (mixed-precision, forward_dtype="float32" under x64=True):
  - test A (test_balmer_decrement_mixed_precision_f32): PASSES.
    Ionizing SED computed in f32; scalars remain f64. The log-offset
    reparametrization keeps the Balmer decrement finite and correct.

TIER B DEFERRAL (pure-float32, jax.enable_x64(False)):
  - test C (test_ionizing_sed_pure_float32_cue_only): XFAIL.
    Q_H integral (_integrate_nion) yields ~1e56 photons/s, exceeding float32 max (~3.4e38).
    Overflow to inf regardless of summand precision. Fix requires log_nion (log10 Q_H)
    output contract change across nebular consumers (Cue, CloudyGrid, compute_nion, q_h).
    That is a reduction-reformulation, scoped to Tier B. This task (Tier A) is the precursor.
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


@pytest.mark.xfail(
    reason=(
        "Tier B: Q_H integral (_integrate_nion) yields ~1e56 photons/s, exceeding float32 max. "
        "Pure-f32 needs log_nion contract across Cue/CloudyGrid/compute_nion/q_h. "
        "Tier A task (_sed_ion reparam) is the precursor."
    ),
    strict=True,
)
def test_ionizing_sed_pure_float32_cue_only(ssp_bare):
    """(C) Pure-float32 isolated stellar+Cue (no AGN SKIRTOR blocker).

    Build a minimal model with stellar SFH + Cue nebular only (no AGN, dust IR,
    radio, xray). Run under pure float32 (jax.enable_x64(False)) to validate
    that the ionizing-SED reparametrization (_sed_ion via apply_log10_scale)
    is safe and does not overflow.

    This test isolates the ionizing-SED path without the AGN SKIRTOR dtype
    mismatch that affects the panchromatic test_B.
    """
    from .conftest import build_minimal_cue_model

    # Build f64 reference
    m64 = build_minimal_cue_model(ssp_bare, "float64")
    p = dict(m64.spec.sample(jax.random.PRNGKey(0)))
    pred64 = m64.predict(p)
    q_h_64 = float(pred64.properties["q_h"])
    dec64 = float(pred64.properties["balmer_decrement"])

    # Build and predict under pure float32
    with jax.enable_x64(False):
        m32 = build_minimal_cue_model(ssp_bare, "float32")
        pred32 = m32.predict(p)
        q_h_32 = float(pred32.properties["q_h"])
        dec32 = float(pred32.properties["balmer_decrement"])

    # Validate ionizing path (q_h) is finite
    assert np.isfinite(q_h_32), (
        f"pure-f32 q_h is non-finite: {q_h_32}; indicates _sed_ion overflow in Cue ionizing flux"
    )

    # Validate Balmer decrement is finite and in Case B range
    assert np.isfinite(dec32), (
        f"pure-f32 Balmer decrement is non-finite: {dec32}; indicates _sed_ion scale issue"
    )
    assert 2.5 < dec32 < 3.2, f"pure-f32 Balmer decrement {dec32} outside Case B range [2.5, 3.2]"

    # Compare to f64 reference
    assert_allclose(q_h_32, q_h_64, rtol=5e-3)
    assert_allclose(dec32, dec64, rtol=5e-3)
