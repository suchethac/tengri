# SPDX-License-Identifier: BSD-3-Clause
"""Test ionizing-SED mass-scale float32 safety (Balmer decrement guard).

Validates that reparametrizing the ionizing-SED scale as a log offset
prevents float32 overflow in the Cue nebular ionizing flux.
Balmer decrement H-alpha/H-beta: intrinsically ≈ 2.86 (Case B, independent of total
mass scale), and **larger than that once reddened**. The fixture is dusty
(``dust_tau_bc`` ~ 2.05), so the observed decrement is ~6.81; the tests assert it
sits *above* intrinsic rather than at it. Asserting the intrinsic range on an
attenuated model is how #1833 stayed hidden — see ``INTRINSIC_DECREMENT_TAU_ZERO``.

TIER A, forward_dtype="float32" under x64=True — NOT a mixed-precision guarantee:
  - test A (test_balmer_decrement_mixed_precision_f32): PASSES, but not for the
    reason its name gives. ``forward_dtype`` casts nothing (#1433), so the two
    builds it compares are bit-identical (measured: both 2.788906888791338) and
    the comparison against the f64 reference cannot fail. What the test still
    establishes is the physics: the decrement is finite and above the intrinsic
    Case B ratio. Float32 safety of the ionizing SED is established by the pure-float32
    tests below, which use ``jax.enable_x64(False)`` — the mechanism that works.

TIER B STEP 1 DELIVERED (pure-float32, jax.enable_x64(False)) — the log_nion contract (#1206):
  - test C1 (test_log_q_h_pure_float32_cue_only): PASSES.
    The log_nion reparametrization makes log_q_h (= log10 Q_H) and the nebular continuum
    rest_sed() (L_nu ~1e29, float32-representable) finite and float64-accurate. This is the
    usable pure-float32 ionizing diagnostic the contract provides.
  - test C2 (test_linear_observables_pure_float32_cue_only): XFAIL(strict) — #1206 item 3.
    The linear q_h property (~1e56 photons/s) and the erg/s line_lums (~1e41, hence
    balmer_decrement) still exceed float32 max. Returning them in L_sun/log10 is a breaking
    unit change (#1206 item 3), not yet done. log_q_h is the finite replacement.
"""

import jax
import numpy as np
import pytest
from numpy.testing import assert_allclose

from .conftest import build_model

pytestmark = pytest.mark.regression_bug

#: Intrinsic (unattenuated) Case B Halpha/Hbeta ratio at 1e4 K.
CASE_B_INTRINSIC = 2.86

#: The decrement this fixture returns with both dust screens at tau = 0 —
#: i.e. the intrinsic ratio the model actually produces. Measured 2.788907.
#:
#: This number is also what the two decrement tests below *used* to see with
#: dust switched ON, which is how #1833 hid: the fixture draws
#: ``dust_tau_bc = 2.05`` and ``dust_tau_diff = 0.79``, so an unattenuated
#: reading was the symptom of attenuation never reaching the Balmer lines.
#: #1841 fixed that, the decrement rose to 6.8075 (0.969 mag of differential
#: extinction, consistent with tau_bc + tau_diff = 2.84), and these tests
#: failed — because they asserted the *intrinsic* range on an *observed*
#: quantity. Asserting "above intrinsic" instead is both correct physics and
#: strictly stronger: it goes red if dust ever stops reddening the lines again.
INTRINSIC_DECREMENT_TAU_ZERO = 2.788907

#: Generous ceiling. The fixture's tau implies ~6.8; anything past this is a
#: runaway rather than a dusty draw.
DECREMENT_CEILING = 30.0


def test_balmer_decrement_mixed_precision_f32(ssp_bare):
    """(A) The Balmer decrement is finite and Case B, at ``forward_dtype="float32"``.

    This test used to be described as the Tier A mixed-precision guarantee: "the
    ionizing SED is computed in float32, but scalars stay f64". It is not. The
    ``forward_dtype`` knob casts nothing (#1433), so ``m32`` and ``m64`` below are
    the same computation and ``assert_allclose(dec32, dec64)`` compares a float64
    result against itself — measured bit-identical, 2.788906888791338 both.

    What survives is worth keeping, so the test stays: the decrement is finite
    and above the intrinsic Case B ratio on this dusty panchromatic model, which
    is a real check on the log-offset reparametrization. It is just not a float32
    check. For that see the pure-float32 tests in this file and
    ``tests/regression/precision/test_forward_dtype_knob.py``.
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
    # The fixture is dusty (tau_bc ~ 2.05), so the OBSERVED decrement must sit
    # above the intrinsic Case B ratio. Asserting the intrinsic range here is
    # what let #1833 pass unnoticed — see INTRINSIC_DECREMENT_TAU_ZERO.
    assert CASE_B_INTRINSIC < dec32 < DECREMENT_CEILING, (
        f"f32 Balmer decrement {dec32} outside "
        f"({CASE_B_INTRINSIC}, {DECREMENT_CEILING}) — an attenuated decrement "
        "must exceed the intrinsic Case B ratio; at or below it means dust is "
        "not reaching the Balmer lines"
    )
    assert_allclose(dec32, dec64, rtol=5e-3)


#: Derived keys that are non-finite in pure float32 for the full model, measured.
#: Every one is a *linear* transition publish whose log counterpart is finite —
#: which is what makes this #1206 item 3 and nothing else.
_LINEAR_OVERFLOW_KEYS = (
    "L_absorbed",
    "L_age",
    "L_agn_bol",
    "L_ir",
    "line_lums",
    "nion",
    "stellar_mass_scale",
)


def test_pure_float32_non_finites_are_the_linear_transition_publishes(ssp_bare):
    """Pins *why* the Balmer decrement below is nan, so the xfail cannot drift.

    A strict xfail records that something fails, not that it fails for the
    recorded reason — a later, unrelated regression tripping the same assertion
    is absorbed silently. The reason on the xfail below was stale for exactly
    that long: it blamed the SKIRTOR ``interp_nd_triweight`` dtype mismatch
    (#1206 item 4, shipped) and the linear ``stellar_mass_scale`` needing a log
    variant (shipped as ``log_stellar_mass_scale``). Both were fixed while the
    test kept xfailing for a different cause.

    So assert the cause directly: in pure float32 every linear erg/s (or
    photons/s) transition publish overflows, and every log counterpart stays
    finite. If that ever stops being true, this test fails here rather than
    letting the xfail quietly stand for something new.
    """
    m64 = build_model(ssp_bare, "float64")
    p = dict(m64.spec.sample(jax.random.PRNGKey(0)))
    p["redshift"] = 1.0

    with jax.enable_x64(False):
        derived = build_model(ssp_bare, "float32").predict_state(p).derived

    non_finite = {
        k
        for k in _LINEAR_OVERFLOW_KEYS
        if k in derived and not np.all(np.isfinite(np.asarray(derived[k], dtype=np.float64)))
    }
    assert non_finite == set(_LINEAR_OVERFLOW_KEYS) & set(derived), (
        "the pure-float32 overflow set has moved: expected every linear transition "
        f"publish to overflow, got {sorted(non_finite)}. If a key became finite, item 3 "
        "has advanced — promote it out of this list and re-check the xfail below"
    )

    for log_key in ("log_nion", "log_stellar_mass_scale", "log_L_ir"):
        if log_key in derived:
            value = np.asarray(derived[log_key], dtype=np.float64)
            assert np.all(np.isfinite(value)), (
                f"{log_key} is non-finite in pure float32 — the log contract that makes "
                "the linear overflow survivable has itself regressed"
            )


def test_balmer_decrement_pure_float32(ssp_bare):
    """(B) Pure-float32 under ``jax.enable_x64(False)``: the full model.

    Disable JAX x64 globally, forcing all scalars and arrays to float32.

    **This was a strict xfail against #1206 item 3, and the marker fired — but
    not for the work it named.** Its reason read: "every linear transition
    publish overflows ... balmer_decrement is a ratio of two of those lines, so
    it is nan." The first half is still true; ``line_lums`` remains erg/s and
    remains ``inf`` in float32, and the companion test above still pins that.
    The second half stopped following from it. #1837 routed the six line-ratio
    diagnostics through the ``log_line_lums`` companion on a peak-relative
    scale, so the ratio no longer reads the overflowed linear array at all —
    the common factor cancels exactly, and the decrement is recovered without
    item 3's breaking unit change.

    So the strictness did its job precisely: it refused to keep reporting a
    clean xfail on work that was already done. The lesson is that an xfail
    reason names a *cause*, and a consumer-side fix can remove the cause
    without the named work landing. Item 3 is still open.

    The assertions below are the reason this is worth keeping as a live test
    rather than deleting: the decrement is not merely finite in float32, it
    lands above the intrinsic Case B ratio (the fixture is reddened) and tracks
    the float64 reference to 5e-3.
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

    assert np.isfinite(dec32), f"pure-f32 Balmer decrement is non-finite: {dec32}"
    assert CASE_B_INTRINSIC < dec32 < DECREMENT_CEILING, (
        f"pure-f32 Balmer decrement {dec32} outside "
        f"({CASE_B_INTRINSIC}, {DECREMENT_CEILING}) — an attenuated decrement "
        "must exceed the intrinsic Case B ratio"
    )
    assert_allclose(dec32, dec64, rtol=5e-3)


def test_dust_reddens_the_balmer_lines(ssp_bare):
    """Zeroing the two dust screens must return the decrement to intrinsic.

    This is the test #1833 needed and did not have. The fixture draws
    ``dust_tau_bc = 2.05`` and ``dust_tau_diff = 0.79``, so the observed
    decrement has to exceed the intrinsic ratio; before #1841 it did not, and
    every assertion in this file was satisfied by an *unattenuated* reading of
    an attenuated model.

    Checking both ends is what makes it load-bearing. A bound on the observed
    value alone can be met by a model with no dust at all; pinning the tau = 0
    value as well says the difference between them is the attenuation.
    """
    m = build_model(ssp_bare, "float64")
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    p["redshift"] = 1.0

    dec_dusty = float(m.predict(p).properties["balmer_decrement"])

    no_dust = dict(p)
    for key in ("dust_tau_bc", "dust_tau_diff"):
        assert key in no_dust, f"fixture no longer samples {key}; update this test"
        no_dust[key] = 0.0
    dec_intrinsic = float(m.predict(no_dust).properties["balmer_decrement"])

    # Only the two screens were touched, so anything that moved is attenuation.
    assert_allclose(dec_intrinsic, INTRINSIC_DECREMENT_TAU_ZERO, rtol=1e-4)
    assert dec_dusty > dec_intrinsic, (
        f"dust did not redden the Balmer lines: dusty {dec_dusty} <= "
        f"intrinsic {dec_intrinsic} at tau_bc={p['dust_tau_bc']:.3f}, "
        f"tau_diff={p['dust_tau_diff']:.3f} (this is #1833)"
    )

    # Differential extinction implied by the ratio of ratios, for the record.
    differential_mag = 2.5 * np.log10(dec_dusty / dec_intrinsic)
    assert 0.5 < differential_mag < 2.0, (
        f"A(Hbeta) - A(Halpha) = {differential_mag:.3f} mag is not consistent "
        f"with tau_bc + tau_diff = "
        f"{p['dust_tau_bc'] + p['dust_tau_diff']:.2f}"
    )


def test_log_q_h_pure_float32_cue_only(ssp_bare):
    """(C1) Pure-float32 stellar+Cue: the log_nion contract makes the ionizing
    readout and the nebular continuum SED finite (issue #1206 step 1).

    log_q_h (= log10 Q_H) and rest_sed() (L_nu ~1e29, float32-representable) stay
    finite and track the float64 reference. This is what the log_nion
    reparametrization (the log-domain Q_H contract) delivers: a usable pure-float32
    ionizing diagnostic where the linear q_h property overflows (see the companion
    strict-xfail test). The gas_logqion Cue seam is exercised transitively — a broken
    seam would poison the nebular continuum, so rest_sed finiteness is the end-to-end
    check. Isolated stellar+Cue avoids the AGN SKIRTOR dtype blocker of test B.
    """
    from .conftest import build_minimal_cue_model

    m64 = build_minimal_cue_model(ssp_bare, "float64")
    p = dict(m64.spec.sample(jax.random.PRNGKey(0)))
    pred64 = m64.predict(p)
    log_q_h_64 = float(pred64.properties["log_q_h"])

    with jax.enable_x64(False):
        m32 = build_minimal_cue_model(ssp_bare, "float32")
        pred32 = m32.predict(p)
        log_q_h_32 = float(pred32.properties["log_q_h"])
        rest_sed_32 = np.asarray(pred32.rest_sed())

    assert np.isfinite(log_q_h_32), f"pure-f32 log_q_h non-finite: {log_q_h_32}"
    assert np.all(np.isfinite(rest_sed_32)), "pure-f32 rest_sed has non-finite entries"
    assert_allclose(log_q_h_32, log_q_h_64, atol=5e-3)  # dex


@pytest.mark.xfail(
    reason=(
        "Tier B item 3 (breaking unit change, not yet done): the linear q_h property "
        "(~1e56 photons/s) and the erg/s line_lums (~1e41, hence balmer_decrement) exceed "
        "float32 max. The log_nion contract (#1206 step 1) gives a finite log_q_h instead — "
        "see test_log_q_h_pure_float32_cue_only. Returning these in L_sun/log10 is #1206 item 3."
    ),
    strict=True,
)
def test_linear_observables_pure_float32_cue_only(ssp_bare):
    """(C2) The linear-erg/s observables that Tier B item 3 must still fix.

    In pure float32 the linear ``q_h`` overflows to inf and ``balmer_decrement``
    (a ratio of erg/s ``line_lums`` ~1e41) is nan. This test is expected to fail
    until item 3 rescales these to L_sun/log10; if it ever XPASSES, promote it.
    """
    from .conftest import build_minimal_cue_model

    with jax.enable_x64(False):
        m32 = build_minimal_cue_model(ssp_bare, "float32")
        p = dict(m32.spec.sample(jax.random.PRNGKey(0)))
        pred32 = m32.predict(p)
        q_h_32 = float(pred32.properties["q_h"])
        dec_32 = float(pred32.properties["balmer_decrement"])

    assert np.isfinite(q_h_32), f"linear q_h overflows float32: {q_h_32}"
    assert np.isfinite(dec_32), f"balmer_decrement is nan in float32: {dec_32}"
