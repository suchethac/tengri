# SPDX-License-Identifier: BSD-3-Clause
"""Source-pinned regression tests for the Cue clip discipline (#477).

The ``CueNebularSEDComponent`` — a thin wrapper over ``CueBackend`` — was
deleted in #738 (Phase 3b); the canonical path is ``NebularSEDComponent`` +
``CueBackend``, exercised by
``tests/components/nebular/test_cue_param_translation.py``,
``test_cue_hybrid_diagnostic.py`` and
``tests/regression/bug/test_bug_464_pytree_meta_arrays.py``.

What remains here is the ±50-dex clip regression on the cue-module
``predict_all_lines`` / ``predict_continuum`` functions, which is independent of
the (now-deleted) component.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract


def test_cue_predict_all_lines_clip_bounded_at_50dex():
    """The exponent clip in ``predict_all_lines`` must not exceed ±50 dex.

    The ±100-dex bound shipped with the original Cue component was wide enough
    that the +51-dex ``gas_logq = logU`` bug (#477) produced saturated-but-
    near-physical line luminosities instead of obviously-broken output. ±50
    dex is the discipline: any normalization slip ≥ 50 dex now hits the
    ceiling/floor uniformly and is visible in inspection.

    Asserts the **rule** (the effective bound is never wider than ±50 dex) rather
    than the source string that used to express it. The bound is now
    ``representable_exponent(50.0)``, which returns 50.0 in float64 and 38.53 in
    float32 -- because ``10**50`` is ``inf`` in float32, so the literal made the
    clip emit the very ``inf`` it exists to prevent (#1206). That is strictly
    *tighter* than ±50 and preserves this defense; a string match could not tell
    the two apart, and pinning the instance would have blocked the fix.
    """
    import inspect
    import math

    import jax

    from tengri.components.nebular import cue
    from tengri.utils.scale import representable_exponent

    for name, func in (
        ("predict_all_lines", cue.predict_all_lines),
        ("predict_continuum", cue.predict_continuum),
    ):
        source = inspect.getsource(func)
        assert "jnp.clip(exponent, -50.0, representable_exponent(50.0))" in source, (
            f"{name} no longer clips its exponent to a representable ±50 dex; see the "
            "#477 follow-up for why ±100 silently masked the gas_logq normalization "
            "bug, and #1206 for why the upper bound must be dtype-aware."
        )

    for x64, label in ((True, "float64"), (False, "float32")):
        with jax.enable_x64(x64):
            bound = representable_exponent(50.0)
            assert bound <= 50.0, f"{label} clip widened to {bound} dex, past the ±50 discipline"
            assert math.isfinite(10.0**bound), (
                f"{label} clip saturates to 10**{bound}, which that dtype cannot hold"
            )


def test_cue_predict_all_lines_clip_saturates_on_synthetic_bug():
    """A synthetic +51-dex error in ``gas_logq`` (the magnitude of the
    pre-#477 bug) must drive every line to the same saturated clip value,
    making the bug visually obvious rather than near-physical."""
    import os

    import chex
    import jax.numpy as jnp

    from tengri.components.nebular import _DEFAULT_CUE_WEIGHTS_PATH
    from tengri.components.nebular.cue import load_cue_weights, predict_all_lines

    if not os.path.exists(_DEFAULT_CUE_WEIGHTS_PATH):
        pytest.skip(f"Cue weights file not found at {_DEFAULT_CUE_WEIGHTS_PATH}")
    weights = load_cue_weights(str(_DEFAULT_CUE_WEIGHTS_PATH))

    # NN-ready 12-vector (the function broadcasts internally over the 16
    # batched sub-emulators). Values centered in their training ranges; the
    # test isn't about absolute luminosities, only about how the clip
    # responds to a synthetic normalization error.
    nn_params = jnp.array(
        [19.7, 5.3, 1.6, 0.6, 3.9, 0.01, 0.2, 48.5, 2.0, 0.0, 0.0, 0.0],
        dtype=jnp.float32,
    )
    correct_gas_logq = jnp.asarray(48.5)
    bug_gas_logq = jnp.asarray(-3.0)  # the pre-#477 value
    gas_logqion = jnp.asarray(52.0)

    _wav, lum_correct = predict_all_lines(
        nn_params=nn_params,
        weights=weights,
        gas_logq=correct_gas_logq,
        gas_logqion=gas_logqion,
    )
    _wav, lum_buggy = predict_all_lines(
        nn_params=nn_params,
        weights=weights,
        gas_logq=bug_gas_logq,
        gas_logqion=gas_logqion,
    )

    chex.assert_equal_shape([lum_correct, lum_buggy])
    chex.assert_tree_all_finite(lum_correct)
    chex.assert_tree_all_finite(lum_buggy)

    log_correct = jnp.log10(jnp.maximum(lum_correct, 1e-300))
    log_buggy = jnp.log10(jnp.maximum(lum_buggy, 1e-300))

    # Correct path: a healthy line forest spans many decades.
    span_correct = float(log_correct.max() - log_correct.min())
    assert span_correct > 1.0, (
        f"correct-gas_logq line luminosities should span multiple decades, got {span_correct} dex"
    )

    # Buggy +51-dex path: the bulk of lines (>90%) hit the +50 clip ceiling
    # uniformly. A few intrinsically-faint lines may sit below the ceiling
    # but the bulk is heavily peaked at the saturation value — that is the
    # bug-detection signal the tight clip provides. Pre-#477 the ±100 clip
    # let the same lines emerge at ~+99 dex, which looked plausible.
    saturated_frac = float((log_buggy >= 49.9).mean())
    assert saturated_frac > 0.9, (
        f"buggy +51-dex gas_logq should saturate the bulk of lines at the "
        f"+50 clip ceiling; only {saturated_frac:.1%} reached it. The clip "
        f"may be too loose to catch this class of normalization bug."
    )
    # And the median is right at the ceiling — the bug signature is a
    # degenerate distribution at the clip value.
    assert float(jnp.median(log_buggy)) >= 49.9
