# SPDX-License-Identifier: BSD-3-Clause
"""Test: log10 ionizing photon production rate property (float32-safe).

Validates that pred.log_q_h is finite in both float64 and float32, while
pred.q_h (the linear form) overflows in float32 — demonstrating the need
for the log-domain property.

See issue #1206 (Tier B).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


def test_log_q_h_property(ssp_bare):
    """Validate log_q_h property existence, values, and float32 safety."""
    from .conftest import build_minimal_cue_model

    # Build minimal Cue model in float64
    model = build_minimal_cue_model(ssp_bare, forward_dtype="float64")

    # 1. Check availability
    assert "log_q_h" in model.available_properties, "log_q_h should be in available_properties"

    # Predict
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    pred = model.predict(params)

    # 2. Check sugar (pred.log_q_h equals pred.properties["log_q_h"])
    assert jnp.allclose(pred.log_q_h, pred.properties["log_q_h"], atol=0.0), (
        "sugar accessor must be exact"
    )

    # 3. Check finiteness and consistency (log_q_h = log10(q_h))
    log_q_h = np.float64(pred.log_q_h)
    q_h = np.float64(pred.q_h)
    assert np.isfinite(log_q_h), "log_q_h must be finite"
    assert np.any(log_q_h != 0.0), (
        "`log_q_h` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )
    assert np.isfinite(q_h), "q_h must be finite in float64"
    assert np.any(q_h != 0.0), (
        "`q_h` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )
    assert np.allclose(log_q_h, np.log10(q_h), rtol=1e-12), "log_q_h should equal log10(q_h)"

    # 3b. log_q_h is a direct read of the published derived["log_nion"] (bit-exact)
    state = model.predict_state(params)
    assert jnp.allclose(pred.log_q_h, state.derived["log_nion"], atol=0.0), (
        'log_q_h must directly equal derived["log_nion"]'
    )

    # 4. Pure float32: log_q_h finite, q_h overflows
    # disable_jit keeps the forward eager so the float32 overflow is observable
    with jax.disable_jit(), jax.enable_x64(False):
        # Rebuild model inside f32 context
        model_f32 = build_minimal_cue_model(ssp_bare, forward_dtype="float32")
        params_f32 = dict(model_f32.spec.sample(jax.random.PRNGKey(0)))
        pred_f32 = model_f32.predict(params_f32)

        log_q_h_f32 = np.asarray(pred_f32.log_q_h, dtype=np.float32)
        q_h_f32 = np.asarray(pred_f32.q_h, dtype=np.float32)

        # log_q_h must be finite
        assert np.isfinite(log_q_h_f32), "log_q_h must be finite in pure float32 (the whole point)"
        assert np.any(log_q_h_f32 != 0.0), (
            "`log_q_h_f32` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )

        # q_h overflows to inf (the problem we're solving)
        assert not np.isfinite(q_h_f32), (
            "q_h should NOT be finite in pure float32 (overflow ~1e56)"
        )
