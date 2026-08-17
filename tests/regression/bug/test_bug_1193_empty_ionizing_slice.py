# SPDX-License-Identifier: BSD-3-Clause
"""Regression: zero ionizing bins must not crash the scale/nion path (#1193 fallout).

Models whose wavelength grid has no bins below the Lyman limit (IR-focused
dust-emitter configs — the VI smoke emitters) have ``n_ion_bins == 0``. The
#1193 log-offset scale reparametrization introduced ``jnp.max`` over the
ionizing slice, and ``_integrate_nion`` takes an ``argmax`` — both reductions
have no identity on zero-size arrays and raise at trace time. The nightly
slow tier caught it (test_vi_smoke_per_emitter); this is the fast-tier lock.
"""

import jax
import jax.numpy as jnp
import pytest

from tengri.utils.scale import apply_log10_scale

pytestmark = pytest.mark.regression_bug


def test_apply_log10_scale_empty_array_passes_through():
    """max-of-empty has no identity — an empty array must pass through, not raise."""
    out = apply_log10_scale(jnp.zeros((0,)), jnp.asarray(-58.0))
    assert out.shape == (0,)
    assert jnp.all(jnp.isfinite(out))


def test_apply_log10_scale_empty_under_jit_and_grad():
    """The empty case must also survive jit tracing (where the crash actually fired)."""
    f = jax.jit(lambda s: jnp.sum(apply_log10_scale(jnp.zeros((0,)), s)))
    assert float(f(jnp.asarray(-58.0))) == 0.0
    assert jnp.isfinite(jax.grad(f)(jnp.asarray(-58.0)))


def test_apply_log10_scale_nonempty_unchanged():
    """The hardening must not perturb the normal path (bit-level)."""
    arr = jnp.array([1e-30, 3e-25, 7e-28])
    out = apply_log10_scale(arr, jnp.asarray(12.0))
    expected = arr * 10.0**12.0
    assert jnp.allclose(out / expected, 1.0, rtol=1e-12)
