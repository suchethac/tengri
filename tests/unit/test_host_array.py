# SPDX-License-Identifier: BSD-3-Clause
"""Host tables and their per-trace device copies (#2271).

Three defects the pair ``host_array`` / ``device_table`` exists to prevent, all
measured on JAX 0.11.1:

* A module-scope ``jnp`` array is a device buffer allocated at import, so its
  dtype is decided by the x64 state at import time, and a backend with no
  float64 (jax-mps) cannot import the package at all.
* JAX memoizes a numpy object's device conversion **by object identity with
  the dtype of its first conversion**. The same module-level array reused
  across traces under a different x64 state comes back with the old dtype:
  loudly (an MLIR verifier error) in operator positions, and **silently** from
  ``jnp.take`` / indexing, which returned float64 under ``enable_x64(False)``.
  An explicit dtype at the conversion is what breaks the memo.
* ``__jax_array__`` is refused for jit arguments ("Triggering __jax_array__()
  during abstractification is no longer supported"), so no array-like wrapper
  can make a table automatically safe; the conversion has to be at the use.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.utils.host_array import device_table, host_array

pytestmark = pytest.mark.unit

TABLE = host_array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5])


def test_host_array_is_plain_contiguous_float64_numpy_and_not_a_device_array():
    assert type(TABLE) is np.ndarray
    assert not isinstance(TABLE, jax.Array)
    assert TABLE.dtype == np.float64
    assert TABLE.flags["C_CONTIGUOUS"] and min(TABLE.strides) > 0
    flipped = host_array(np.arange(6.0)[::-1])
    assert flipped.flags["C_CONTIGUOUS"] and min(flipped.strides) > 0, (
        "a reversed view has a negative stride, which jax-mps refuses at device_put"
    )


def test_a_scalar_table_keeps_rank_zero():
    """``np.ascontiguousarray`` promotes a 0-d array to shape ``(1,)``; a scalar
    constant that comes back as a 1-vector breaks every ``jnp.stack`` of
    parameters it enters (measured: Cue's ``_prepare_nn_params``)."""
    s = host_array(math.log10(4.0 * math.pi))
    assert s.shape == () and s.ndim == 0
    assert device_table(s).shape == ()
    assert host_array(np.float64(2.5)).shape == ()


def test_device_table_follows_the_x64_state_of_the_trace_not_of_import():
    with jax.enable_x64(True):
        assert device_table(TABLE).dtype == jnp.float64
        assert device_table(host_array(1.5)).dtype == jnp.float64
    with jax.enable_x64(False):
        assert device_table(TABLE).dtype == jnp.float32
        assert device_table(host_array(1.5)).dtype == jnp.float32


def test_a_zero_d_table_is_strongly_typed():
    """It must promote a float32 operand to the process dtype the way the 0-d jnp
    constants it replaces did; a weak Python float would not (measured: 3e-6 dex
    on log_line_lums)."""
    with jax.enable_x64(True):
        out = jnp.ones(3, dtype=jnp.float32) - device_table(host_array(1.5))
        assert out.dtype == jnp.float64


@pytest.mark.parametrize("order", ["on_then_off", "off_then_on"])
@pytest.mark.parametrize(
    "form",
    ["take", "index", "searchsorted", "interp", "sum", "operator", "vmap_input"],
)
def test_the_same_table_retraced_under_the_other_x64_state_has_that_states_dtype(order, form):
    """The memo defect, pinned per entry point and per order, on the module TABLE
    object itself (a fresh object each call would hide it)."""

    def body(x):
        t = device_table(TABLE)
        idx = jnp.clip(x.astype(jnp.int32), 0, 14)
        if form == "take":
            return jnp.take(t, idx)
        if form == "index":
            return t[idx]
        if form == "searchsorted":
            return jnp.searchsorted(t, x, side="right").astype(x.dtype)
        if form == "interp":
            return jnp.interp(x, t, t * 2.0)
        if form == "sum":
            return jnp.sum(t) + x
        if form == "operator":
            return jnp.sum(x[..., None] / t, axis=-1)
        return jax.vmap(lambda ti: ti * x)(t).sum(axis=0)

    states = [True, False] if order == "on_then_off" else [False, True]
    for x64 in states:
        with jax.enable_x64(x64):
            out = jax.jit(body)(jnp.linspace(0.7, 6.9, 5))
            want = jnp.float64 if x64 else jnp.float32
            assert out.dtype == want, f"{form} {order}: x64={x64} gave {out.dtype}"
