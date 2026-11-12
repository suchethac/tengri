# SPDX-License-Identifier: BSD-3-Clause
"""MAP certification compiles its value-and-gradient and Hessian once per objective.

``stationarity`` used to wrap the loss in a fresh ``jax.jit`` on every call, so each
certificate (and each continuation round, and each restart of the multistart) re-traced
and recompiled the same program: 23-40 s per call on a float32 spectroscopic fit. The
compiled pair is now cached per loss object, weakly, so a dropped loss frees its
programs.

The contract pinned here:

* a second ``stationarity`` call on the same loss compiles nothing;
* the cached result is identical to a fresh compile of the same objective;
* the cache holds no strong reference to the loss.
"""

from __future__ import annotations

import gc
import weakref

import jax.numpy as jnp
import pytest
from jax._src import test_util as jtu

from tengri.inference.map_stationarity import stationarity

pytestmark = pytest.mark.contract


def _make_loss():
    """A fresh loss object with a fixed, non-trivial body (200-D, positive definite)."""
    curvature = jnp.diag(jnp.arange(1.0, 201.0))

    def loss(vec):
        return 0.5 * vec @ curvature @ vec + jnp.sum(jnp.sin(vec))

    return loss


def test_second_call_on_the_same_loss_compiles_nothing():
    loss = _make_loss()
    x = jnp.ones(200)
    stationarity(loss, x)
    with jtu.count_jit_compilation_cache_miss() as misses:
        stationarity(loss, x)
    assert misses() == 0, f"second stationarity call recompiled {misses()} programs"


def test_cached_result_is_identical_to_a_fresh_compile():
    x = jnp.ones(200)
    loss = _make_loss()
    stationarity(loss, x)
    cached = stationarity(loss, x)
    fresh = stationarity(_make_loss(), x)
    assert cached == fresh


def test_cache_holds_no_strong_reference_to_the_loss():
    loss = _make_loss()
    ref = weakref.ref(loss)
    stationarity(loss, jnp.ones(200))
    del loss
    gc.collect()
    assert ref() is None, "the stationarity cache keeps the loss alive"
