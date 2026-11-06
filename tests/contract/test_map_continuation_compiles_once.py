# SPDX-License-Identifier: BSD-3-Clause
"""A MAP continuation round compiles once per objective, and matches the un-jitted round.

Each round of :func:`lbfgs_to_gtol` built fresh while-loop closures, so every round
compiled its condition and body again (about 2 programs per round, 84 s per round on a
float32 spectroscopic fit). The rounds of one certificate now share a jitted runner keyed
weakly on the objective. The contract pinned here:

* a second round on the same objective compiles nothing;
* the jitted round returns exactly what the un-jitted :func:`lbfgs_to_gtol` returns.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import test_util as jtu

from tengri.inference.backends import map_dispatch
from tengri.inference.lbfgs_driver import lbfgs_to_gtol

pytestmark = pytest.mark.contract


def _make_loss():
    """A fresh loss object with a fixed, non-trivial body (200-D, positive definite)."""
    curvature = jnp.diag(jnp.arange(1.0, 201.0))

    def loss(vec):
        return 0.5 * vec @ curvature @ vec + jnp.sum(jnp.sin(vec))

    return loss


def test_second_round_on_the_same_objective_compiles_nothing():
    loss = _make_loss()
    run = map_dispatch._compiled_round(loss)
    x = jnp.full(200, 0.3)
    run(x, n_steps=20, gtol=1e-6)
    with jtu.count_jit_compilation_cache_miss() as misses:
        run(x, n_steps=20, gtol=1e-6)
    assert misses() == 0, f"second continuation round recompiled {misses()} programs"


def test_compiled_round_matches_the_uncompiled_round_exactly():
    loss = _make_loss()
    x = jnp.full(200, 0.3)
    ref_x, ref_value, ref_trace = lbfgs_to_gtol(loss, x, n_steps=30, gtol=1e-6)
    got_x, got_value, got_trace = map_dispatch._compiled_round(loss)(x, n_steps=30, gtol=1e-6)
    np.testing.assert_array_equal(np.asarray(got_x), np.asarray(ref_x))
    np.testing.assert_array_equal(np.asarray(got_value), np.asarray(ref_value))
    np.testing.assert_array_equal(np.asarray(got_trace), np.asarray(ref_trace))
