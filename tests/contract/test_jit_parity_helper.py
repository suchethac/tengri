# SPDX-License-Identifier: BSD-3-Clause
"""The parity helper must fail when parity is violated.

``assert_jit_matches_eager`` is called by 96 tests across 64 files. If it could
not fail, those 96 tests would be exactly as vacuous as the
``chex.assert_tree_all_finite``-only checks they replaced — worse, actually,
since they would *look* like correctness tests.

So the helper is exercised here against a function that genuinely disagrees
with itself under ``jit``, and the disagreement is produced the way the real
bug produces it: a Python value read at trace time and frozen into the compiled
graph. ``jax.jit`` traces once, so a counter incremented inside the traced
function advances on the eager call and then never again.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from tests._jit_parity import (
    _PEAK_RELATIVE_FLOOR,
    _noise_floor,
    assert_jit_matches_eager,
    assert_vmap_matches_loop,
)

pytestmark = pytest.mark.contract


class TestItPassesWhenItShould:
    def test_a_pure_kernel_agrees(self):
        """The overwhelmingly common case: jit changes nothing observable."""
        out = assert_jit_matches_eager(lambda x: jnp.sin(x) * 2.0, jnp.linspace(0.0, 3.0, 16))
        chex.assert_shape(out, (16,))

    def test_the_jitted_result_is_returned(self):
        """Callers keep their existing assertions on the returned value, so
        returning ``None`` would silently defuse every converted test."""
        x = jnp.arange(4.0)
        out = assert_jit_matches_eager(lambda a: a + 1.0, x)
        assert out is not None
        np.testing.assert_allclose(np.asarray(out), np.arange(4.0) + 1.0)

    def test_keyword_arguments_reach_both_calls(self):
        out = assert_jit_matches_eager(lambda a, *, scale: a * scale, jnp.ones(3), scale=2.5)
        np.testing.assert_allclose(np.asarray(out), np.full(3, 2.5))


class TestItFailsWhenItShould:
    """Each case is a real JAX failure mode, not a synthetic mismatch."""

    def test_catches_a_python_value_frozen_at_trace_time(self):
        """The canonical retrace bug: state read during tracing is baked in.

        Eager sees ``n=1``; the traced graph captured ``n=2`` and returns it
        forever. A finiteness assertion passes on both.
        """
        calls = {"n": 0}

        def leaky(x):
            calls["n"] += 1
            return x + calls["n"]

        with pytest.raises(AssertionError, match="assert_trees_all_close"):
            assert_jit_matches_eager(leaky, jnp.zeros(3))

    def test_the_bad_kernel_would_have_passed_a_finiteness_check(self):
        """Proves the upgrade is not cosmetic: the assertion these 96 tests
        used to make is green on the very kernel the new one rejects."""
        calls = {"n": 0}

        def leaky(x):
            calls["n"] += 1
            return x + calls["n"]

        import jax

        chex.assert_tree_all_finite(jax.jit(leaky)(jnp.zeros(3)))  # old check: passes

    def test_atol_is_actually_plumbed_through(self):
        """A tolerance argument that never reached chex would make every
        caller's override silently inert. The leaky kernel differs by exactly
        1.0, so a tolerance either side of that pins the wiring."""
        def leaky_factory():
            calls = {"n": 0}

            def leaky(x):
                calls["n"] += 1
                return x + calls["n"]

            return leaky

        with pytest.raises(AssertionError):
            assert_jit_matches_eager(leaky_factory(), jnp.zeros(3), atol=0.5)
        assert_jit_matches_eager(leaky_factory(), jnp.zeros(3), atol=2.0)


class TestTheNoiseFloorIsScaleAware:
    """A fixed absolute tolerance would be wrong at both ends of this
    codebase's dynamic range; the floor tracks the data instead."""

    @pytest.mark.parametrize("peak", [1e-30, 1.0, 1e44])
    def test_floor_tracks_the_peak(self, peak):
        floor = _noise_floor(jnp.array([0.0, peak / 2, peak]))
        assert floor == pytest.approx(_PEAK_RELATIVE_FLOOR * peak, rel=1e-9)

    def test_an_all_zero_output_demands_exact_agreement(self):
        """No peak means no licence to be approximate."""
        assert _noise_floor(jnp.zeros(8)) == 0.0

    def test_non_finite_entries_do_not_inflate_the_floor(self):
        """One inf in the output must not license unlimited disagreement
        everywhere else."""
        floor = _noise_floor(jnp.array([1.0, jnp.inf, jnp.nan]))
        assert floor == pytest.approx(_PEAK_RELATIVE_FLOOR)

    def test_integer_leaves_are_ignored(self):
        """Index/count outputs are exact; they must not set a float tolerance."""
        assert _noise_floor(jnp.array([3, 900], dtype=jnp.int32)) == 0.0


class TestVmapParity:
    def test_agrees_with_the_loop(self):
        batch = jnp.arange(12.0).reshape(4, 3)
        out = assert_vmap_matches_loop(lambda row: jnp.sum(row**2), batch)
        chex.assert_shape(out, (4,))

    def test_catches_state_captured_once_by_the_batched_trace(self):
        """``vmap`` traces the body **once** for the whole batch; a loop runs
        it per row. Any per-call Python state therefore advances N times in the
        loop and once under ``vmap`` — the batched output is still finite and
        still shape (N,), so only the looped reference distinguishes them."""
        calls = {"n": 0}

        def stateful(row):
            calls["n"] += 1
            return jnp.sum(row) + calls["n"]

        with pytest.raises(AssertionError, match="assert_trees_all_close"):
            assert_vmap_matches_loop(stateful, jnp.arange(12.0).reshape(4, 3))
