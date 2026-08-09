# SPDX-License-Identifier: BSD-3-Clause
"""A lazy singleton must not let its first caller decide what it caches.

``test_cue_nlr_grammar.py`` failed on CI with::

    jax.errors.UnexpectedTracerError: ... float32[16,12] wrapped in a
    DynamicJaxprTracer to escape the scope of the transformation.
    The leaked intermediate value was created on line cue.py:428
    (load_cue_weights).

and passed locally. Same code, opposite result, because the two run tests in a
different order.

The mechanism: **any** ``jnp`` operation executed while a trace is active
returns a tracer bound to that trace, however concrete its inputs are.
``load_cue_weights`` builds its arrays with ``jnp.stack`` — 16 line
sub-networks x 12 parameters is the leaked ``float32[16,12]`` — and the result
is cached, in ``CueBackend`` and then in the ``_CUE_AGN_BACKEND`` module
global. So whichever caller runs first decides whether the cache holds real
arrays or tracers for the rest of the process. If that caller is inside
``jax.jit``, every later reader dies, with a traceback naming the loader rather
than whoever poisoned it.

Reproduced deterministically before fixing::

    batched_param_shifts type : DynamicJaxprTracer
    shape/dtype               : (16, 12) float32
    VERDICT: LEAK — UnexpectedTracerError

Three of the four cached AGN backends reach ``jnp`` during construction — Cue
via ``load_cue_weights``, and Feltre / Synthesizer-NLR via ``jnp.sort``, which
the BLR backend inherits. Only Cue had been caught, and only because a test
order happened to expose it. The tests below pin the **rule** at the cache
boundary rather than the one instance that got noticed: a first call from
inside a trace must still cache concrete arrays.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug

_WEIGHTS = os.path.join("data", "cue_weights.npz")
_needs_weights = pytest.mark.skipif(
    not os.path.exists(_WEIGHTS), reason="data/cue_weights.npz absent (data-gated)"
)


def _load_inside_a_trace(load):
    """Call ``load`` while a jit trace is active and hand back what it cached."""
    holder = {}

    @jax.jit
    def _traced(x):
        holder["value"] = load()
        return x * 2.0

    _traced(jnp.array(1.0))
    return holder["value"]


@_needs_weights
def test_cue_weights_loaded_inside_a_trace_are_still_concrete():
    """The exact shape that leaked, pinned."""
    from tengri.components.nebular.cue import load_cue_weights

    weights = _load_inside_a_trace(lambda: load_cue_weights(_WEIGHTS))
    arr = weights.batched_param_shifts

    assert arr.shape == (16, 12), f"guarding the wrong array; got {arr.shape}"
    assert not isinstance(arr, jax.core.Tracer), (
        "load_cue_weights cached a tracer; the next reader outside this trace "
        "will fail with UnexpectedTracerError"
    )


@_needs_weights
def test_the_cached_weights_are_usable_after_the_trace_exits():
    """The symptom, as a user meets it: a later read, far from the cause."""
    from tengri.components.nebular.cue import load_cue_weights

    weights = _load_inside_a_trace(lambda: load_cue_weights(_WEIGHTS))
    total = float(jnp.sum(weights.batched_param_shifts))
    assert total == total, "sum is NaN — the array did not survive the trace"


@_needs_weights
def test_the_cached_agn_backend_survives_being_warmed_inside_a_trace():
    """The cache boundary itself, which is where the guard lives.

    ``get_cue_agn_backend`` is the lazy singleton the forward model actually
    calls; guarding ``load_cue_weights`` alone would leave the other three
    cached backends relying on call order.
    """
    import tengri.components.agn.nlr_cloudy as nlr_cloudy

    saved = nlr_cloudy._CUE_AGN_BACKEND
    nlr_cloudy._CUE_AGN_BACKEND = None  # force this test to be the first caller
    try:
        backend = _load_inside_a_trace(nlr_cloudy.get_cue_agn_backend)
        arr = backend.weights.batched_param_shifts
        assert not isinstance(arr, jax.core.Tracer), (
            "the module-level backend cache was populated with tracers"
        )
        assert float(jnp.sum(arr)) is not None
    finally:
        nlr_cloudy._CUE_AGN_BACKEND = saved


def test_the_guard_is_applied_at_every_cached_accessor():
    """Pin the rule, not the one instance that happened to be caught.

    A future accessor added beside these without the guard is the same bug
    again, and would be found the same way: by a test in an unrelated file
    failing on CI but not locally.

    This one is structural and would stay green if ``_eager_construction``
    itself were neutered — verified: with the guard stubbed to a
    ``nullcontext``, the three behavioral tests above fail and this one passes.
    That is the intended division. This catches *a new accessor that forgot the
    guard*; those catch *a guard that stopped working*. Neither covers the
    other, which is why both are here.
    """
    import inspect

    import tengri.components.agn.nlr_cloudy as nlr_cloudy

    accessors = [
        "get_cue_agn_backend",
        "get_feltre_backend",
        "get_synthesizer_nlr_backend",
        "get_synthesizer_blr_backend",
    ]
    missing = [
        name
        for name in accessors
        if "_eager_construction" not in inspect.getsource(getattr(nlr_cloudy, name))
    ]
    assert not missing, (
        f"cached backend accessor(s) construct without _eager_construction(): {missing}. "
        "Whichever caller runs first will decide whether the cache holds arrays or tracers."
    )
