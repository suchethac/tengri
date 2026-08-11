# SPDX-License-Identifier: BSD-3-Clause
"""The Cue weight tables must not hold trace-scoped arrays (#1631).

``load_cue_weights`` builds its arrays with ``jnp.array``/``jnp.stack``/
``jnp.pad``. Inside a jit trace every one of those returns a
``DynamicJaxprTracer`` -- tracing is symbolic, so an op on a *constant* still
produces a tracer -- and ``nlr_cloudy.get_cue_agn_backend`` memoizes the
resulting backend in a module-level global. First caller inside a trace poisons
the global for every later trace::

    FAILED tests/contract/test_cue_nlr_grammar.py::test_cue_nlr_lines_appear_and_logU_is_not_a_noop
      jax.errors.UnexpectedTracerError: ... float32[16,12] wrapped in a
      DynamicJaxprTracer to escape the scope of the transformation

``float32[16,12]`` names the culprit exactly: ``_LINE_NAMES`` has 16 line
sub-networks and Cue takes 12 parameters, so it is
``batched_param_shifts = jnp.stack([n.param_shift for n in nets])``.

``load_cue_weights``'s own docstring already said **"JIT-compatible: no --
performs file I/O and array padding at load time"**. Nothing enforced it, and
the AGN NLR path reaches it lazily from inside a trace via
``nlr_cue_block -> compute_nlr_sed_cue -> get_cue_agn_backend``.

Why it hid: the test passes *alone*. It needs an earlier test in the same
process to build the backend inside some other trace first, so it surfaced as a
shard-dependent failure -- red in one contract shard, green in the other.

The fix is #1462's, for the identical class (``load_grahsp_templates``): cache
**NumPy**. A NumPy array cannot be a tracer under any trace, so the cache is
trace-independent by construction rather than by discipline, and consumers
already wrap with ``jnp.asarray`` at use.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

_WEIGHTS = os.path.join("data", "cue_weights.npz")
_HAS_WEIGHTS = os.path.exists(_WEIGHTS)

#: Every ``CueWeights`` field that is a plain array (the tuple-of-array fields
#: are checked separately below).
_ARRAY_FIELDS = (
    "sorted_line_wav",
    "nn_line_wav",
    "line_old_idx",
    "cont_wav",
    "batched_param_shifts",
    "batched_param_scales",
    "batched_W_out",
    "batched_b_out",
    "batched_pca_scale",
    "batched_pca_shift",
    "batched_pca_comp",
    "batched_pca_mean",
)


@pytest.mark.skipif(not _HAS_WEIGHTS, reason="data/cue_weights.npz absent (data-gated)")
def test_cue_weight_arrays_are_numpy():
    """The structural guarantee: NumPy cannot be trace-scoped.

    Asserted on the type rather than by reproducing the leak, so the guard
    holds even if the surrounding jit structure changes (#1462's reasoning).
    """
    from tengri.components.nebular.cue import load_cue_weights

    weights = load_cue_weights(_WEIGHTS)

    checked = 0
    for name in _ARRAY_FIELDS:
        arr = getattr(weights, name)
        assert isinstance(arr, np.ndarray), f"{name} is {type(arr).__name__}, expected np.ndarray"
        assert not isinstance(arr, jax.Array), f"{name} is a JAX array; it can be trace-scoped"
        checked += 1
    assert checked == len(_ARRAY_FIELDS)

    # The batched hidden layers are tuples of arrays -- the same hazard, one
    # level down, and they are what the emulator actually multiplies.
    for name in ("batched_W_hidden", "batched_b_hidden"):
        stack = getattr(weights, name)
        assert stack, f"{name} is empty"
        for i, arr in enumerate(stack):
            assert isinstance(arr, np.ndarray), f"{name}[{i}] is {type(arr).__name__}"

    # And one level down again: the per-line sub-networks.
    sub = weights.line_nets[0]
    assert isinstance(sub.param_shift, np.ndarray)
    assert isinstance(sub.W[0], np.ndarray)


@pytest.mark.skipif(not _HAS_WEIGHTS, reason="data/cue_weights.npz absent (data-gated)")
def test_agn_cue_backend_survives_first_construction_inside_a_trace(monkeypatch):
    """The behavior #1631 reported: build the global inside trace 1, use it in trace 2.

    ``get_cue_agn_backend`` caches into a module global, so the arms must run in
    that order and the global must start empty -- otherwise an earlier test in
    the process has already populated it eagerly and this proves nothing.
    """
    from tengri.components.agn import nlr_cloudy

    monkeypatch.setattr(nlr_cloudy, "_CUE_AGN_BACKEND", None)

    # The two arms must be DISTINCT functions. Wrapping one function in
    # ``jax.jit`` twice does not give two traces: the second call hits JAX's
    # cache keyed on the wrapped function and never retraces, so the stale
    # tracer is never touched and the test passes with the bug present. That
    # is exactly how the first version of this test was green before the fix.
    def _build(x):
        backend = nlr_cloudy.get_cue_agn_backend()
        return x * jnp.asarray(backend.weights.batched_param_shifts).sum()

    def _reuse(x):
        backend = nlr_cloudy.get_cue_agn_backend()
        return x * jnp.asarray(backend.weights.batched_param_scales).sum()

    first = jax.jit(_build)(1.0)  # trace 1: builds and caches the backend
    second = jax.jit(_reuse)(2.0)  # trace 2: reuses it -> leaks before the fix

    assert np.isfinite(float(first))
    assert np.isfinite(float(second))


@pytest.mark.skipif(not _HAS_WEIGHTS, reason="data/cue_weights.npz absent (data-gated)")
def test_weights_built_inside_a_trace_are_not_tracers(monkeypatch):
    """The mechanism, pinned directly: no trace may produce trace-scoped weights.

    Complements the two-trace test above, which can only observe the leak once
    it has already been cached. This one catches the moment of creation, so it
    stays meaningful even if ``get_cue_agn_backend`` stops memoizing.
    """
    from tengri.components.agn import nlr_cloudy

    monkeypatch.setattr(nlr_cloudy, "_CUE_AGN_BACKEND", None)
    observed = {}

    def _inspect(x):
        backend = nlr_cloudy.get_cue_agn_backend()
        arr = backend.weights.batched_param_shifts
        observed["is_tracer"] = isinstance(arr, jax.core.Tracer)
        observed["type"] = type(arr).__name__
        return x * 1.0

    jax.jit(_inspect)(1.0)

    assert observed["is_tracer"] is False, (
        f"Cue weights built inside a jit trace are {observed['type']}; a cached "
        "backend then hands trace-scoped arrays to the next trace (#1631)."
    )
