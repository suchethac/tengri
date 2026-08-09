# SPDX-License-Identifier: BSD-3-Clause
"""The Cue AGN-NLR backend must survive being built inside a JIT trace.

``get_cue_agn_backend`` caches a ``CueBackend`` in a module-level global. If the
first construction happens *inside* a trace, the instance captures
``DynamicJaxprTracer`` values — and because the global outlives the trace, every
later out-of-trace call raises ``UnexpectedTracerError``.

That makes correctness depend on execution order: whichever caller traces a
Cue-NLR model first decides whether the rest of the process works. It surfaced
as ``test_cue_nlr_grammar`` failing in CI only when a Cue-NLR model had been
traced earlier in the same worker, while passing in isolation.

Same defect class as the GRAHSP template cache (#1462).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytestmark = [pytest.mark.regression_bug]


def _reset_singleton():
    """Drop the cached backend so the next call reconstructs it."""
    from tengri.components.agn import nlr_cloudy

    nlr_cloudy._CUE_AGN_BACKEND = None


def test_backend_built_inside_a_trace_is_usable_afterwards():
    """Construct the backend under trace, then call it eagerly.

    The eager call is the assertion: a backend that captured tracers raises
    ``UnexpectedTracerError`` here rather than returning an array.
    """
    from tengri.components.agn.nlr_cloudy import compute_nlr_sed_cue, get_cue_agn_backend

    try:
        _reset_singleton()
        get_cue_agn_backend()
    except FileNotFoundError:
        pytest.skip("cue_weights.npz not available")

    _reset_singleton()
    wave = jnp.linspace(1000.0, 10000.0, 128)

    # First touch happens inside a trace — this is what poisoned the global.
    def _traced(l_bol):
        return compute_nlr_sed_cue(wave, l_disc_bol_erg=l_bol)

    jax.jit(_traced)(jnp.asarray(1e45))

    # Now use it outside any trace. Pre-fix this raised UnexpectedTracerError.
    eager = compute_nlr_sed_cue(wave, l_disc_bol_erg=1e45)
    assert jnp.all(jnp.isfinite(eager))
    assert float(jnp.max(eager)) > 0.0
