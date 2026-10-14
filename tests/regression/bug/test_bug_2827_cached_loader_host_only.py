# SPDX-License-Identifier: BSD-3-Clause
"""Regression for #2827: a module-level cache must not hold a JAX tracer.

``_load_relagn_disc_grid`` is ``functools.cache``-d. When its first call runs
under ``jax.jit``, any ``jnp.array`` it builds is a ``DynamicJaxprTracer``; the
cache then hands that tracer to every later model in the process, which raises
``UnexpectedTracerError``. The loader must cache host numpy arrays and convert
at the call site.

Marker: regression_bug
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

_WAVE = np.geomspace(1.0e3, 1.0e5, 96)


def _predict(lbol: float) -> np.ndarray:
    from tengri.components.agn.disc import load_relagn_default_grid, relagn_disc_from_grid

    def _run(log_lbol):
        return relagn_disc_from_grid(
            load_relagn_default_grid(),
            jnp.asarray(_WAVE),
            agn_log_lbol=log_lbol,
            agn_log_mbh=8.0,
            agn_log_mdot=-1.0,
            agn_astar=0.5,
            agn_cos_inc=0.8,
        )

    return np.asarray(jax.jit(_run)(lbol))


def test_relagn_grid_first_loaded_under_jit_serves_a_second_model():
    from tengri.components.agn import disc

    disc._load_relagn_disc_grid.cache_clear()
    first = _predict(11.5)  # first load happens inside a jit trace
    second = _predict(11.0)  # must not raise UnexpectedTracerError
    assert np.all(np.isfinite(first))
    assert np.all(np.isfinite(second))
    np.testing.assert_allclose(second, first * 10.0 ** (11.0 - 11.5), rtol=1e-6)


def test_nthcomp_template_cache_holds_host_arrays_not_tracers():
    from tengri.components.agn import _nthcomp

    if not _nthcomp._DEFAULT_TEMPLATE_PATH.exists():
        pytest.skip("nthcomp templates not built in this checkout")
    _nthcomp._load_nthcomp_templates_impl.cache_clear()
    # First load happens inside a jit trace; the cache must keep numpy only.
    jax.jit(lambda x: (_nthcomp._load_nthcomp_templates_impl(), x)[1])(0.0)
    cached = _nthcomp._load_nthcomp_templates_impl()
    assert cached[-1] is True
    for arr in cached[:5]:
        assert isinstance(arr, np.ndarray), type(arr)
