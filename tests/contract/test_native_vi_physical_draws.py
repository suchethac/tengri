# SPDX-License-Identifier: BSD-3-Clause
"""The native VI physical-space draw map matches the per-draw loop it replaces (#715).

The loop unflattened and converted one draw at a time in Python. The chunked
jitted map must reproduce it draw for draw, across chunk boundaries, and must
stay on the jitted path (a silent eager fallback would hide a lost speedup).
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from tengri.inference.backends.vi.native import _physical_draws

pytestmark = pytest.mark.contract


class _StubFitter:
    """Nonlinear elementwise map standing in for ``Fitter._to_physical``."""

    def _to_physical(self, p):
        return {"a": jnp.exp(p["x"]) * 2.0, "b": p["y"] + p["x"] ** 2}


def _loop_reference(fitter, unflatten, converged_dict, residuals_flat):
    rows = {}
    for i in range(residuals_flat.shape[0]):
        res = unflatten(residuals_flat[i])
        combined = {k: converged_dict[k] + res[k] for k in converged_dict}
        for k, v in fitter._to_physical(combined).items():
            rows.setdefault(k, []).append(v)
    return {k: jnp.stack(v) for k, v in rows.items()}


def _problem(n_draws, seed=0):
    init = {"x": jnp.zeros(()), "y": jnp.zeros(())}
    flat0, unflatten = ravel_pytree(init)
    converged = {"x": jnp.asarray(0.3), "y": jnp.asarray(-0.2)}
    residuals = jax.random.normal(jax.random.PRNGKey(seed), (n_draws, flat0.shape[0]))
    return unflatten, converged, residuals


def test_matches_per_draw_loop_across_chunk_boundaries():
    # 300 draws = one full 256-draw chunk plus a ragged 44-draw tail.
    unflatten, converged, residuals = _problem(300)
    fitter = _StubFitter()
    got = _physical_draws(fitter, unflatten, converged, residuals)
    want = _loop_reference(fitter, unflatten, converged, residuals)
    assert set(got) == set(want)
    for k in want:
        assert got[k].shape == (300,)
        np.testing.assert_allclose(np.asarray(got[k]), np.asarray(want[k]), rtol=1e-12)


def test_stays_on_jitted_path():
    unflatten, converged, residuals = _problem(40)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _physical_draws(_StubFitter(), unflatten, converged, residuals)
    assert not [w for w in caught if "cannot be traced" in str(w.message)]
