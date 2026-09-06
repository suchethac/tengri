# SPDX-License-Identifier: BSD-3-Clause
r"""Regression: SKIRTOR templates thread through JIT as arrays, not a callable (#1198).

The monolithic SKIRTOR AGN threads its template grid into ``predict_observables_jit``
as a runtime input (``template_data['agn']['skirtor']``). It used to thread the
*interpolation closure* returned by ``create_skirtor_from_grid`` — a Python
function, which JAX cannot treat as a traced argument. Under the WavePrecomp
build this made ``predict_photometry`` raise::

    TypeError: ... value is of type <class 'function'> ... at path
    template_data['agn']['skirtor']

while the exact build silently baked the whole grid into the HLO as a constant
(large compile). Eager and JIT SED paths therefore disagreed structurally
(a working eager SED vs a crashing / constant-baked JIT SED).

The fix threads the template *arrays* (:class:`SKIRTORGrid`, a JAX pytree of
``grid``/``wave_grid``/``axes``/``edges``) and interpolates from them inside the
pure-JAX model function, so the data always threads (small compile) and eager
matches JIT bit-for-bit.

Data-free: uses only the committed SKIRTOR template grid (no SSP), so it runs in
CI — unlike the SEDModel-level jit-threading tests, which are gated on a
bare-stellar SSP and skip.

References
----------
.. [1] M. Stalevski et al., MNRAS, 420, 2756 (2012). arXiv:1109.1286.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

_skirtor = pytest.importorskip("tengri.components.agn.skirtor")

# Skip only if the committed SKIRTOR grid is genuinely absent.
try:
    _GRID = _skirtor._load_skirtor_default_grid()
except (FileNotFoundError, OSError) as exc:  # pragma: no cover - grid absent
    # Narrowed from `except Exception`: at module scope this skip takes the
    # whole file, so a changed loader signature or a broken import would have
    # reported as "grid not available" for every test here at once.
    pytest.skip(f"SKIRTOR grid not available: {exc}", allow_module_level=True)

_WAVE = jnp.geomspace(1.0e3, 1.0e7, 400)
_KW = dict(agn_log_lbol=44.0, agn_tau_skirtor=7.0, agn_oa_skirtor=40.0, agn_torus_frac=0.5)


def test_default_grid_is_pure_array_pytree():
    """The threaded template is a pytree of arrays — never a Python callable."""
    from tengri.components.agn.skirtor import SKIRTORGrid

    assert isinstance(_GRID, SKIRTORGrid)
    leaves = jax.tree_util.tree_leaves(_GRID)
    assert leaves, "grid pytree has no leaves"
    for leaf in leaves:
        assert not callable(leaf) or isinstance(leaf, (jnp.ndarray, np.ndarray)), (
            f"SKIRTOR grid leaf is a {type(leaf).__name__} — it must be array data so "
            "it can thread through jax.jit as a runtime input (not a baked closure)."
        )
        assert isinstance(leaf, (jnp.ndarray, np.ndarray)), f"non-array leaf {type(leaf)}"


def test_grid_threads_through_jit_as_argument():
    """Threading the grid as a jit ARGUMENT works and matches eager — the exact
    failure mode when a closure was threaded (a function cannot be a jit arg)."""
    from tengri.components.agn.skirtor import skirtor_sed

    def f(grid):
        return skirtor_sed(_WAVE, _template=grid, **_KW)

    eager = np.asarray(f(_GRID))
    jitted = np.asarray(jax.jit(f)(_GRID))
    assert np.all(np.isfinite(eager)) and np.any(eager > 0)
    # Signal-scaled atol: jit vs eager can differ by op-fusion round-off on the
    # torus UV noise floor, where atol=0 flakes (cf. #1195). The point of the
    # test is structural (arrays thread; no crash), not bit-exactness.
    np.testing.assert_allclose(jitted, eager, rtol=1e-8, atol=1e-12 * np.max(np.abs(eager)))


def test_threaded_grid_matches_default_load():
    """Interpolating from the threaded grid equals the auto-loaded default."""
    from tengri.components.agn.skirtor import skirtor_sed

    threaded = np.asarray(skirtor_sed(_WAVE, _template=_GRID, **_KW))
    default = np.asarray(skirtor_sed(_WAVE, _template=None, **_KW))
    np.testing.assert_allclose(threaded, default, rtol=1e-10, atol=0.0)


def test_legacy_callable_template_still_supported():
    """A legacy callable ``_template`` (the closure) still works — back-compat."""
    from tengri.components.agn.skirtor import _load_skirtor_default, skirtor_sed

    closure = _load_skirtor_default()
    assert callable(closure)
    via_closure = np.asarray(skirtor_sed(_WAVE, _template=closure, **_KW))
    via_grid = np.asarray(skirtor_sed(_WAVE, _template=_GRID, **_KW))
    np.testing.assert_allclose(via_closure, via_grid, rtol=1e-10, atol=0.0)
