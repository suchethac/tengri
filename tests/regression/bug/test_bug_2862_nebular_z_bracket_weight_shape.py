# SPDX-License-Identifier: BSD-3-Clause
"""The nebular continuum z-table's bracket weight is a scalar, extrapolated or not.

#2862 routed ``nebular_band_z.z_bracket(..., extrapolate=True)`` through the shared
``_z_response.z_bracket`` so a redshift past the table keeps a nonzero derivative.
That helper returns the pair of node weights ``(1 - t, t)``, shape ``(2,)``, while the
nebular module's contract (and both of its blends, ``continuum_band_from_slab`` and
``nebular_subband_decomposition``) is the scalar upper weight ``t``. The pair reached
``(1.0 - weight) * bands[0] + weight * bands[1]`` and failed to broadcast against the
``(n_filter,)`` band rows, so every free-redshift Cue fit on the default LUT path
raised ``TypeError: mul got incompatible shapes`` at its first likelihood compile.

These tests pin the scalar contract on a synthetic table: inside the hull the weight
equals the non-extrapolating one, past the end node it continues the edge cell's line
(the #2862 intent), and the blended band keeps its ``(n_filter,)`` shape under ``jit``
and ``grad``.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.nebular.nebular_band_z import (
    continuum_band_from_slab,
    continuum_z_slab,
    z_bracket,
)

pytestmark = pytest.mark.regression_bug

#: ln(1 + z) at three table nodes, z = 0.1, 0.2, 0.3.
_LNZ = jnp.log1p(jnp.asarray([0.1, 0.2, 0.3]))
#: log10 band L_nu per Q_H at each node, (n_z, n_filter).
_LOG_TABLE = jnp.asarray(
    [
        [1.0, 2.0, 3.0, 4.0],
        [1.5, 2.5, 3.5, 4.5],
        [2.5, 3.5, 4.5, 5.5],
    ]
)
_KEEP = jnp.ones(_LOG_TABLE.shape, dtype=bool)
_LOG_NION = 0.0


def _expected_t(z, lo=0.2, hi=0.3):
    return (np.log1p(z) - np.log1p(lo)) / (np.log1p(hi) - np.log1p(lo))


@pytest.mark.parametrize("extrapolate", [False, True])
def test_the_bracket_weight_is_a_scalar_inside_the_hull(extrapolate):
    i, w = z_bracket(_LNZ, 0.25, extrapolate=extrapolate)
    chex.assert_shape(w, ())
    assert int(i) == 1
    np.testing.assert_allclose(float(w), _expected_t(0.25), rtol=1e-12)


def test_the_extrapolated_weight_continues_the_edge_cell_past_the_table():
    """Past the last node the weight exceeds 1 (the #2862 intent), still a scalar."""
    i_in, w_in = z_bracket(_LNZ, 0.33, extrapolate=False)
    i_out, w_out = z_bracket(_LNZ, 0.33, extrapolate=True)
    chex.assert_shape(w_out, ())
    assert int(i_in) == int(i_out) == 1
    assert float(w_in) == 1.0, "the clamped read holds the end node"
    np.testing.assert_allclose(float(w_out), _expected_t(0.33), rtol=1e-12)
    assert float(w_out) > 1.0


def _band(z):
    slab, lower, weight = continuum_z_slab(_LOG_TABLE, _LNZ, z)
    return continuum_band_from_slab(_LOG_NION, slab, _KEEP, lower, weight)


@pytest.mark.parametrize("z", [0.25, 0.33])
def test_the_blended_continuum_band_keeps_the_filter_shape_under_jit_and_grad(z):
    band = jax.jit(_band)(z)
    chex.assert_shape(band, (_LOG_TABLE.shape[1],))
    t = _expected_t(z)
    expected = (1.0 - t) * 10.0 ** np.asarray(_LOG_TABLE[1]) + t * 10.0 ** np.asarray(
        _LOG_TABLE[2]
    )
    np.testing.assert_allclose(np.asarray(band), expected, rtol=1e-12)

    dz = float(jax.grad(lambda zz: jnp.sum(_band(zz)))(jnp.asarray(z)))
    assert np.isfinite(dz), "the z gradient must be finite, inside the hull and past it"
    assert dz != 0.0, "the z gradient must survive, inside the hull and past it"
