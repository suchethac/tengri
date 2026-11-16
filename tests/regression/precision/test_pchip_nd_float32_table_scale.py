# SPDX-License-Identifier: BSD-3-Clause
r"""The N-dimensional PCHIP read is scale-free in float32.

The monotone-cubic tangent test multiplies two secants. A template table of per-solar-
luminosity values near :math:`10^{19}` (the SKIRTOR band table) squares past float32's
:math:`3.4\times10^{38}`, so the reverse pass returned NaN for every coordinate of the
leading axes, and the AGN fits had a NaN float32 gradient and Hessian. The cubic is
positively homogeneous in the node values, so the read normalizes each window column by
its own maximum and scales back.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.utils.grid_interp import interp_nd_pchip

pytestmark = pytest.mark.regression_bug

DIMS = (5, 4, 4, 3)
POINT = (0.43, 0.37, 0.52, 0.61)


def _derivatives(scale: float, dtype):
    rng = np.random.default_rng(0)
    base = np.exp(rng.normal(size=DIMS))
    axes = tuple(jnp.asarray(np.linspace(0.0, 1.0, n), dtype) for n in DIMS)
    point = tuple(jnp.asarray(v, dtype) for v in POINT)

    def read(grid, *p):
        return interp_nd_pchip(grid, axes, p)

    grid = jnp.asarray(base * scale, dtype)
    value = read(grid, *point)
    d_coord = jax.grad(read, argnums=(1, 2, 3, 4))(grid, *point)
    d_grid = jax.grad(read)(grid, *point)
    return float(value) / scale, np.asarray(d_coord) / scale, np.asarray(d_grid)


@pytest.mark.parametrize("scale", [1e-30, 1e-20, 1.0, 1e19, 1e25, 1e30])
def test_float32_read_and_gradients_match_float64_at_every_table_scale(scale):
    with jax.enable_x64(True):
        v64, c64, g64 = _derivatives(scale, jnp.float64)
    with jax.enable_x64(False):
        v32, c32, g32 = _derivatives(scale, jnp.float32)
    assert np.isfinite(v32) and np.all(np.isfinite(c32)) and np.all(np.isfinite(g32))
    assert v32 != 0.0 and np.any(c32 != 0.0) and np.any(g32 != 0.0)
    assert abs(v32 - v64) <= 1e-5 * abs(v64)
    assert np.max(np.abs(c32 - c64)) <= 1e-4 * np.max(np.abs(c64))
    assert np.max(np.abs(g32 - g64)) <= 1e-4 * np.max(np.abs(g64))
