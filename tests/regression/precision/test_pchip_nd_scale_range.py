# SPDX-License-Identifier: BSD-3-Clause
r"""The N-dimensional PCHIP read is scale-free over the whole exponent range.

The window read divides each column by a power of two near its peak, so the harmonic-mean
product of two secants can neither overflow (a table of ~1e19 per-Lsun values squares past
float32's range) nor underflow (``1e-20`` squares to zero in float32). The scale itself must
stay finite: a peak within a factor of two of the largest float used to push the power of
two to infinity and return 0 or NaN where the unscaled read was finite.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.utils.grid_interp import interp_nd_pchip

pytestmark = pytest.mark.regression_bug

DIMS = (5, 4, 3)
POINT = (0.43, 0.37, 0.61)
F64_PEAKS = [1e-290, 1e-200, 1e-100, 1e-20, 1.0, 1e20, 1e100, 1e200, 1e300, 5e307, 1.7e308]
#: The derivative with respect to a coordinate is ~ peak / (node spacing), up to ~8 peak here,
#: so above these peaks it legitimately exceeds the largest float.
_COORD_GRAD_CEILING_F64 = 1e307
_COORD_GRAD_CEILING_F32 = 1e37
#: Below ~1e-30 the float32 coordinate derivative is built from products under the smallest
#: normal number (1.2e-38) and the hardware flushes them; the values themselves hold to 1e-34.
_COORD_GRAD_FLOOR_F32 = 1e-30
F32_PEAKS = [1e-34, 1e-30, 1e-20, 1.0, 1e19, 1e30, 1e38, 1.7e38, 3.3e38]


def _base(kind):
    rng = np.random.default_rng(3)
    if kind == "positive":
        grid = np.exp(rng.normal(size=DIMS))
    elif kind == "mixed_sign":
        grid = rng.normal(size=DIMS)
    elif kind == "zeros":
        grid = np.zeros(DIMS)
    else:  # monotone ramp with flat runs: slopes of both signs and exact zeros
        grid = np.cumsum(rng.integers(-1, 2, size=DIMS).astype(float), axis=0)
    peak = np.max(np.abs(grid))
    return grid / peak if peak > 0 else grid


def _read(grid, dtype):
    axes = tuple(jnp.asarray(np.linspace(0.0, 1.0, n), dtype) for n in DIMS)
    point = tuple(jnp.asarray(v, dtype) for v in POINT)

    def value(g, *p):
        return interp_nd_pchip(g, axes, p)

    g = jnp.asarray(grid, dtype)
    return (
        float(value(g, *point)),
        np.asarray(jax.grad(value, argnums=(1, 2, 3))(g, *point), dtype=np.float64),
        np.asarray(jax.grad(value)(g, *point), dtype=np.float64),
    )


def _reference(kind):
    with jax.enable_x64(True):
        return _read(_base(kind), jnp.float64)


@pytest.mark.parametrize("kind", ["positive", "mixed_sign", "zeros", "ramp"])
@pytest.mark.parametrize("peak", F64_PEAKS)
def test_float64_read_is_scale_free_to_the_largest_float(kind, peak):
    ref_value, ref_coord, ref_grid = _reference(kind)
    with jax.enable_x64(True):
        value, coord, grid_grad = _read(_base(kind) * peak, jnp.float64)
    assert np.isfinite(value) and np.all(np.isfinite(grid_grad))
    assert abs(value / peak - ref_value) <= 1e-12 * max(abs(ref_value), 1e-300)
    if peak <= _COORD_GRAD_CEILING_F64:
        assert np.all(np.isfinite(coord))
        assert np.max(np.abs(coord / peak - ref_coord)) <= 1e-10 * max(
            np.max(np.abs(ref_coord)), 1e-300
        )
    assert np.max(np.abs(grid_grad - ref_grid)) <= 1e-12


@pytest.mark.parametrize("kind", ["positive", "mixed_sign", "zeros", "ramp"])
@pytest.mark.parametrize("peak", F32_PEAKS)
def test_float32_read_is_scale_free_to_the_largest_float(kind, peak):
    ref_value, ref_coord, ref_grid = _reference(kind)
    with jax.enable_x64(False):
        value, coord, grid_grad = _read(_base(kind) * peak, jnp.float32)
    assert np.isfinite(value) and np.all(np.isfinite(grid_grad))
    assert abs(value / peak - ref_value) <= 2e-5 * max(abs(ref_value), 1e-30)
    if _COORD_GRAD_FLOOR_F32 <= peak <= _COORD_GRAD_CEILING_F32:
        assert np.all(np.isfinite(coord))
        assert np.max(np.abs(coord / peak - ref_coord)) <= 2e-4 * max(
            np.max(np.abs(ref_coord)), 1e-30
        )
    assert np.max(np.abs(grid_grad - ref_grid)) <= 2e-5


@pytest.mark.parametrize("peak", [1e-40, 1e-44])
def test_float32_subnormal_tables_stay_finite(peak):
    with jax.enable_x64(False):
        value, coord, grid_grad = _read(_base("positive") * peak, jnp.float32)
    assert np.isfinite(value) and np.all(np.isfinite(coord)) and np.all(np.isfinite(grid_grad))


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_a_non_finite_node_gives_a_non_finite_read(bad):
    grid = _base("positive")
    grid[2, 1, 1] = bad
    with jax.enable_x64(True):
        value, _, _ = _read(grid, jnp.float64)
    assert not np.isfinite(value)
