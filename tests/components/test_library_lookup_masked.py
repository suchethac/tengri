# SPDX-License-Identifier: BSD-3-Clause
"""Masked PCHIP lookup over populated cells (#2861, #2575).

``library_lookup(..., population_mask=...)`` reads a three-axis grid through its
populated nodes only: node-exact at populated cells, the tangents of the populated
subset, a hold at the populated hull, and no NaN where a row of the mask is empty.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from tengri.utils.grid_interp import library_lookup

pytestmark = pytest.mark.contract

_V = np.linspace(0.0, 1.0, 4)
_B = np.linspace(0.0, 1.0, 5)
_N = np.linspace(0.0, 1.0, 6)
_AXES = (jnp.asarray(_V), jnp.asarray(_B), jnp.asarray(_N))
_NO_EXTRAP = (False, False, False)


_JIT_CACHE: dict = {}


def _lookup_jit(mask, axes=_AXES, extrapolate=_NO_EXTRAP):
    """Jitted masked lookup over ``(grid, point)`` for one fixed mask (keeps tests fast)."""
    key = (mask.tobytes(), mask.shape, tuple(np.asarray(a).tobytes() for a in axes))
    if key not in _JIT_CACHE:
        mask_j = jnp.asarray(mask)

        @jax.jit
        def run(grid, point):
            return library_lookup(
                grid, axes, point, extrapolate=extrapolate, population_mask=mask_j
            )

        _JIT_CACHE[key] = run
    return _JIT_CACHE[key]


def _holey_mask():
    """Mask of shape (n_density, n_b) with gaps along density and B, one empty B row."""
    mask = np.ones((len(_N), len(_B)))
    mask[2:4, 1] = 0.0  # density gap in B row 1
    mask[:, 3] = 0.0  # B row 3 entirely unpopulated
    mask[0, 2] = 0.0  # a single hole at the lowest density
    return mask


def _grid(seed=2861):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(len(_V), len(_B), len(_N), 2))
    return jnp.asarray(values)


def test_node_exact_at_populated_cells():
    mask = _holey_mask()
    grid = _grid()
    run = _lookup_jit(mask)
    populated = mask.T > 0
    for i_v, v in enumerate(_V):
        for j_b, b in enumerate(_B):
            for k_n, n in enumerate(_N):
                if not populated[j_b, k_n]:
                    continue
                got = run(grid, (v, b, n))
                np.testing.assert_allclose(
                    np.asarray(got),
                    np.asarray(grid[i_v, j_b, k_n]),
                    rtol=1e-12,
                    atol=1e-12,
                    err_msg=f"node (v={i_v}, b={j_b}, n={k_n})",
                )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_gradient_matches_finite_difference_on_each_axis(axis):
    mask = jnp.asarray(_holey_mask())
    grid = _grid()
    point = np.array([0.37, 0.52, 0.45])  # off-node, between populated brackets

    run = _lookup_jit(np.asarray(mask))

    def f(p):
        return run(grid, tuple(p[i] for i in range(3)))[0]

    f = jax.jit(f)
    grad = np.asarray(jax.jit(jax.grad(f))(jnp.asarray(point)))
    assert np.all(np.isfinite(grad))
    step = 1e-6
    up, down = point.copy(), point.copy()
    up[axis] += step
    down[axis] -= step
    fd = (float(f(jnp.asarray(up))) - float(f(jnp.asarray(down)))) / (2.0 * step)
    assert grad[axis] != 0.0, f"gradient along axis {axis} is identically zero"
    assert grad[axis] == pytest.approx(fd, rel=1e-5, abs=1e-8)


def test_full_mask_equals_unmasked_pchip():
    grid = _grid()
    full = jnp.ones((len(_N), len(_B)))
    rng = np.random.default_rng(7)
    # The masked read reduces (velocity, density, B). PCHIP tangents are nonlinear, so the
    # reference is the unmasked read with the same axis order (its grid transposed to
    # (v, density, B)); the default (v, B, density) order differs by O(1) on random data.
    axes_vnb = (_AXES[0], _AXES[2], _AXES[1])
    grid_vnb = jnp.transpose(grid, (0, 2, 1, 3))
    run_masked = _lookup_jit(np.asarray(full))
    run_plain = jax.jit(lambda g, p: library_lookup(g, axes_vnb, p, extrapolate=_NO_EXTRAP))
    for _ in range(12):
        point = tuple(jnp.asarray(rng.uniform(0.0, 1.0)) for _ in range(3))
        masked = run_masked(grid, point)
        plain = run_plain(grid_vnb, (point[0], point[2], point[1]))
        np.testing.assert_allclose(np.asarray(masked), np.asarray(plain), rtol=1e-11, atol=1e-12)


def test_density_read_is_pchip_of_populated_subset():
    """1-D check: between populated brackets the value is PCHIP of the populated subset.

    The grid is constant in v and B, so the three-axis read reduces to one density
    read. The subset is {0, 1, 4, 5} (nodes 2 and 3 masked); the expected value is
    the Fritsch-Carlson cubic through those four nodes, which scipy's PchipInterpolator
    builds from the same populated subset.
    """
    subset_idx = np.array([0, 1, 4, 5])
    node_vals = np.array([0.0, 1.0, 1.6, 4.0])  # monotone increasing on the subset
    column = np.zeros(len(_N))
    column[subset_idx] = node_vals
    mask_n = np.zeros(len(_N))
    mask_n[subset_idx] = 1.0
    mask = jnp.asarray(np.tile(mask_n[:, None], (1, len(_B))))
    grid = jnp.asarray(
        np.broadcast_to(column[None, None, :, None], (len(_V), len(_B), len(_N), 1))
    )
    reference = PchipInterpolator(_N[subset_idx], node_vals)
    run = _lookup_jit(np.asarray(mask))
    queries = np.linspace(_N[0], _N[-1], 37)
    got = np.array([float(run(grid, (0.3, 0.6, n))[0]) for n in queries])
    np.testing.assert_allclose(got, reference(queries), rtol=1e-12, atol=1e-12)
    # Monotone data gives a monotone read: no overshoot across the masked gap.
    gap = np.linspace(_N[1], _N[4], 50)
    values = np.array([float(run(grid, (0.3, 0.6, n))[0]) for n in gap])
    assert np.all(np.diff(values) >= -1e-12)


def test_empty_row_gives_no_nan_and_matches_dropped_row():
    """A B row with no populated cell carries no weight; the result is the same as if
    that row were absent from the grid, and is finite."""
    grid = _grid()
    mask = _holey_mask()
    point = (0.4, 0.55, 0.45)
    got = _lookup_jit(mask)(grid, point)
    assert bool(jnp.all(jnp.isfinite(got)))
    keep = [0, 1, 2, 4]
    axes_b = (jnp.asarray(_V), jnp.asarray(_B[keep]), jnp.asarray(_N))
    dropped = _lookup_jit(mask[:, keep], axes_b)(grid[:, keep], point)
    np.testing.assert_allclose(np.asarray(got), np.asarray(dropped), rtol=1e-10, atol=1e-12)


def test_all_masked_is_nan():
    grid = _grid()
    got = _lookup_jit(np.zeros((len(_N), len(_B))))(grid, (0.4, 0.3, 0.5))
    assert bool(jnp.all(jnp.isnan(got)))


def test_masked_axis_extrapolation_rejected():
    with pytest.raises(ValueError, match="populated hull"):
        library_lookup(
            _grid(),
            _AXES,
            (0.4, 0.3, 0.5),
            extrapolate=(True, True, True),
            population_mask=jnp.asarray(_holey_mask()),
        )


def test_mask_shape_checked():
    with pytest.raises(ValueError, match="population_mask shape"):
        library_lookup(
            _grid(),
            _AXES,
            (0.4, 0.3, 0.5),
            extrapolate=_NO_EXTRAP,
            population_mask=jnp.ones((len(_B), len(_N))),
        )
