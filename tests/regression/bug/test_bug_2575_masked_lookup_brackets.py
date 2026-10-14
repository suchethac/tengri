# SPDX-License-Identifier: BSD-3-Clause
r"""Masked lookups interpolate between populated brackets (#2575).

``interp_nd_triweight(..., population_mask=...)`` renormalized over the populated cells
by snapping to the nearest populated node. A query that straddles a masked gap then
returned a value from the wrong side of the gap, or NaN, although the populated nodes
either side of it bracket the query.

Owner ruling 2026-10-09: interpolation between populated brackets wins. Along each
masked axis a query between two populated nodes returns their linear interpolant; the
kernel's own weights are kept where the stencil reaches populated nodes only. A query
with populated cells on one side only holds the nearest populated node. NaN is returned
only when no cell is populated at all.

The grids below are linear along the masked axis, so the bracket answer is exact.
"""

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

from tengri.utils.grid_interp import edges_for_grid, interp_nd_triweight

#: Grid nodes, shape (n_v, n_b, n_density). Chosen so that a query between two populated
#: nodes falls inside a masked gap of the kernel stencil.
_V = jnp.linspace(0.0, 1.0, 4)
_B = jnp.linspace(0.0, 1.0, 5)
_N = jnp.linspace(0.0, 1.0, 6)
_AXES = (_V, _B, _N)
_EDGES = tuple(edges_for_grid(a) for a in _AXES)


def _grid(field, mask_nb):
    """Build a zero-filled (n_v, n_b, n_n, 1) grid of ``field(b, n)`` on populated cells.

    ``mask_nb`` is (n_b, n_n) here for readability; the interpolator takes
    (n_density, n_b), so the transpose is applied at the call site.
    """
    bb, nn = np.meshgrid(np.asarray(_B), np.asarray(_N), indexing="ij")
    vals = np.where(mask_nb > 0, field(bb, nn), 0.0)
    out = np.broadcast_to(vals, (len(_V), len(_B), len(_N)))[..., np.newaxis]
    return jnp.asarray(out)


def _interp(grid, mask_nb, point):
    return float(
        interp_nd_triweight(
            grid,
            _AXES,
            _EDGES,
            point,
            population_mask=jnp.asarray(np.asarray(mask_nb).T),
        )[0]
    )


def test_density_gap_gives_linear_bracket_value():
    """Density nodes 2 and 3 are masked everywhere; n = 0.45 sits between 0.2 and 0.8.

    The query is off-center so that a symmetric kernel average cannot hit the answer.

    The field is 2 n + 1, linear along density, so the bracket interpolant is exact:
    f(0.45) = 1.9. Snapping to the nearest populated node gives 1.4 or 2.6.
    """
    mask = np.ones((len(_B), len(_N)))
    mask[:, 2:4] = 0.0
    grid = _grid(lambda b, n: 2.0 * n + 1.0, mask)
    value = _interp(grid, mask, (0.4, 0.3, 0.45))
    assert value == pytest.approx(1.9, rel=1e-9)


def test_b_gap_gives_linear_bracket_value():
    """B nodes 2 and 3 are masked everywhere; b = 0.4 sits between 0.25 and 1.0.

    The field is 3 b + 0.5, linear along B, so the bracket interpolant at b = 0.4
    (between 0.25 and 1.0) gives 1.7.
    """
    mask = np.ones((len(_B), len(_N)))
    mask[2:4, :] = 0.0
    grid = _grid(lambda b, n: 3.0 * b + 0.5, mask)
    value = _interp(grid, mask, (0.4, 0.4, 0.5))
    assert value == pytest.approx(1.7, rel=1e-9)


def test_edge_hold_below_lowest_populated_density():
    """Density nodes 0..3 are masked, so a query at n = 0.1 has no populated node below.

    It holds the nearest populated node, n = 0.8, whose field value is 2.6.
    """
    mask = np.ones((len(_B), len(_N)))
    mask[:, 0:4] = 0.0
    grid = _grid(lambda b, n: 2.0 * n + 1.0, mask)
    value = _interp(grid, mask, (0.4, 0.3, 0.1))
    assert value == pytest.approx(2.6, rel=1e-9)


def test_edge_hold_above_highest_populated_density():
    """Density nodes 2..5 are masked, so a query at n = 0.9 holds n = 0.2 (value 1.4)."""
    mask = np.ones((len(_B), len(_N)))
    mask[:, 2:6] = 0.0
    grid = _grid(lambda b, n: 2.0 * n + 1.0, mask)
    value = _interp(grid, mask, (0.4, 0.3, 0.9))
    assert value == pytest.approx(1.4, rel=1e-9)


def test_all_masked_returns_nan():
    """NaN only when no cell is populated at all."""
    mask = np.zeros((len(_B), len(_N)))
    grid = _grid(lambda b, n: 2.0 * n + 1.0, mask)
    value = _interp(grid, mask, (0.4, 0.3, 0.5))
    assert np.isnan(value)


def test_fully_populated_mask_matches_unmasked_kernel():
    """With every cell populated the kernel's own weights apply on every axis.

    The masked path must then agree with the plain contraction, so the bracket branch
    never changes a grid that has no gap.
    """
    rng = np.random.default_rng(2575)
    grid = jnp.asarray(rng.normal(size=(len(_V), len(_B), len(_N), 2)))
    mask = np.ones((len(_B), len(_N)))
    point = (0.37, 0.61, 0.42)
    plain = interp_nd_triweight(grid, _AXES, _EDGES, point)
    masked = interp_nd_triweight(grid, _AXES, _EDGES, point, population_mask=jnp.asarray(mask.T))
    np.testing.assert_allclose(np.asarray(masked), np.asarray(plain), rtol=1e-9, atol=1e-12)
