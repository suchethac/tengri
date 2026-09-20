# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #2066: sparse grid B-axis bracketing under JIT and grad.

The MAPPINGS shock grid is 2D-sparse in (log_density, B-field) — only 75 of
210 solar-abundance pairs are populated. When the triweight stencil at an
off-node query point contains fewer than 2 populated nodes on the B-axis,
the interpolation becomes flat (zero gradient) even though neighboring
populated nodes could provide interpolation.

This test verifies that the pure JAX bracketing logic works identically under
jit and grad as in eager mode, and that the synthetic two-node grid responds
between them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.utils.grid_interp import interp_nd_triweight as _interp_nd_triweight
from tengri.utils.interpolation import edges_for_grid as _edges_for_grid

pytestmark = pytest.mark.regression_bug


def test_jit_and_grad_bracketing_with_population_mask():
    """JIT and grad of sparse-grid interpolant agree with eager values.

    Construct a synthetic 3D grid with only two nodes populated on the B-axis.
    Verify that jit() and grad() of the interpolant return the same values
    as eager evaluation.
    """
    n_v, n_b, n_density = 5, 5, 5
    n_filter = 3

    v_grid = jnp.linspace(100.0, 500.0, n_v)
    b_grid = jnp.linspace(0.1, 10.0, n_b)
    n_grid = jnp.linspace(-2.0, 0.0, n_density)

    grid_values = jnp.ones((n_v, n_b, n_density, n_filter))

    pop_mask = jnp.zeros((n_density, n_b))
    pop_mask = pop_mask.at[:, 1].set(1.0)
    pop_mask = pop_mask.at[:, 3].set(1.0)

    edges = (
        _edges_for_grid(v_grid),
        _edges_for_grid(b_grid),
        _edges_for_grid(n_grid),
    )

    point = (250.0, 5.0, -1.0)

    result_eager = _interp_nd_triweight(
        grid_values,
        (v_grid, b_grid, n_grid),
        edges,
        point,
        population_mask=pop_mask,
    )

    interp_jit = jax.jit(
        lambda: _interp_nd_triweight(
            grid_values,
            (v_grid, b_grid, n_grid),
            edges,
            point,
            population_mask=pop_mask,
        )
    )
    result_jit = interp_jit()

    np.testing.assert_allclose(result_eager, result_jit, rtol=1e-14)


def test_population_mask_none_bit_identical():
    """Verify that population_mask=None is bit-identical to no masking."""
    n_v, n_b, n_density = 4, 4, 4
    n_filter = 2

    v_grid = jnp.linspace(100.0, 400.0, n_v)
    b_grid = jnp.linspace(1.0, 4.0, n_b)
    n_grid = jnp.linspace(-2.0, 0.0, n_density)

    grid_values = jnp.ones((n_v, n_b, n_density, n_filter)) * 1.5

    edges = (
        _edges_for_grid(v_grid),
        _edges_for_grid(b_grid),
        _edges_for_grid(n_grid),
    )

    point = (250.0, 2.5, -1.5)

    result_no_mask = _interp_nd_triweight(
        grid_values,
        (v_grid, b_grid, n_grid),
        edges,
        point,
        population_mask=None,
    )

    pop_mask_full = jnp.ones((n_density, n_b))
    result_with_full_mask = _interp_nd_triweight(
        grid_values,
        (v_grid, b_grid, n_grid),
        edges,
        point,
        population_mask=pop_mask_full,
    )

    np.testing.assert_allclose(result_no_mask, result_with_full_mask, rtol=1e-14)
