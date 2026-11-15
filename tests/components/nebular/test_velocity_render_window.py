# SPDX-License-Identifier: BSD-3-Clause
"""The velocity-triweight line renderer evaluates each line only on its support window.

``place_line_profiles_velocity`` used to build the full ``(n_wave, n_lines)`` profile
matrix. A triweight profile is exactly zero outside ``|u| < 1``, so a line only touches
the grid nodes inside its window. The windowed renderer must reproduce the dense
reference to round-off (values and gradients), on realistic and coarse grids, and fall
back to the dense path when a window is too wide to hold.

Tracks issue #3526 (exact Cue line path speed-up).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.nebular._shared import (
    VELOCITY_WINDOW_NODES,
    place_line_profiles_velocity,
    place_line_profiles_velocity_dense,
    velocity_window_nodes,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _mist_like_grid() -> np.ndarray:
    """Dense optical block inside a coarse UV/IR envelope, as the SSP grids are."""
    return np.concatenate(
        [
            np.linspace(1000.0, 3500.0, 3000),
            np.linspace(3500.0, 9000.0, 5000),
            np.geomspace(9000.0, 1.0e6, 2992),
        ]
    )


def _coarse_grid() -> np.ndarray:
    return np.geomspace(900.0, 1.0e6, 500)


def _random_lines(seed: int, n: int = 138) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    wav = np.sort(rng.uniform(912.0, 25000.0, n))
    lum = rng.uniform(1e-3, 1e3, n)
    return wav, lum


def _midpoint_lines(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lines centered between two nodes, where a sub-pixel profile vanishes on the grid."""
    wav = 0.5 * (grid[100:200] + grid[101:201])
    return wav, np.ones_like(wav)


@pytest.mark.parametrize("sigma", [0.0, 1.0, 50.0, 100.0, 300.0, 800.0, 2000.0])
@pytest.mark.parametrize("grid_name", ["mist_like", "coarse"])
def test_windowed_render_matches_dense_reference(grid_name, sigma):
    """Values agree with the dense reference to round-off across the declared widths."""
    grid = _mist_like_grid() if grid_name == "mist_like" else _coarse_grid()
    wav, lum = _random_lines(seed=3)
    sparse = np.asarray(
        place_line_profiles_velocity(jnp.asarray(wav), jnp.asarray(lum), jnp.asarray(grid), sigma)
    )
    dense = np.asarray(
        place_line_profiles_velocity_dense(
            jnp.asarray(wav), jnp.asarray(lum), jnp.asarray(grid), sigma
        )
    )
    scale = np.max(np.abs(dense))
    np.testing.assert_allclose(sparse, dense, rtol=1e-12, atol=1e-14 * scale)


def test_windowed_render_matches_dense_for_lines_centered_between_nodes():
    """A sub-pixel line that the dense path scatters to its nearest node is reproduced exactly."""
    grid = _coarse_grid()
    wav, lum = _midpoint_lines(grid)
    sparse = np.asarray(
        place_line_profiles_velocity(jnp.asarray(wav), jnp.asarray(lum), jnp.asarray(grid), 0.0)
    )
    dense = np.asarray(
        place_line_profiles_velocity_dense(
            jnp.asarray(wav), jnp.asarray(lum), jnp.asarray(grid), 0.0
        )
    )
    np.testing.assert_allclose(sparse, dense, rtol=1e-12, atol=1e-14 * np.max(np.abs(dense)))


@pytest.mark.gradient
@pytest.mark.parametrize("grid_name", ["mist_like", "coarse"])
def test_windowed_render_gradients_match_dense(grid_name):
    """d/dL and d/dsigma_v through the window agree with the dense reference."""
    grid = jnp.asarray(_mist_like_grid() if grid_name == "mist_like" else _coarse_grid())
    wav, lum = _random_lines(seed=5)
    wav, lum = jnp.asarray(wav), jnp.asarray(lum)

    def total(render):
        return lambda L, s: jnp.sum(
            render(wav, L, grid, s) * jnp.linspace(0.5, 1.5, grid.shape[0])
        )

    g_sparse = jax.grad(total(place_line_profiles_velocity), argnums=(0, 1))(lum, 120.0)
    g_dense = jax.grad(total(place_line_profiles_velocity_dense), argnums=(0, 1))(lum, 120.0)
    for a, b in zip(g_sparse, g_dense):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-9, atol=1e-14)


@pytest.mark.bounds
def test_declared_width_keeps_windows_within_budget_on_realistic_grid():
    """At the default 100 km/s and the 500 km/s prior bound, windows fit the budget."""
    grid = jnp.asarray(_mist_like_grid())
    wav = jnp.asarray(_random_lines(seed=7)[0])
    assert int(velocity_window_nodes(wav, grid, 100.0)) <= VELOCITY_WINDOW_NODES
    assert int(velocity_window_nodes(wav, grid, 500.0)) <= VELOCITY_WINDOW_NODES
    # A window wider than the budget is still rendered exactly (dense fallback).
    assert int(velocity_window_nodes(wav, grid, 2000.0)) > VELOCITY_WINDOW_NODES


@pytest.mark.gradient
def test_windowed_render_forward_mode_matches_dense():
    """Forward-mode through the cond/checkpoint wrapper (Hessian paths) matches dense."""
    grid = jnp.asarray(_mist_like_grid())
    wav, lum = (jnp.asarray(x) for x in _random_lines(seed=9))

    def jac(render):
        return jax.jacfwd(lambda s: render(wav, lum, grid, s))(120.0)

    np.testing.assert_allclose(
        np.asarray(jac(place_line_profiles_velocity)),
        np.asarray(jac(place_line_profiles_velocity_dense)),
        rtol=1e-9,
        atol=1e-14,
    )
