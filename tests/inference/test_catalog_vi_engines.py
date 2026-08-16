# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for build_native_vi_catalog_linear_engine and
build_native_vi_catalog_nonlinear_engine in isolation.

These tests use a toy linear model (y = A @ x) so they run without any SSP
data.  They verify the engines' public contracts:
  - run_fn(init, key, data, noise, ...) -> (flat, n_iters)
  - draw_fn(pos, subkeys, noise) -> residuals shape (n_draws, d)
  - different noise -> different converged positions (no state sharing)
"""

import chex
import pytest

pytestmark = pytest.mark.contract

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from tengri.inference.backends.vi.native import (
    build_native_vi_catalog_linear_engine,
    build_native_vi_catalog_nonlinear_engine,
)

# ---------------------------------------------------------------------------
# Shared toy model
# ---------------------------------------------------------------------------


def _make_problem(n_data=8, n_params=4, seed=0):
    """Linear model y = A @ x; known posterior is tractable."""
    key = jax.random.PRNGKey(seed)
    A = jax.random.normal(key, (n_data, n_params))
    x_true = jax.random.normal(jax.random.PRNGKey(seed + 1), (n_params,))
    data = A @ x_true
    noise = jnp.ones(n_data)

    init = {"x": jnp.zeros(n_params)}
    flat0, unravel = ravel_pytree(init)

    def signal_response(params):
        return A @ params["x"]

    def flatten(d):
        return ravel_pytree(d)[0]

    return signal_response, data, noise, flat0, flatten, unravel, x_true


# ---------------------------------------------------------------------------
# Linear engine tests
# ---------------------------------------------------------------------------


def test_linear_engine_returns_callables():
    sr, _data, _noise, _flat0, flatten, unravel, _ = _make_problem()
    run_fn, draw_fn, hamiltonian_fn = build_native_vi_catalog_linear_engine(sr, flatten, unravel)
    assert callable(run_fn)
    assert callable(draw_fn)
    assert callable(hamiltonian_fn)


def test_linear_engine_run_output_shape():
    sr, data, noise, flat0, flatten, unravel, _ = _make_problem()
    run_fn, _, _ = build_native_vi_catalog_linear_engine(sr, flatten, unravel)
    best, n_iters = run_fn(flat0, jax.random.PRNGKey(0), data, noise, 10, 2, 1e-2)
    chex.assert_equal_shape([best, flat0])
    chex.assert_shape(n_iters, ())


def test_linear_engine_lowers_hamiltonian():
    sr, data, noise, flat0, flatten, unravel, _ = _make_problem()
    run_fn, _, hamiltonian_fn = build_native_vi_catalog_linear_engine(sr, flatten, unravel)
    best, _ = run_fn(flat0, jax.random.PRNGKey(42), data, noise, 30, 3, 1e-3)
    h_init = float(hamiltonian_fn(flat0, data, noise))
    h_best = float(hamiltonian_fn(best, data, noise))
    assert h_best < h_init


def test_linear_engine_draw_shape():
    sr, data, noise, flat0, flatten, unravel, _ = _make_problem()
    run_fn, draw_fn, _ = build_native_vi_catalog_linear_engine(sr, flatten, unravel)
    best, _ = run_fn(flat0, jax.random.PRNGKey(0), data, noise, 10, 2, 1e-2)
    n_draws = 7
    draw_keys = jax.random.split(jax.random.PRNGKey(1), n_draws)
    residuals = draw_fn(best, draw_keys, noise)
    chex.assert_shape(residuals, (n_draws, flat0.shape[0]))


def test_linear_engine_different_noise_different_result():
    """Independent catalogs with different noise converge to different points."""
    sr, data, noise1, flat0, flatten, unravel, _ = _make_problem(seed=0)
    noise2 = noise1 * 0.1  # much tighter noise → stronger data pull
    run_fn, _, _ = build_native_vi_catalog_linear_engine(sr, flatten, unravel)
    key = jax.random.PRNGKey(7)
    best1, _ = run_fn(flat0, key, data, noise1, 15, 2, 1e-3)
    best2, _ = run_fn(flat0, key, data, noise2, 15, 2, 1e-3)
    assert not jnp.allclose(best1, best2, atol=1e-4)


def test_linear_engine_vmap_over_catalog():
    """vmap(run_fn) over two galaxies with different data/noise executes without error."""
    sr, data, noise, flat0, flatten, unravel, _ = _make_problem(n_data=6, n_params=3)
    run_fn, _, _ = build_native_vi_catalog_linear_engine(sr, flatten, unravel)

    batch_data = jnp.stack([data, data * 0.5])  # (2, n_data)
    batch_noise = jnp.stack([noise, noise * 2.0])  # (2, n_data)
    batch_init = jnp.stack([flat0, flat0])  # (2, d)
    batch_keys = jax.random.split(jax.random.PRNGKey(3), 2)  # (2, 2)

    bests, n_iters = jax.vmap(lambda ini, k, d, n: run_fn(ini, k, d, n, 5, 2, 0.0))(
        batch_init, batch_keys, batch_data, batch_noise
    )
    chex.assert_shape(bests, (2, flat0.shape[0]))
    chex.assert_shape(n_iters, (2,))


# ---------------------------------------------------------------------------
# Nonlinear engine tests
# ---------------------------------------------------------------------------


def test_nonlinear_engine_returns_callables():
    sr, _data, _noise, _flat0, flatten, unravel, _ = _make_problem()
    run_fn, draw_fn, hamiltonian_fn = build_native_vi_catalog_nonlinear_engine(
        sr, flatten, unravel
    )
    assert callable(run_fn)
    assert callable(draw_fn)
    assert callable(hamiltonian_fn)


def test_nonlinear_engine_run_output_shape():
    sr, data, noise, flat0, flatten, unravel, _ = _make_problem()
    run_fn, _, _ = build_native_vi_catalog_nonlinear_engine(sr, flatten, unravel)
    best, n_iters = run_fn(flat0, jax.random.PRNGKey(0), data, noise, 5, 2, 1e-2)
    chex.assert_equal_shape([best, flat0])
    chex.assert_shape(n_iters, ())


def test_nonlinear_engine_draw_shape():
    """draw_fn returns mirrored pairs: n_keys keys -> 2*n_keys residuals."""
    sr, data, noise, flat0, flatten, unravel, _ = _make_problem()
    run_fn, draw_fn, _ = build_native_vi_catalog_nonlinear_engine(sr, flatten, unravel)
    best, _ = run_fn(flat0, jax.random.PRNGKey(0), data, noise, 5, 2, 1e-2)
    n_keys = 4
    draw_keys = jax.random.split(jax.random.PRNGKey(1), n_keys)
    residuals = draw_fn(best, draw_keys, noise)
    chex.assert_shape(residuals, (2 * n_keys, flat0.shape[0]))


def test_nonlinear_engine_different_noise_different_result():
    sr, data, noise1, flat0, flatten, unravel, _ = _make_problem(seed=0)
    noise2 = noise1 * 0.1
    run_fn, _, _ = build_native_vi_catalog_nonlinear_engine(sr, flatten, unravel)
    key = jax.random.PRNGKey(7)
    best1, _ = run_fn(flat0, key, data, noise1, 5, 2, 1e-3)
    best2, _ = run_fn(flat0, key, data, noise2, 5, 2, 1e-3)
    assert not jnp.allclose(best1, best2, atol=1e-4)
