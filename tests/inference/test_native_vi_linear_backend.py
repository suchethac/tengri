# SPDX-License-Identifier: BSD-3-Clause
"""Tests for build_native_vi_linear_engine in isolation."""

import chex
import jax
import jax.numpy as jnp
import pytest
from jax.flatten_util import ravel_pytree

from tengri.inference.backends.vi.native import build_native_vi_linear_engine

pytestmark = pytest.mark.contract


def _make_linear_problem(n_data=8, n_params=3, seed=0):
    """Trivial linear model y = A @ x with Gaussian noise."""
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


def test_build_returns_callables():
    sr, data, noise, _flat0, flatten, unravel, _ = _make_linear_problem()
    run_fn, draw_res, hamiltonian = build_native_vi_linear_engine(
        sr, data, noise, flatten, unravel
    )
    assert callable(run_fn)
    assert callable(draw_res)
    assert callable(hamiltonian)


def test_converges_on_linear_problem():
    sr, data, noise, flat0, flatten, unravel, _ = _make_linear_problem()
    run_fn, _, hamiltonian = build_native_vi_linear_engine(sr, data, noise, flatten, unravel)
    key = jax.random.PRNGKey(42)
    best_flat, n_iters = run_fn(flat0, key, n_iter=30, n_samp=3, rtol=1e-3)
    assert float(hamiltonian(best_flat)) < float(hamiltonian(flat0))
    assert int(n_iters) > 0


def test_draw_residuals_shape():
    sr, data, noise, flat0, flatten, unravel, _ = _make_linear_problem()
    run_fn, draw_res, _ = build_native_vi_linear_engine(sr, data, noise, flatten, unravel)
    key = jax.random.PRNGKey(0)
    best_flat, _ = run_fn(flat0, key, n_iter=10, n_samp=2, rtol=1e-2)
    draw_keys = jax.random.split(jax.random.PRNGKey(1), 5)
    residuals = draw_res(best_flat, draw_keys)
    chex.assert_shape(residuals, (5, flat0.shape[0]))


def test_hamiltonian_is_scalar():
    sr, data, noise, flat0, flatten, unravel, _ = _make_linear_problem()
    _, _, hamiltonian = build_native_vi_linear_engine(sr, data, noise, flatten, unravel)
    h = hamiltonian(flat0)
    chex.assert_shape(h, ())


def test_independent_engines_dont_share_state():
    """Two engines built from different data should produce different results."""
    sr1, data1, noise1, flat0, flatten, unravel, _ = _make_linear_problem(seed=0)
    sr2, data2, noise2, _, _, _, _ = _make_linear_problem(seed=99)
    run1, _, _ = build_native_vi_linear_engine(sr1, data1, noise1, flatten, unravel)
    run2, _, _ = build_native_vi_linear_engine(sr2, data2, noise2, flatten, unravel)
    key = jax.random.PRNGKey(7)
    f1, _ = run1(flat0, key, n_iter=5, n_samp=2, rtol=0.0)
    f2, _ = run2(flat0, key, n_iter=5, n_samp=2, rtol=0.0)
    assert not jnp.allclose(f1, f2)
