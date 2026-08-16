# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for forward_chunk_size chunked signal_response logic.

Tests the lax.map(vmap_K) chunked path in isolation, without spinning up a full
PopulationFitter (which requires SSP data). We build a toy signal_response closure
that mirrors what _run_native_vi_linear / _run_native_vi_nonlinear construct and
verify that the K>1 path returns identical results to the K=1 path.

Model: each galaxy has a scalar param 'x' in p["gal"]["x"].
       forward_one(ub, xi) = scale * jnp.array([ub["x"]]) * bias  (shape n_dpg)
       This mirrors the real case: ub_scalars["param"] is a scalar, A is 1D.
"""

import math

import chex
import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.bounds


def _make_signal_responses(n_gal, n_dpg, K, seed=1):
    """Build K=1 and K>1 signal_response closures for a toy linear model.

    forward_one(ub_scalars, xi) = A * ub_scalars["x"], shape (n_dpg,)
    where A is a fixed (n_dpg,) vector and ub_scalars["x"] is a scalar.

    p["gal"]["x"] has shape (n_padded,) — a scalar param per galaxy.
    """
    A = jax.random.normal(jax.random.PRNGKey(seed), (n_dpg,))
    n_padded = math.ceil(n_gal / K) * K
    n_chunks = n_padded // K

    def forward_one(ub_scalars, _xi):
        return A * ub_scalars["x"]

    def signal_response_k1(p):
        predictions = jax.lax.map(lambda ub: forward_one(ub, None), p["gal"])
        return predictions.reshape(-1)  # (n_padded * n_dpg,) but n_padded == n_gal here

    def signal_response_kn(p):
        chunked_gal = jax.tree.map(lambda arr: arr.reshape(n_chunks, K), p["gal"])
        predictions = jax.lax.map(
            lambda chunk: jax.vmap(lambda ub: forward_one(ub, None))(chunk),
            chunked_gal,
        )
        return predictions.reshape(n_padded, n_dpg)[:n_gal].reshape(-1)

    return signal_response_k1, signal_response_kn, n_padded


def _make_params(n_gal, n_padded, seed=0):
    """p["gal"]["x"]: shape (n_padded,), padded with zeros beyond n_gal."""
    x_real = jax.random.normal(jax.random.PRNGKey(seed), (n_gal,))
    x_padded = jnp.concatenate([x_real, jnp.zeros(n_padded - n_gal)], axis=0)
    return {"gal": {"x": x_padded}}


@pytest.mark.parametrize("K", [2, 4])
@pytest.mark.parametrize("n_gal,n_dpg", [(4, 3), (8, 5), (6, 2)])
def test_chunked_matches_sequential(n_gal, n_dpg, K):
    """K>1 chunked path produces same predictions as K=1 sequential path."""
    sr_k1, sr_kn, n_padded = _make_signal_responses(n_gal, n_dpg, K)

    # K=1 baseline: use n_padded=n_gal (no padding)
    p_k1 = _make_params(n_gal, n_gal)
    p_kn = _make_params(n_gal, n_padded)

    out_k1 = sr_k1(p_k1)
    out_kn = sr_kn(p_kn)

    chex.assert_shape(out_k1, (n_gal * n_dpg,))
    chex.assert_shape(out_kn, (n_gal * n_dpg,))
    assert jnp.allclose(out_k1, out_kn, atol=1e-10), (
        f"K={K}, n_gal={n_gal}: max diff = {jnp.max(jnp.abs(out_k1 - out_kn)):.2e}"
    )


def test_chunked_output_shape_non_divisible():
    """n_gal not divisible by K: padded galaxies are trimmed correctly."""
    n_gal, n_dpg, K = 7, 3, 4
    _, sr_kn, n_padded = _make_signal_responses(n_gal, n_dpg, K)
    assert n_padded == 8  # ceil(7/4)*4
    p = _make_params(n_gal, n_padded)
    out = sr_kn(p)
    chex.assert_shape(out, (n_gal * n_dpg,))


def test_padding_zeros_dont_affect_real_predictions():
    """Predictions of real galaxies are unaffected by values in padded slots."""
    n_gal, n_dpg, K = 6, 2, 4
    _, sr_kn, n_padded = _make_signal_responses(n_gal, n_dpg, K)

    p_zeros = _make_params(n_gal, n_padded)
    p_large = {"gal": {"x": p_zeros["gal"]["x"].at[n_gal:].set(1e6)}}

    out_zeros = sr_kn(p_zeros)
    out_large = sr_kn(p_large)
    assert jnp.allclose(out_zeros, out_large, atol=1e-10)


def test_k1_is_identity_for_divisible_n_gal():
    """When n_gal divisible by K, K>1 path matches K=1 with no padding."""
    n_gal, n_dpg, K = 8, 3, 4
    sr_k1, sr_kn, n_padded = _make_signal_responses(n_gal, n_dpg, K)
    assert n_padded == n_gal
    p = _make_params(n_gal, n_gal)
    assert jnp.allclose(sr_k1(p), sr_kn(p), atol=1e-12)
