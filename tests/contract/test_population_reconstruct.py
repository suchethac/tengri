# SPDX-License-Identifier: BSD-3-Clause
import chex
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.sfh.registry import compute_field_gp

pytestmark = pytest.mark.contract


def test_reconstruction_matches_the_forward_model_exactly():
    from tengri.inference.population.reconstruct import centered_fields

    n = 16
    grid = jnp.asarray(np.linspace(6.0, 10.14, n))
    xi = jnp.asarray(np.random.default_rng(0).normal(size=n))
    d_log_age = float(grid[1] - grid[0])

    gp_x, k0_half = compute_field_gp(xi, 0.9, 8.0e7, n, d_log_age, log_age_grid=grid)
    want = gp_x - k0_half
    got = centered_fields(xi, 0.9, 8.0e7, grid)
    chex.assert_trees_all_close(got, want, rtol=0.0, atol=0.0)


def test_reconstruction_carries_the_lognormal_bias_term():
    """A reconstruction that returns gp_x alone must fail this."""
    from tengri.inference.population.reconstruct import centered_fields

    n = 12
    grid = jnp.asarray(np.linspace(6.0, 10.14, n))
    xi = jnp.zeros(n)
    got = centered_fields(xi, 1.6, 5.0e7, grid)
    # With xi = 0 the correlated part vanishes and only -k0_half survives.
    expected = -0.5 * (1.6 * np.log(10.0)) ** 2
    chex.assert_trees_all_close(got, jnp.full((n,), expected), rtol=1e-12)
    assert float(jnp.max(jnp.abs(got))) > 1.0, (
        "bias term is large here; a zero result means it was dropped"
    )


def test_grid_mismatch_raises_instead_of_reshaping_into_garbage():
    """A trailing axis that is not the field grid must fail loudly.

    Without this guard ``xi.reshape(-1, n_grid)`` silently redistributes the
    elements and the error surfaces much later as an unrelated vmap complaint
    about ``sigma`` having the wrong length.
    """
    from tengri.inference.population.reconstruct import centered_fields

    grid = jnp.asarray(np.linspace(6.0, 10.14, 16))
    xi_wrong = jnp.zeros((4, 1000, 256))  # 256 latents against a 16-point grid
    with pytest.raises(ValueError, match="trailing axis"):
        centered_fields(xi_wrong, 0.9, 8.0e7, grid)


def test_multi_axis_xi_round_trips_shape():
    """(N, K, n) is the shape the estimator actually passes."""
    from tengri.inference.population.reconstruct import centered_fields

    n = 16
    grid = jnp.asarray(np.linspace(6.0, 10.14, n))
    xi = jnp.asarray(np.random.default_rng(0).normal(size=(3, 5, n)))
    out = centered_fields(xi, 0.9, 8.0e7, grid)
    chex.assert_shape(out, (3, 5, n))
    chex.assert_tree_all_finite(out)
