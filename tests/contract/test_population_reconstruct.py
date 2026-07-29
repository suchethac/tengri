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
