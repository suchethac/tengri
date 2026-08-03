# SPDX-License-Identifier: BSD-3-Clause
import chex
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.population.kernel import ou_logpdf

pytestmark = pytest.mark.contract


def _dense_drw_logpdf(m, mean, sigma_dex, tau_yr, times_yr):
    """Reference: build K densely and call scipy. O(n^3), test-only."""
    from scipy.stats import multivariate_normal

    var = (sigma_dex * np.log(10.0)) ** 2
    dt = np.abs(times_yr[:, None] - times_yr[None, :])
    cov = var * np.exp(-dt / tau_yr)
    return multivariate_normal.logpdf(np.asarray(m), mean=np.full(len(m), mean), cov=cov)


def test_ou_logpdf_matches_dense_multivariate_normal():
    times = np.logspace(6.0, 10.1, 12)
    m = np.linspace(-0.7, 0.9, 12)
    got = float(ou_logpdf(jnp.asarray(m), -0.3, 0.8, 1.5e8, jnp.asarray(times)))
    want = _dense_drw_logpdf(m, -0.3, 0.8, 1.5e8, times)
    chex.assert_trees_all_close(got, want, rtol=1e-9)


def test_ou_logpdf_invariant_under_joint_reversal():
    times = np.logspace(6.0, 10.1, 9)
    m = np.linspace(-0.4, 0.6, 9)
    fwd = ou_logpdf(jnp.asarray(m), 0.0, 0.5, 2e8, jnp.asarray(times))
    rev = ou_logpdf(jnp.asarray(m[::-1]), 0.0, 0.5, 2e8, jnp.asarray(times[::-1]))
    chex.assert_trees_all_close(fwd, rev, rtol=1e-12)


def test_ou_logpdf_vmaps_over_sigma_and_tau():
    import jax

    times = jnp.asarray(np.logspace(6.0, 10.1, 6))
    m = jnp.asarray(np.linspace(-0.2, 0.3, 6))
    sigmas = jnp.asarray([0.3, 0.8, 1.5])
    taus = jnp.asarray([1e7, 5e7, 2e8])
    out = jax.vmap(lambda s, t: ou_logpdf(m, 0.0, s, t, times))(sigmas, taus)
    chex.assert_shape(out, (3,))
    chex.assert_tree_all_finite(out)
