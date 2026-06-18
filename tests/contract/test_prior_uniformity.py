# SPDX-License-Identifier: BSD-3-Clause
"""Contract test for bounded-parameter transform uniformity.

Verifies that an N(0,1) latent variable yields a uniform prior on (lo, hi),
not a midpoint-peaked one. This is the fix for issue #716.

Tests:
- Gaussian-CDF standardization produces uniform distribution
- Both to_bounded()/to_unbounded() and Uniform.unstandardize()/standardize()
  agree exactly
- Statistical properties: median, quantiles, uniformity test
"""

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.contract

from tengri.parameters.priors import Uniform
from tengri.utils.transforms import to_bounded, to_unbounded


class TestPriorUniformity:
    """Test that N(0,1) latent → uniform distribution, not midpoint-peaked."""

    def test_to_bounded_produces_uniform_distribution(self):
        """to_bounded(N(0,1)) produces uniform on (lo, hi).

        Draws ~200k samples from standard normal, transforms via to_bounded,
        and verifies the distribution is uniform.

        Tolerance: P(sigma < 0.7) should be ~0.1538 (uniform on (0.1, 4.0)),
        not ~0.045 (sigmoid midpoint-peaked).
        """
        key = jax.random.PRNGKey(0)
        lo, hi = 0.1, 4.0
        n_samples = 200_000

        # Draw N(0,1) samples
        keys = jax.random.split(key, n_samples)
        normal_draws = jax.vmap(lambda k: jax.random.normal(k))(keys)

        # Transform to bounded
        bounded = jax.vmap(lambda u: to_bounded(u, lo, hi))(normal_draws)

        # Check empirical CDF: P(x < 0.7)
        # Uniform(0.1, 4.0): P(x < 0.7) = (0.7 - 0.1) / (4.0 - 0.1) = 0.6/3.9 ≈ 0.1538
        empirical_cdf_at_07 = (bounded < 0.7).mean()
        expected_cdf_at_07 = (0.7 - lo) / (hi - lo)
        assert jnp.abs(empirical_cdf_at_07 - expected_cdf_at_07) < 0.01

    def test_uniform_prior_unstandardize_is_uniform(self):
        """Uniform.unstandardize(N(0,1)) produces uniform on (lo, hi).

        Same test as above, but via the Uniform prior class.
        """
        key = jax.random.PRNGKey(1)
        prior = Uniform(0.1, 4.0)
        n_samples = 200_000

        keys = jax.random.split(key, n_samples)
        normal_draws = jax.vmap(lambda k: jax.random.normal(k))(keys)
        bounded = jax.vmap(prior.unstandardize)(normal_draws)

        empirical_cdf_at_07 = (bounded < 0.7).mean()
        expected_cdf_at_07 = (0.7 - 0.1) / (4.0 - 0.1)
        assert jnp.abs(empirical_cdf_at_07 - expected_cdf_at_07) < 0.01

    def test_median_of_uniform_transform(self):
        """Median of to_bounded(N(0,1)) should be ~2.05 for Uniform(0.1, 4.0).

        At u=0 (median of N(0,1)), Φ(0) = 0.5, so θ = lo + 0.5*(hi-lo) = 2.05.
        """
        key = jax.random.PRNGKey(2)
        lo, hi = 0.1, 4.0
        n_samples = 200_000

        keys = jax.random.split(key, n_samples)
        normal_draws = jax.vmap(lambda k: jax.random.normal(k))(keys)
        bounded = jax.vmap(lambda u: to_bounded(u, lo, hi))(normal_draws)

        empirical_median = jnp.median(bounded)
        expected_median = lo + 0.5 * (hi - lo)
        assert jnp.abs(empirical_median - expected_median) < 0.05

    def test_quantiles_match_uniform(self):
        """25th and 75th quantiles should match Uniform(lo, hi).

        For Uniform(0.1, 4.0):
        - 25th percentile: 0.1 + 0.25 * (4.0 - 0.1) ≈ 1.075
        - 75th percentile: 0.1 + 0.75 * (4.0 - 0.1) ≈ 3.025
        """
        key = jax.random.PRNGKey(3)
        lo, hi = 0.1, 4.0
        n_samples = 200_000

        keys = jax.random.split(key, n_samples)
        normal_draws = jax.vmap(lambda k: jax.random.normal(k))(keys)
        bounded = jax.vmap(lambda u: to_bounded(u, lo, hi))(normal_draws)

        q25_emp = jnp.quantile(bounded, 0.25)
        q75_emp = jnp.quantile(bounded, 0.75)

        q25_exp = lo + 0.25 * (hi - lo)
        q75_exp = lo + 0.75 * (hi - lo)

        assert jnp.abs(q25_emp - q25_exp) < 0.1
        assert jnp.abs(q75_emp - q75_exp) < 0.1

    def test_to_bounded_to_unbounded_roundtrip(self):
        """to_unbounded(to_bounded(u)) ≈ u for typical N(0,1) draws.

        Verifies the inverse relationship to numerical precision.
        """
        key = jax.random.PRNGKey(4)
        lo, hi = 0.1, 4.0
        n_samples = 1000

        keys = jax.random.split(key, n_samples)
        u_orig = jax.vmap(lambda k: jax.random.normal(k))(keys)

        # Forward
        bounded = jax.vmap(lambda u: to_bounded(u, lo, hi))(u_orig)

        # Inverse
        u_reconstructed = jax.vmap(lambda theta: to_unbounded(theta, lo, hi))(bounded)

        # Should match within floating-point precision
        assert jnp.allclose(u_orig, u_reconstructed, atol=1e-5, rtol=1e-5)

    def test_uniform_prior_roundtrip(self):
        """Uniform.standardize(Uniform.unstandardize(ξ)) ≈ ξ.

        Verifies the inverse relationship in the Uniform prior class.
        """
        key = jax.random.PRNGKey(5)
        prior = Uniform(0.1, 4.0)
        n_samples = 1000

        keys = jax.random.split(key, n_samples)
        xi_orig = jax.vmap(lambda k: jax.random.normal(k))(keys)

        # Forward
        theta = jax.vmap(prior.unstandardize)(xi_orig)

        # Inverse
        xi_reconstructed = jax.vmap(prior.standardize)(theta)

        # Should match within floating-point precision
        assert jnp.allclose(xi_orig, xi_reconstructed, atol=1e-5, rtol=1e-5)

    def test_transforms_and_prior_agree(self):
        """to_bounded and Uniform.unstandardize produce identical results.

        Both should use the same Gaussian-CDF formula.
        """
        key = jax.random.PRNGKey(6)
        prior = Uniform(0.1, 4.0)
        lo, hi = 0.1, 4.0
        n_samples = 1000

        keys = jax.random.split(key, n_samples)
        normal_draws = jax.vmap(lambda k: jax.random.normal(k))(keys)

        # Via to_bounded
        via_transforms = jax.vmap(lambda u: to_bounded(u, lo, hi))(normal_draws)

        # Via Uniform prior
        via_prior = jax.vmap(prior.unstandardize)(normal_draws)

        # Should be identical (bit-exact if same formula)
        assert jnp.allclose(via_transforms, via_prior, atol=1e-10)

    def test_uniform_cdf_property(self):
        """For Uniform(0.1, 4.0), P(x ≤ q) = (q - 0.1) / 3.9 for q ∈ [0.1, 4.0].

        Tests the cumulative distribution function property.
        """
        key = jax.random.PRNGKey(7)
        prior = Uniform(0.1, 4.0)
        n_samples = 200_000

        keys = jax.random.split(key, n_samples)
        normal_draws = jax.vmap(lambda k: jax.random.normal(k))(keys)
        samples = jax.vmap(prior.unstandardize)(normal_draws)

        # Test multiple quantile points
        test_quantiles = [0.1, 1.0, 2.0, 3.0, 3.9]
        for q in test_quantiles:
            empirical_cdf = (samples <= q).mean()
            expected_cdf = (q - 0.1) / 3.9
            # Tolerance widens slightly at extremes (1e-3 enough)
            assert jnp.abs(empirical_cdf - expected_cdf) < 0.015
