# SPDX-License-Identifier: BSD-3-Clause
"""Statistical moment tests for tengri distributions.

Inspired by prospector's test_priors.py: every distribution should be verified
for both mean AND variance against analytical formulas via MC sampling.

References
----------
- Prospector test_priors.py (bd-j/prospector)
- Uniform: mean=(a+b)/2, var=(b-a)^2/12
- LogUniform (reciprocal): mean=(b-a)/ln(b/a), var=(b^2-a^2)/(2*ln(b/a)) - mean^2
- Gaussian: mean=mu, var=sigma^2 (exact by definition; clipping reduces var slightly)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.parameters.priors import Fixed, Gaussian, LogUniform, Uniform

pytestmark = pytest.mark.contract

N_SAMPLES = 20_000
RTOL_MEAN = 0.05
RTOL_VAR = 0.10


def _draw(dist, n=N_SAMPLES, seed=42):
    """Draw n samples from dist using vectorized JAX."""
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    return jax.vmap(dist.sample)(keys)


# ── Uniform ───────────────────────────────────────────────────────


class TestUniformMoments:
    def test_mean_symmetric(self):
        d = Uniform(lo=-1.0, hi=1.0)
        samples = _draw(d)
        expected = 0.0
        assert abs(float(jnp.mean(samples)) - expected) < RTOL_MEAN

    def test_variance_symmetric(self):
        d = Uniform(lo=-1.0, hi=1.0)
        samples = _draw(d)
        expected_var = (2.0) ** 2 / 12.0  # (hi-lo)^2/12
        assert abs(float(jnp.var(samples)) - expected_var) / expected_var < RTOL_VAR

    def test_mean_asymmetric(self):
        lo, hi = 2.0, 8.0
        d = Uniform(lo=lo, hi=hi)
        samples = _draw(d)
        expected = (lo + hi) / 2.0
        assert abs(float(jnp.mean(samples)) - expected) / expected < RTOL_MEAN

    def test_variance_asymmetric(self):
        lo, hi = 2.0, 8.0
        d = Uniform(lo=lo, hi=hi)
        samples = _draw(d)
        expected_var = (hi - lo) ** 2 / 12.0
        assert abs(float(jnp.var(samples)) - expected_var) / expected_var < RTOL_VAR

    def test_mean_wide(self):
        lo, hi = -100.0, 100.0
        d = Uniform(lo=lo, hi=hi)
        samples = _draw(d)
        expected = 0.0
        assert abs(float(jnp.mean(samples))) < 5.0  # absolute tolerance for zero mean

    def test_variance_wide(self):
        lo, hi = -100.0, 100.0
        d = Uniform(lo=lo, hi=hi)
        samples = _draw(d)
        expected_var = (hi - lo) ** 2 / 12.0
        assert abs(float(jnp.var(samples)) - expected_var) / expected_var < RTOL_VAR


# ── LogUniform ────────────────────────────────────────────────────


class TestLogUniformMoments:
    """LogUniform is the reciprocal distribution: p(x) = 1 / (x * ln(hi/lo)).

    Analytical moments:
        mean = (hi - lo) / ln(hi/lo)
        E[x^2] = (hi^2 - lo^2) / (2 * ln(hi/lo))
        var = E[x^2] - mean^2
    """

    @staticmethod
    def _analytical_moments(lo, hi):
        log_ratio = np.log(hi / lo)
        mean = (hi - lo) / log_ratio
        ex2 = (hi**2 - lo**2) / (2.0 * log_ratio)
        var = ex2 - mean**2
        return mean, var

    def test_mean_decade(self):
        lo, hi = 1.0, 10.0
        d = LogUniform(lo=lo, hi=hi)
        samples = _draw(d)
        expected_mean, _ = self._analytical_moments(lo, hi)
        assert abs(float(jnp.mean(samples)) - expected_mean) / expected_mean < RTOL_MEAN

    def test_variance_decade(self):
        lo, hi = 1.0, 10.0
        d = LogUniform(lo=lo, hi=hi)
        samples = _draw(d)
        _, expected_var = self._analytical_moments(lo, hi)
        assert abs(float(jnp.var(samples)) - expected_var) / expected_var < RTOL_VAR

    def test_mean_two_decades(self):
        lo, hi = 0.1, 10.0
        d = LogUniform(lo=lo, hi=hi)
        samples = _draw(d)
        expected_mean, _ = self._analytical_moments(lo, hi)
        assert abs(float(jnp.mean(samples)) - expected_mean) / expected_mean < RTOL_MEAN

    def test_variance_two_decades(self):
        lo, hi = 0.1, 10.0
        d = LogUniform(lo=lo, hi=hi)
        samples = _draw(d)
        _, expected_var = self._analytical_moments(lo, hi)
        assert abs(float(jnp.var(samples)) - expected_var) / expected_var < RTOL_VAR

    def test_all_samples_in_support(self):
        lo, hi = 0.01, 100.0
        d = LogUniform(lo=lo, hi=hi)
        samples = _draw(d)
        assert bool(jnp.all(samples >= lo))
        assert bool(jnp.all(samples <= hi))


# ── Gaussian ──────────────────────────────────────────────────────


class TestGaussianMoments:
    """Gaussian with clipping at [lo, hi].

    For wide bounds relative to sigma, mean ≈ mu and var ≈ sigma^2.
    Clipping reduces variance; we test the unclipped regime.
    """

    def test_mean_standard(self):
        mu, sigma = 0.0, 1.0
        d = Gaussian(mu=mu, sigma=sigma)
        samples = _draw(d)
        assert abs(float(jnp.mean(samples)) - mu) < RTOL_MEAN

    def test_variance_standard(self):
        mu, sigma = 0.0, 1.0
        d = Gaussian(mu=mu, sigma=sigma)
        samples = _draw(d)
        # Clipping at ±3-4 sigma — variance should be close to sigma^2
        assert abs(float(jnp.var(samples)) - sigma**2) / sigma**2 < RTOL_VAR

    def test_mean_shifted(self):
        mu, sigma = 5.0, 0.5
        d = Gaussian(mu=mu, sigma=sigma)
        samples = _draw(d)
        assert abs(float(jnp.mean(samples)) - mu) / abs(mu) < RTOL_MEAN

    def test_variance_shifted(self):
        mu, sigma = 5.0, 0.5
        d = Gaussian(mu=mu, sigma=sigma)
        samples = _draw(d)
        assert abs(float(jnp.var(samples)) - sigma**2) / sigma**2 < RTOL_VAR

    def test_mean_negative(self):
        mu, sigma = -3.0, 0.8
        d = Gaussian(mu=mu, sigma=sigma)
        samples = _draw(d)
        assert abs(float(jnp.mean(samples)) - mu) / abs(mu) < RTOL_MEAN

    def test_variance_negative(self):
        mu, sigma = -3.0, 0.8
        d = Gaussian(mu=mu, sigma=sigma)
        samples = _draw(d)
        assert abs(float(jnp.var(samples)) - sigma**2) / sigma**2 < RTOL_VAR


# ── Fixed — degenerate distribution (no variance, fixed mean) ─────


class TestFixedMoments:
    def test_mean_equals_value(self):
        d = Fixed(value=3.14)
        samples = _draw(d, n=100)
        assert bool(jnp.all(samples == 3.14))

    def test_variance_zero(self):
        d = Fixed(value=-1.5)
        samples = _draw(d, n=100)
        assert float(jnp.var(samples)) == pytest.approx(0.0)


# ── Standardize / unstandardize roundtrip ─────────────────────────


class TestStandardizeRoundtrip:
    """Distribution transforms should be exact inverses of each other."""

    def test_uniform_roundtrip(self):
        d = Uniform(lo=1.0, hi=5.0)
        theta = jnp.array([1.5, 2.0, 3.7, 4.9])
        xi = jax.vmap(d.standardize)(theta)
        theta_back = jax.vmap(d.unstandardize)(xi)
        np.testing.assert_allclose(np.array(theta_back), np.array(theta), rtol=1e-5)

    def test_uniform_unstandardize_roundtrip(self):
        d = Uniform(lo=-2.0, hi=3.0)
        xi = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        theta = jax.vmap(d.unstandardize)(xi)
        xi_back = jax.vmap(d.standardize)(theta)
        np.testing.assert_allclose(np.array(xi_back), np.array(xi), rtol=1e-5)

    def test_loguniform_roundtrip(self):
        d = LogUniform(lo=0.1, hi=10.0)
        theta = jnp.array([0.2, 0.5, 1.0, 5.0, 9.0])
        xi = jax.vmap(d.standardize)(theta)
        theta_back = jax.vmap(d.unstandardize)(xi)
        np.testing.assert_allclose(np.array(theta_back), np.array(theta), rtol=1e-5)

    def test_gaussian_roundtrip(self):
        d = Gaussian(mu=2.0, sigma=0.5)
        theta = jnp.array([1.5, 1.8, 2.0, 2.2, 2.5])
        xi = jax.vmap(d.standardize)(theta)
        theta_back = jax.vmap(d.unstandardize)(xi)
        np.testing.assert_allclose(np.array(theta_back), np.array(theta), rtol=1e-5)
