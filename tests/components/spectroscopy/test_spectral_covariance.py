# SPDX-License-Identifier: BSD-3-Clause
"""Tests for spectral covariance matrix input.

Verifies:
1. Spectroscopy accepts a covariance matrix
2. Covariance is validated (shape, symmetry)
3. Inverse is precomputed
4. has_covariance property works
5. Summary includes covariance indicator
6. Diagonal covariance reproduces per-pixel noise
"""

import chex
import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

pytestmark = pytest.mark.bounds

from tengri.observation.spectroscopy import Spectroscopy


@pytest.fixture
def wave_obs():
    return jnp.linspace(4000.0, 9000.0, 50)


@pytest.fixture
def diagonal_cov(wave_obs):
    n = len(wave_obs)
    sigma = jnp.full(n, 0.1)
    return jnp.diag(sigma**2)


class TestConstruction:
    def test_no_covariance_by_default(self, wave_obs):
        spec = Spectroscopy(wave_obs=wave_obs)
        assert not spec.has_covariance
        assert spec.cov_inv is None

    def test_covariance_accepted(self, wave_obs, diagonal_cov):
        spec = Spectroscopy(wave_obs=wave_obs, covariance=diagonal_cov)
        assert spec.has_covariance
        assert spec.cov_inv is not None

    def test_wrong_shape_raises(self, wave_obs):
        bad_cov = jnp.eye(10)
        with pytest.raises(ValueError, match="covariance shape"):
            Spectroscopy(wave_obs=wave_obs, covariance=bad_cov)


class TestInversePrecomputation:
    def test_inverse_shape(self, wave_obs, diagonal_cov):
        spec = Spectroscopy(wave_obs=wave_obs, covariance=diagonal_cov)
        n = len(wave_obs)
        chex.assert_shape(spec.cov_inv, (n, n))

    def test_inverse_correct_for_diagonal(self, wave_obs):
        n = len(wave_obs)
        sigma = jnp.linspace(0.05, 0.2, n)
        cov = jnp.diag(sigma**2)
        spec = Spectroscopy(wave_obs=wave_obs, covariance=cov)
        expected_inv = jnp.diag(1.0 / sigma**2)
        assert_allclose(spec.cov_inv, expected_inv, rtol=1e-10)

    def test_inverse_times_cov_is_identity(self, wave_obs):
        n = len(wave_obs)
        rng = jax.random.PRNGKey(42)
        A = jax.random.normal(rng, (n, n)) * 0.01
        cov = A @ A.T + 0.1 * jnp.eye(n)
        spec = Spectroscopy(wave_obs=wave_obs, covariance=cov)
        product = spec.cov_inv @ cov
        assert_allclose(product, jnp.eye(n), atol=1e-8)


class TestChi2Equivalence:
    def test_diagonal_cov_equals_perpixel(self, wave_obs):
        n = len(wave_obs)
        sigma = jnp.full(n, 0.1)
        cov = jnp.diag(sigma**2)
        spec = Spectroscopy(wave_obs=wave_obs, covariance=cov)
        diff = jax.random.normal(jax.random.PRNGKey(0), (n,))
        chi2_diagonal = jnp.sum((diff / sigma) ** 2)
        chi2_cov = diff @ spec.cov_inv @ diff
        assert_allclose(chi2_cov, chi2_diagonal, rtol=1e-10)

    def test_offdiagonal_changes_chi2(self, wave_obs):
        n = len(wave_obs)
        sigma = jnp.full(n, 0.1)
        cov_diag = jnp.diag(sigma**2)
        rng = jax.random.PRNGKey(1)
        A = jax.random.normal(rng, (n, n)) * 0.001
        cov_full = cov_diag + A @ A.T
        spec_diag = Spectroscopy(wave_obs=wave_obs, covariance=cov_diag)
        spec_full = Spectroscopy(wave_obs=wave_obs, covariance=cov_full)
        diff = jax.random.normal(jax.random.PRNGKey(2), (n,))
        chi2_diag = diff @ spec_diag.cov_inv @ diff
        chi2_full = diff @ spec_full.cov_inv @ diff
        assert chi2_diag != chi2_full


class TestSummary:
    def test_summary_includes_cov(self, wave_obs, diagonal_cov):
        spec = Spectroscopy(wave_obs=wave_obs, covariance=diagonal_cov)
        assert "cov_matrix" in spec.summary()

    def test_summary_no_cov_by_default(self, wave_obs):
        spec = Spectroscopy(wave_obs=wave_obs)
        assert "cov_matrix" not in spec.summary()
