# SPDX-License-Identifier: BSD-3-Clause
"""Pure helpers for HMC + importance sampling evidence estimation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats as scipy_stats

from tengri.inference.backends.mcmc import hmc_is


@pytest.fixture
def gaussian_target():
    """Analytic Gaussian target in D=4 dimensions.

    Returns a tuple (log_target, D, m, S, log_evidence_true) where
    log_target(x) = logpdf(x; m, S) + logC with known logC=3.7.
    """
    D = 4
    m = jnp.array([1.0, -0.5, 2.0, 0.1])
    A = jnp.eye(D) + 0.3 * jax.random.normal(jax.random.PRNGKey(101), (D, D))
    S = A @ A.T + jnp.eye(D)
    logC = 3.7

    def log_target(x):
        return (
            scipy_stats.multivariate_normal.logpdf(np.array(x), mean=np.array(m), cov=np.array(S))
            + logC
        )

    return log_target, D, m, S, logC


@pytest.fixture
def gaussian_chain(gaussian_target):
    """Generate 2000 draws from the Gaussian target distribution."""
    _, D, m, S, _ = gaussian_target
    key = jax.random.PRNGKey(202)
    L = jnp.linalg.cholesky(S)
    z = jax.random.normal(key, (2000, D))
    chain = m[None, :] + z @ L.T
    return np.array(chain)


class TestFitProposal:
    """Tests for _fit_proposal: Student-t proposal fitting."""

    def test_fit_proposal_returns_student_t_proposal(self, gaussian_chain):
        """_fit_proposal returns a StudentTProposal with correct shape."""
        proposal = hmc_is._fit_proposal(gaussian_chain, df=5.0, inflation=1.5)
        assert hasattr(proposal, "mean")
        assert hasattr(proposal, "chol")
        assert hasattr(proposal, "df")
        assert proposal.mean.shape == (4,)
        assert proposal.chol.shape == (4, 4)
        assert proposal.df == 5.0

    def test_fit_proposal_covariance_inflation(self, gaussian_chain):
        """Inflated covariance is larger than the empirical chain covariance."""
        proposal_unscaled = hmc_is._fit_proposal(gaussian_chain, df=5.0, inflation=1.0)
        proposal_inflated = hmc_is._fit_proposal(gaussian_chain, df=5.0, inflation=1.5)

        var_unscaled = np.diag(proposal_unscaled.chol @ proposal_unscaled.chol.T)
        var_inflated = np.diag(proposal_inflated.chol @ proposal_inflated.chol.T)

        np.testing.assert_array_less(var_unscaled, var_inflated)

    def test_fit_proposal_spectral_floor(self, gaussian_chain):
        """Spectral floor ensures finite Cholesky on rank-deficient chain."""
        chain_deficient = gaussian_chain.copy()
        chain_deficient[:, 0] = chain_deficient[0, 0]

        proposal = hmc_is._fit_proposal(chain_deficient, df=5.0, inflation=1.5)

        assert np.all(np.isfinite(proposal.mean))
        assert np.all(np.isfinite(proposal.chol))

    def test_fit_proposal_mean_near_chain_mean(self, gaussian_chain):
        """Proposal mean is close to the empirical chain mean."""
        proposal = hmc_is._fit_proposal(gaussian_chain, df=5.0, inflation=1.5)
        chain_mean = gaussian_chain.mean(axis=0)
        np.testing.assert_allclose(proposal.mean, chain_mean, rtol=1e-3)


class TestProposalLogpdf:
    """Tests for _proposal_logpdf: Student-t density evaluation."""

    def test_proposal_logpdf_shape(self, gaussian_chain):
        """_proposal_logpdf returns shape (n,) for n points."""
        proposal = hmc_is._fit_proposal(gaussian_chain, df=5.0, inflation=1.5)
        x = gaussian_chain[:10]
        logpdf = hmc_is._proposal_logpdf(proposal, x)
        assert logpdf.shape == (10,)

    def test_proposal_logpdf_vs_scipy(self, gaussian_chain):
        """_proposal_logpdf agrees with scipy.stats.multivariate_t."""
        proposal = hmc_is._fit_proposal(gaussian_chain, df=5.0, inflation=1.5)
        x_test = gaussian_chain[:20]

        logpdf_ours = np.array(hmc_is._proposal_logpdf(proposal, x_test))

        cov = proposal.chol @ proposal.chol.T
        logpdf_scipy = scipy_stats.multivariate_t.logpdf(
            x_test, loc=np.array(proposal.mean), shape=np.array(cov), df=proposal.df
        )

        np.testing.assert_allclose(logpdf_ours, logpdf_scipy, rtol=1e-5)


class TestIsLogEvidence:
    """Tests for _is_log_evidence: importance sampling evidence estimation."""

    def test_is_log_evidence_gaussian_analytic(self, gaussian_target, gaussian_chain):
        """IS evidence on analytic Gaussian target is within 3-sigma."""
        log_target, _, _, _, log_evidence_true = gaussian_target
        key = jax.random.PRNGKey(303)
        proposal = hmc_is._fit_proposal(gaussian_chain, df=5.0, inflation=1.5)

        log_z, log_z_err, ess, max_weight_frac = hmc_is._is_log_evidence(
            log_target, key, proposal, n_draws=20_000, chunk_size=4096
        )

        assert np.isfinite(log_z)
        assert np.isfinite(log_z_err)
        assert np.isfinite(ess)
        assert np.isfinite(max_weight_frac)

        error_tolerance = 3 * log_z_err + 0.02
        assert abs(log_z - log_evidence_true) < error_tolerance

        assert log_z_err < 0.05
        assert ess > 2000
        assert max_weight_frac < 0.05

    def test_is_log_evidence_chunking_reproducibility(self, gaussian_target, gaussian_chain):
        """Chunking does not change the log_z estimate with the same key."""
        log_target, _, _, _, _ = gaussian_target
        key = jax.random.PRNGKey(404)
        proposal = hmc_is._fit_proposal(gaussian_chain, df=5.0, inflation=1.5)

        result_big = hmc_is._is_log_evidence(
            log_target, key, proposal, n_draws=4096, chunk_size=4096
        )

        result_chunked = hmc_is._is_log_evidence(
            log_target, key, proposal, n_draws=4096, chunk_size=512
        )

        np.testing.assert_allclose(result_big[0], result_chunked[0], rtol=1e-10, atol=1e-10)

    def test_is_log_evidence_all_outputs_finite(self, gaussian_target, gaussian_chain):
        """All outputs of _is_log_evidence are finite."""
        log_target, _, _, _, _ = gaussian_target
        key = jax.random.PRNGKey(505)
        proposal = hmc_is._fit_proposal(gaussian_chain, df=5.0, inflation=1.5)

        log_z, log_z_err, ess, max_weight_frac = hmc_is._is_log_evidence(
            log_target, key, proposal, n_draws=5000, chunk_size=1024
        )

        assert np.all(np.isfinite([log_z, log_z_err, ess, max_weight_frac]))

    def test_is_log_evidence_ess_reasonable(self, gaussian_target, gaussian_chain):
        """ESS is a reasonable fraction of n_draws."""
        log_target, _, _, _, _ = gaussian_target
        key = jax.random.PRNGKey(606)
        proposal = hmc_is._fit_proposal(gaussian_chain, df=5.0, inflation=1.5)
        n_draws = 10000

        _, _, ess, _ = hmc_is._is_log_evidence(
            log_target, key, proposal, n_draws=n_draws, chunk_size=2048
        )

        assert 0 < ess <= n_draws

    def test_is_log_evidence_max_weight_frac_bounded(self, gaussian_target, gaussian_chain):
        """Maximum weight fraction is in [0, 1]."""
        log_target, _, _, _, _ = gaussian_target
        key = jax.random.PRNGKey(707)
        proposal = hmc_is._fit_proposal(gaussian_chain, df=5.0, inflation=1.5)

        _, _, _, max_weight_frac = hmc_is._is_log_evidence(
            log_target, key, proposal, n_draws=5000, chunk_size=1024
        )

        assert 0 <= max_weight_frac <= 1
