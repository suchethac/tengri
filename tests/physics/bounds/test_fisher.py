# SPDX-License-Identifier: BSD-3-Clause
"""Tests for tengri.analysis.diagnostics.fisher.

Tests cover the pure-math functions (no forward model required):
- fisher_parameter_errors: marginal uncertainties from FIM inverse
- fisher_correlation_matrix: normalised covariance in [-1, 1]

compute_jacobian and compute_fisher_matrix require a forward-model instance and
are not tested here (integration-level concern).
"""

import chex
import pytest

pytestmark = pytest.mark.bounds

import jax.numpy as jnp

from tengri.analysis.diagnostics.fisher import fisher_correlation_matrix, fisher_parameter_errors

# ── Helpers ───────────────────────────────────────────────────────


def _diagonal_fim(variances):
    """Return a diagonal FIM from a list/array of inverse-variances (σ⁻²)."""
    return jnp.diag(jnp.array(variances, dtype=float))


# ── fisher_parameter_errors ───────────────────────────────────────


class TestFisherParameterErrors:
    def test_diagonal_fim_recovers_sigmas(self):
        """For diagonal FIM with values 1/σ², errors should equal σ."""
        sigmas = jnp.array([0.5, 1.0, 2.0, 0.1])
        fim = _diagonal_fim(1.0 / sigmas**2)
        errors = fisher_parameter_errors(fim)
        assert jnp.allclose(errors, sigmas, rtol=1e-5)

    def test_output_shape(self):
        n = 5
        fim = jnp.eye(n) * 4.0  # F = 4I → σ = 0.5
        errors = fisher_parameter_errors(fim)
        chex.assert_shape(errors, (n,))

    def test_errors_positive(self):
        """Errors must be strictly positive."""
        fim = jnp.eye(4) * 9.0
        errors = fisher_parameter_errors(fim)
        assert jnp.all(errors > 0)

    def test_identity_fim_gives_unit_errors(self):
        """F = I (information 1 per parameter) → σ = 1."""
        fim = jnp.eye(3)
        errors = fisher_parameter_errors(fim)
        assert jnp.allclose(errors, 1.0, rtol=1e-5)

    def test_larger_fim_gives_smaller_errors(self):
        """More Fisher information → tighter constraints."""
        fim_tight = jnp.eye(3) * 100.0
        fim_loose = jnp.eye(3) * 1.0
        errors_tight = fisher_parameter_errors(fim_tight)
        errors_loose = fisher_parameter_errors(fim_loose)
        assert jnp.all(errors_tight < errors_loose)

    def test_correlated_fim_finite(self):
        """Full (non-diagonal) positive-definite FIM should return finite errors."""
        # Construct a 3×3 positive-definite matrix via A^T A
        a = jnp.array([[2.0, 1.0, 0.0], [0.5, 3.0, 0.5], [0.0, 0.2, 1.5]])
        fim = a.T @ a
        errors = fisher_parameter_errors(fim)
        chex.assert_tree_all_finite(errors)
        assert jnp.all(errors > 0)

    def test_output_is_real_valued(self):
        fim = jnp.eye(4) * 2.0
        errors = fisher_parameter_errors(fim)
        # No imaginary part (sqrt of positive diagonal of positive-definite matrix)
        assert jnp.all(jnp.isreal(errors))


# ── fisher_correlation_matrix ─────────────────────────────────────


class TestFisherCorrelationMatrix:
    def test_diagonal_fim_gives_identity_correlation(self):
        """Uncorrelated parameters → correlation matrix = I."""
        fim = _diagonal_fim([1.0, 4.0, 9.0])
        corr = fisher_correlation_matrix(fim)
        assert jnp.allclose(corr, jnp.eye(3), atol=1e-5)

    def test_diagonal_values_are_one(self):
        """Diagonal of correlation matrix is always 1."""
        a = jnp.array([[3.0, 1.0, 0.0], [0.5, 2.0, 0.5], [0.1, 0.3, 2.0]])
        fim = a.T @ a
        corr = fisher_correlation_matrix(fim)
        assert jnp.allclose(jnp.diag(corr), 1.0, atol=1e-5)

    def test_output_shape(self):
        n = 4
        fim = jnp.eye(n) * 2.0
        corr = fisher_correlation_matrix(fim)
        chex.assert_shape(corr, (n, n))

    def test_values_in_minus_one_to_one(self):
        """All correlation values must lie in [-1, 1]."""
        a = jnp.array([[4.0, 2.0, 1.0], [0.5, 3.0, 0.8], [0.2, 0.6, 2.5]])
        fim = a.T @ a
        corr = fisher_correlation_matrix(fim)
        assert jnp.all(corr >= -1.0 - 1e-5)
        assert jnp.all(corr <= 1.0 + 1e-5)

    def test_symmetric(self):
        """Correlation matrix must be symmetric."""
        a = jnp.array([[2.0, 0.5, 0.1], [0.0, 3.0, 0.4], [0.2, 0.1, 1.5]])
        fim = a.T @ a
        corr = fisher_correlation_matrix(fim)
        assert jnp.allclose(corr, corr.T, atol=1e-5)

    def test_maximally_correlated_two_param(self):
        """Two perfectly correlated params have off-diagonal = ±1."""
        # Build covariance with ρ = 0.999
        rho = 0.999
        cov = jnp.array([[1.0, rho], [rho, 1.0]])
        fim = jnp.linalg.inv(cov)
        corr = fisher_correlation_matrix(fim)
        assert jnp.allclose(jnp.abs(corr[0, 1]), 1.0, atol=1e-2)

    def test_identity_fim_is_identity_correlation(self):
        fim = jnp.eye(5)
        corr = fisher_correlation_matrix(fim)
        assert jnp.allclose(corr, jnp.eye(5), atol=1e-5)

    def test_positive_definite_fim(self):
        """Result should be finite for any positive-definite FIM."""
        # 4×4 PD matrix
        a = jnp.array(
            [
                [5.0, 1.0, 0.5, 0.1],
                [0.2, 4.0, 0.3, 0.4],
                [0.1, 0.2, 3.0, 0.6],
                [0.3, 0.1, 0.2, 2.0],
            ]
        )
        fim = a.T @ a
        corr = fisher_correlation_matrix(fim)
        chex.assert_tree_all_finite(corr)
