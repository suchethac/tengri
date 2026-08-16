# SPDX-License-Identifier: BSD-3-Clause
"""#1355: partially non-centered DRW field — the math primitive.

The field is non-centered today: ``s = sigma_s * L(tau) * xi`` with ``xi ~ N(0, I)``, which
is exactly what the standardized parameterization means for this block. The multiplicative
coupling between ``sigma`` and the latents is the funnel measured on this posterior
(hyperparameter-to-field curvature correlation 0.93-0.99 in every configuration).

Partial non-centering (Papaspiliopoulos, Roberts & Skold, Stat. Sci. 22, 59, 2007)
interpolates with an exponent ``a`` in [0, 1]:

    s = sigma_s**a * L(tau) @ zeta ,     zeta ~ N(0, sigma_s**(2 - 2a) I)

``a = 1`` is today's non-centered form; ``a = 0`` is fully centered (the prior carries
sigma). ``a`` is a build-time structural choice, not a sampled parameter.

The property that matters is **a-invariance**: ``a`` is a change of variables, so the
posterior over ``(sigma, s)`` must not depend on it. That holds only if the latent
log-prior carries the ``-n(1-a) log sigma_s`` normalizer. Dropping that term leaves a
perfectly runnable sampler that quietly targets a different distribution for every ``a`` —
no error, no warning, wrong science. It is pinned first and hardest here.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

LN10 = float(np.log(10.0))


def _grid(n=16, lo=6.0, hi=10.1):
    return jnp.linspace(lo, hi, n)


def _dense_K(sigma_dex, tau_yr, grid):
    """Reference DRW covariance, built densely and independently of the recursion."""
    t = 10.0 ** np.asarray(grid)
    var = (sigma_dex * LN10) ** 2
    return var * np.exp(-np.abs(t[:, None] - t[None, :]) / tau_yr)


def _dense_C(tau_yr, grid):
    """Unit-variance correlation matrix."""
    t = 10.0 ** np.asarray(grid)
    return np.exp(-np.abs(t[:, None] - t[None, :]) / tau_yr)


class TestUnitVarianceRealization:
    def test_unit_realization_is_a_square_root_of_the_correlation_matrix(self):
        from tengri.components.stellar.sfh.gp_sfh import drw_unit_gp_from_xi

        grid, tau = _grid(24), 8e7
        n = grid.shape[0]
        # Columns of the implied L~ come from unit basis vectors.
        cols = [np.asarray(drw_unit_gp_from_xi(jnp.eye(n)[k], tau, grid)) for k in range(n)]
        L = np.stack(cols, axis=1)
        np.testing.assert_allclose(L @ L.T, _dense_C(tau, grid), rtol=0, atol=1e-12)

    def test_unit_realization_has_unit_marginal_variance(self):
        from tengri.components.stellar.sfh.gp_sfh import drw_unit_gp_from_xi

        grid, tau = _grid(12), 5e7
        n = grid.shape[0]
        L = np.stack(
            [np.asarray(drw_unit_gp_from_xi(jnp.eye(n)[k], tau, grid)) for k in range(n)], axis=1
        )
        np.testing.assert_allclose(np.diag(L @ L.T), np.ones(n), rtol=0, atol=1e-12)


class TestPartialTransform:
    def test_a_equals_one_reproduces_the_production_non_centered_field(self):
        from tengri.components.stellar.sfh.gp_sfh import (
            drw_innovations_gp_from_xi,
            drw_partial_gp_from_zeta,
        )

        grid, sigma, tau = _grid(20), 0.45, 7e7
        xi = jnp.asarray(np.random.default_rng(0).standard_normal(20))
        want, want_k0 = drw_innovations_gp_from_xi(xi, sigma, tau, grid)
        got, got_k0 = drw_partial_gp_from_zeta(xi, sigma, tau, grid, centering=1.0)
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-14, atol=0)
        assert float(got_k0) == pytest.approx(float(want_k0), rel=1e-14)

    @pytest.mark.parametrize("a", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_transform_times_latent_covariance_reconstructs_K(self, a):
        """(sigma^a L~) Cov(zeta) (sigma^a L~)^T must equal K for every a."""
        from tengri.components.stellar.sfh.gp_sfh import drw_partial_gp_from_zeta

        grid, sigma, tau = _grid(16), 0.6, 6e7
        n = grid.shape[0]
        sigma_s = sigma * LN10
        A = np.stack(
            [
                np.asarray(
                    drw_partial_gp_from_zeta(jnp.eye(n)[k], sigma, tau, grid, centering=a)[0]
                )
                for k in range(n)
            ],
            axis=1,
        )
        cov_zeta = sigma_s ** (2.0 - 2.0 * a) * np.eye(n)
        np.testing.assert_allclose(
            A @ cov_zeta @ A.T, _dense_K(sigma, tau, grid), rtol=1e-10, atol=0
        )


class TestLatentLogPrior:
    @pytest.mark.parametrize("a", [0.0, 0.3, 1.0])
    def test_log_prior_matches_the_exact_gaussian_density(self, a):
        from tengri.components.stellar.sfh.gp_sfh import drw_latent_log_prior

        rng = np.random.default_rng(3)
        n, sigma = 10, 0.5
        sigma_s = sigma * LN10
        zeta = rng.standard_normal(n)
        var = sigma_s ** (2.0 - 2.0 * a)
        want = -0.5 * float(zeta @ zeta) / var - 0.5 * n * float(np.log(2 * np.pi * var))
        got = float(drw_latent_log_prior(jnp.asarray(zeta), sigma, centering=a))
        assert got == pytest.approx(want, rel=1e-12)

    def test_a_equals_one_reduces_to_the_standardized_quadratic(self):
        """At a=1 the prior must be the plain -1/2 zeta^T zeta of standardized space."""
        from tengri.components.stellar.sfh.gp_sfh import drw_latent_log_prior

        zeta = jnp.asarray(np.random.default_rng(4).standard_normal(9))
        got = float(drw_latent_log_prior(zeta, 0.37, centering=1.0))
        want = -0.5 * float(zeta @ zeta) - 0.5 * 9 * float(np.log(2 * np.pi))
        assert got == pytest.approx(want, rel=1e-12)


class TestAInvariance:
    """The whole point: ``a`` is a reparameterization, so the posterior cannot move."""

    @pytest.mark.parametrize("a", [0.0, 0.25, 0.5, 0.75])
    def test_joint_density_over_sigma_and_s_is_independent_of_a(self, a):
        from tengri.components.stellar.sfh.gp_sfh import (
            drw_latent_log_prior,
            drw_partial_gp_from_zeta,
        )

        grid, sigma, tau = _grid(14), 0.55, 9e7
        n = grid.shape[0]
        sigma_s = sigma * LN10
        rng = np.random.default_rng(7)
        # Fix a field realization s, then ask each parameterization for its own latent.
        s = np.asarray(
            drw_partial_gp_from_zeta(
                jnp.asarray(rng.standard_normal(n)), sigma, tau, grid, centering=1.0
            )[0]
        )

        def latent_for(exponent):
            A = np.stack(
                [
                    np.asarray(
                        drw_partial_gp_from_zeta(
                            jnp.eye(n)[k], sigma, tau, grid, centering=exponent
                        )[0]
                    )
                    for k in range(n)
                ],
                axis=1,
            )
            return np.linalg.solve(A, s), A

        z_ref, A_ref = latent_for(1.0)
        z_a, A_a = latent_for(a)

        # log p(s) = log p(zeta) - log|det A|  (change of variables). Must match across a.
        ref = float(drw_latent_log_prior(jnp.asarray(z_ref), sigma, centering=1.0)) - float(
            np.linalg.slogdet(A_ref)[1]
        )
        got = float(drw_latent_log_prior(jnp.asarray(z_a), sigma, centering=a)) - float(
            np.linalg.slogdet(A_a)[1]
        )
        assert got == pytest.approx(ref, rel=1e-9), (
            f"a={a} changes the density on s by {got - ref:.3e} — the "
            "-n(1-a)log(sigma_s) normalizer is missing or wrong"
        )

    def test_dropping_the_normalizer_would_break_invariance(self):
        """Neuter check: the guard above must actually be sensitive to that term."""
        from tengri.components.stellar.sfh.gp_sfh import drw_latent_log_prior

        n, sigma, a = 12, 0.5, 0.0
        sigma_s = sigma * LN10
        zeta = jnp.asarray(np.random.default_rng(11).standard_normal(n))
        full = float(drw_latent_log_prior(zeta, sigma, centering=a))
        without = -0.5 * float(zeta @ zeta) / sigma_s ** (2 - 2 * a) - 0.5 * n * float(
            np.log(2 * np.pi)
        )
        assert abs(full - without) > 1.0, (
            "the normalizer contributes < 1 nat here, so the invariance test above "
            "could not detect its absence"
        )


class TestNoDenseMatrix:
    def test_scales_to_a_large_grid_without_building_n_by_n(self):
        """O(n): a dense n x n at n=20000 would be 3.2 GB in f64."""
        from tengri.components.stellar.sfh.gp_sfh import drw_partial_gp_from_zeta

        n = 20000
        grid = _grid(n)
        zeta = jnp.asarray(np.random.default_rng(0).standard_normal(n))
        out, _ = drw_partial_gp_from_zeta(zeta, 0.4, 7e7, grid, centering=0.5)
        assert out.shape == (n,)
        assert bool(jnp.all(jnp.isfinite(out)))
