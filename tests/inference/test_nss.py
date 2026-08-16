# SPDX-License-Identifier: BSD-3-Clause
"""Tests for Nested Slice Sampling integration.

Tests the local NS implementation (tengri.inference.ns) against known analytic
evidence values for simple targets, verifies guards, and checks
Posterior.log_evidence.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract


import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import erf

# ── Helpers ──────────────────────────────────────────────────────────


def _analytic_logZ_gaussian(D, prior_half_width=5.0):
    """Analytic log-evidence for a D-dim unit Gaussian with Uniform prior.

    Prior: Uniform(-w, w) per dimension → π(x) = 1/(2w)^D.
    Likelihood: L(x) = exp(-0.5 * ||x||^2).
    Evidence: Z = (sqrt(2π) * erf(w/sqrt(2)) / (2w))^D.
    """
    w = prior_half_width
    return D * float(np.log(np.sqrt(2 * np.pi) * erf(w / np.sqrt(2)) / (2 * w)))


def _make_gaussian_nss(D, n_live=200, num_delete=10, prior_half_width=5.0):
    """Set up NSS for a D-dim unit Gaussian target."""
    from tengri.inference.backends.nested.nss import as_top_level_api

    w = prior_half_width
    names = [f"x{i}" for i in range(D)]

    def logprior_fn(params):
        lp = 0.0
        for k in names:
            lp = lp + jax.scipy.stats.uniform.logpdf(params[k], -w, 2 * w)
        return lp

    def loglik_fn(params):
        x = jnp.array([params[k] for k in names])
        return -0.5 * jnp.sum(x**2)

    algo = as_top_level_api(
        logprior_fn,
        loglik_fn,
        num_inner_steps=D,
        num_delete=num_delete,
    )
    return algo, names, w


def _run_nss_to_convergence(algo, names, w, n_live, num_delete, key, tol=-3.0, max_iter=500):
    """Run NSS until convergence, return (logZ, n_iter)."""

    init_keys = jax.random.split(key, n_live)

    def prior_sample(k):
        ks = jax.random.split(k, len(names))
        return {
            name: jax.random.uniform(ks[i], minval=-w, maxval=w) for i, name in enumerate(names)
        }

    particles = jax.vmap(prior_sample)(init_keys)
    live = algo.init(particles)
    step = jax.jit(algo.step)

    dead_points = []
    n_iter = 0
    for _i in range(1, max_iter + 1):
        key, subkey = jax.random.split(key)
        live, dead = step(subkey, live)
        dead_points.append(dead)
        n_iter = _i

        remaining = float(live.integrator.logZ_live - live.integrator.logZ)
        if remaining < tol:
            break

    logZ = float(jnp.logaddexp(live.integrator.logZ, live.integrator.logZ_live))
    return logZ, n_iter, live, dead_points


# ── Posterior.log_evidence ───────────────────────────────────────────


class TestPosteriorLogEvidence:
    """Test the log_evidence field on Posterior."""

    def test_default_none(self):
        """log_evidence defaults to None."""
        from tengri.inference.posterior import Posterior

        post = Posterior(
            samples=None,
            params={},
            method="test",
            wall_time_s=0.0,
            diagnostics={},
        )
        assert post.log_evidence is None

    def test_set_evidence(self):
        """log_evidence can be set."""
        from tengri.inference.posterior import Posterior

        post = Posterior(
            samples=None,
            params={},
            method="test",
            wall_time_s=0.0,
            diagnostics={},
            log_evidence=-42.0,
        )
        assert post.log_evidence == -42.0

    def test_summary_table_includes_evidence(self):
        """summary_table prints evidence when present."""
        from tengri.inference.posterior import Posterior

        post = Posterior(
            samples={"x": jnp.ones(10)},
            params={"x": jnp.array(1.0)},
            method="NSS",
            wall_time_s=1.0,
            diagnostics={"log_evidence_err": 0.5},
            log_evidence=-42.0,
        )
        table = post.summary_table()
        assert "log Z" in table
        assert "-42.00" in table

    def test_summary_table_no_evidence(self):
        """summary_table omits evidence line when None."""
        from tengri.inference.posterior import Posterior

        post = Posterior(
            samples={"x": jnp.ones(10)},
            params={"x": jnp.array(1.0)},
            method="NUTS",
            wall_time_s=1.0,
            diagnostics={},
        )
        table = post.summary_table()
        assert "log Z" not in table


# ── NS components ───────────────────────────────────────────────────


class TestSliceSampling:
    """Test the Hit-and-Run Slice Sampling kernel."""

    def test_hrss_samples_from_target(self):
        """HRSS generates samples near a 1D Gaussian mode."""
        from tengri.inference.backends.nested.slice_sampling import hrss_as_top_level_api

        def logdensity(position):
            x = position["x"]
            return -0.5 * x**2

        cov = jnp.eye(1)
        algo = hrss_as_top_level_api(logdensity, cov)
        state = algo.init({"x": jnp.array(0.0)})

        key = jax.random.PRNGKey(0)
        samples = []
        for _i in range(80):
            key, subkey = jax.random.split(key)
            state, _info = algo.step(subkey, state)
            samples.append(float(state.position["x"]))

        samples = np.array(samples)
        assert abs(np.mean(samples)) < 0.5
        assert 0.3 < np.std(samples) < 2.0

    def test_direction_has_correct_norm(self):
        """Direction proposal from covariance is Mahalanobis-normalized."""
        from tengri.inference.backends.nested.slice_sampling import (
            sample_direction_from_covariance,
        )

        cov = jnp.diag(jnp.array([1.0, 4.0, 9.0]))
        key = jax.random.PRNGKey(42)
        pos = {"a": jnp.array(0.0), "b": jnp.array(0.0), "c": jnp.array(0.0)}
        from jax.flatten_util import ravel_pytree

        d = sample_direction_from_covariance(key, pos, cov)
        d_flat, _ = ravel_pytree(d)
        invcov = jnp.linalg.inv(cov)
        mahal = jnp.sqrt(d_flat @ invcov @ d_flat)
        # Normalized to Mahalanobis norm 2 (scaled by 2 in the code)
        np.testing.assert_allclose(float(mahal), 2.0, atol=1e-10)


class TestParticles:
    """Test particle statistics."""

    def test_covariance_matches_numpy(self):
        """particles_covariance_matrix matches numpy (ddof=0)."""
        from tengri.inference.backends.nested.particles import particles_covariance_matrix

        key = jax.random.PRNGKey(0)
        n = 50
        particles = {"a": jax.random.normal(key, (n,)), "b": jax.random.normal(key, (n,))}
        cov = particles_covariance_matrix(particles)

        data = np.column_stack([np.array(particles["a"]), np.array(particles["b"])])
        expected = np.cov(data, ddof=0, rowvar=False)
        np.testing.assert_allclose(np.array(cov), expected, atol=1e-10)


class TestIntegrator:
    """Test the evidence integrator."""

    def test_init_integrator(self):
        """Integrator initializes with logX=0, logZ=-inf."""
        from tengri.inference.backends.nested.base import StateWithLogLikelihood
        from tengri.inference.backends.nested.integrator import init_integrator

        particles = StateWithLogLikelihood(
            position={"x": jnp.ones(10)},
            logdensity=jnp.zeros(10),
            loglikelihood=-jnp.ones(10),
            loglikelihood_birth=jnp.full(10, jnp.nan),
        )
        integ = init_integrator(particles)
        assert float(integ.logX) == 0.0
        assert float(integ.logZ) == -jnp.inf


# ── Evidence computation ─────────────────────────────────────────────


class TestNSSEvidence:
    """Test NSS evidence estimates against analytic values."""

    @pytest.mark.parametrize("D", [1, 2, 3])
    def test_gaussian_evidence(self, D):
        """NSS recovers correct evidence for a D-dim unit Gaussian."""
        n_live = 200
        num_delete = 10
        algo, names, w = _make_gaussian_nss(D, n_live, num_delete)

        key = jax.random.PRNGKey(42 + D)
        logZ, _n_iter, _, _ = _run_nss_to_convergence(algo, names, w, n_live, num_delete, key)

        true_logZ = _analytic_logZ_gaussian(D)
        # Allow 1 nat tolerance (stochastic algorithm)
        assert abs(logZ - true_logZ) < 1.0, (
            f"D={D}: logZ={logZ:.3f}, true={true_logZ:.3f}, error={abs(logZ - true_logZ):.3f}"
        )

    def test_converges_in_finite_iterations(self):
        """NSS terminates before max_iter for a simple problem."""
        D = 2
        n_live = 200
        num_delete = 10
        algo, names, w = _make_gaussian_nss(D, n_live, num_delete)

        key = jax.random.PRNGKey(0)
        _, n_iter, _, _ = _run_nss_to_convergence(
            algo, names, w, n_live, num_delete, key, max_iter=500
        )
        assert n_iter < 500

    def test_different_seeds_consistent(self):
        """Different seeds produce similar evidence estimates."""
        D = 2
        n_live = 200
        num_delete = 10
        algo, names, w = _make_gaussian_nss(D, n_live, num_delete)

        logZ1, *_ = _run_nss_to_convergence(
            algo, names, w, n_live, num_delete, jax.random.PRNGKey(0)
        )
        logZ2, *_ = _run_nss_to_convergence(
            algo, names, w, n_live, num_delete, jax.random.PRNGKey(1)
        )
        # Should agree within ~2 nats for n_live=200
        assert abs(logZ1 - logZ2) < 2.0


class TestNSSPostProcessing:
    """Test finalize, sample, ess utilities."""

    def test_finalize_and_sample(self):
        """finalize + sample produce resampled particles."""
        from tengri.inference.backends.nested.utils import finalize, sample

        D = 2
        n_live = 100
        num_delete = 5
        algo, names, w = _make_gaussian_nss(D, n_live, num_delete)

        key = jax.random.PRNGKey(42)
        _, _, live, dead_points = _run_nss_to_convergence(algo, names, w, n_live, num_delete, key)

        ns_run = finalize(live, dead_points)
        key, sample_key = jax.random.split(key)
        resampled = sample(sample_key, ns_run, 500)

        # Should have 500 samples
        assert resampled.position["x0"].shape == (500,)
        assert resampled.position["x1"].shape == (500,)

        # Posterior mean should be near 0 for symmetric Gaussian
        for name in names:
            mean = float(jnp.mean(resampled.position[name]))
            assert abs(mean) < 1.0, f"{name}: mean={mean}"

    def test_ess_positive(self):
        """ESS is positive."""
        from tengri.inference.backends.nested.utils import ess, finalize

        D = 2
        n_live = 100
        num_delete = 5
        algo, names, w = _make_gaussian_nss(D, n_live, num_delete)

        key = jax.random.PRNGKey(42)
        _, _, live, dead_points = _run_nss_to_convergence(algo, names, w, n_live, num_delete, key)

        ns_run = finalize(live, dead_points)
        key, ess_key = jax.random.split(key)
        ess_val = float(ess(ess_key, ns_run))
        assert ess_val > 0


class TestNSSUtils:
    """Test utility functions."""

    def test_log1mexp(self):
        """log1mexp matches direct computation."""
        from tengri.inference.backends.nested.utils import log1mexp

        x = jnp.array([-2.0, -1.0, -0.5, -0.1])
        result = log1mexp(x)
        expected = jnp.log(1 - jnp.exp(x))
        np.testing.assert_allclose(np.array(result), np.array(expected), atol=1e-10)

    def test_uniform_prior(self):
        """uniform_prior generates samples within bounds."""
        from tengri.inference.backends.nested.utils import uniform_prior

        key = jax.random.PRNGKey(0)
        bounds = {"a": (-2.0, 3.0), "b": (0.0, 1.0)}
        particles, logprior_fn = uniform_prior(key, 100, bounds)

        assert particles["a"].shape == (100,)
        assert float(jnp.min(particles["a"])) >= -2.0
        assert float(jnp.max(particles["a"])) <= 3.0
        assert float(jnp.min(particles["b"])) >= 0.0
        assert float(jnp.max(particles["b"])) <= 1.0

        # logprior should be -log(5) - log(1) for a point in bounds
        lp = logprior_fn({"a": jnp.array(0.0), "b": jnp.array(0.5)})
        expected = float(np.log(1 / 5.0) + np.log(1 / 1.0))
        np.testing.assert_allclose(float(lp), expected, atol=1e-10)


# ── Fitter integration (no SSP data needed) ─────────────────────────


class TestFitterNSSGuards:
    """Test that Fitter.run('nss') correctly rejects invalid configs."""

    def test_rejects_stochastic_model(self):
        """NSS raises ValueError for stochastic SFH models."""
        from unittest.mock import MagicMock

        from tengri.inference.fitter import Fitter

        # Create a minimal mock
        mock_model = MagicMock()
        mock_spec = MagicMock()
        mock_spec.stochastic = True
        mock_spec.free_params = ["sfh_alpha"]
        mock_spec.get_fixed_values.return_value = {"redshift": 0.1}
        mock_spec.get_distribution.return_value = MagicMock(bounds=(0.0, 5.0))
        mock_model.spec = mock_spec
        mock_model.observation = None

        fitter = Fitter(mock_model, jnp.ones(5), jnp.ones(5))

        with pytest.raises(ValueError, match="stochastic"):
            fitter.run("nss")
