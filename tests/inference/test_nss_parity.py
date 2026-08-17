# SPDX-License-Identifier: BSD-3-Clause
"""Parity tests: verify tengri.inference.ns matches handley-lab/blackjax.

These tests require the handley-lab blackjax fork to be installed:
    pip install "blackjax @ git+https://github.com/handley-lab/blackjax@nested_sampling"

They are skipped when the fork is not available (standard blackjax installed).

Tests verify:
  1. Identical results (bitwise) for the same random seed
  2. Identical evidence estimates
  3. Comparable performance (wall time within 2x)
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Skip all tests in this module if the handley fork is not installed
try:
    from blackjax.ns.nss import as_top_level_api as bj_as_top_level_api
    from blackjax.ns.utils import ess as bj_ess, finalize as bj_finalize, sample as bj_sample

    HAS_HANDLEY_FORK = True
except (ImportError, ModuleNotFoundError):
    HAS_HANDLEY_FORK = False

pytestmark = [
    pytest.mark.skipif(
        not HAS_HANDLEY_FORK, reason="Requires handley-lab/blackjax fork (nested_sampling branch)"
    ),
    pytest.mark.contract,
]


# ── Test fixtures ────────────────────────────────────────────────────

PRIOR_HALF_WIDTH = 5.0
D = 2
N_LIVE = 100
NUM_DELETE = 5
NUM_INNER_STEPS = D
N_ITER = 50  # fixed number of iterations for parity


def _make_functions(w=PRIOR_HALF_WIDTH, d=D):
    """Create logprior and loglikelihood functions."""
    names = [f"x{i}" for i in range(d)]

    def logprior_fn(params):
        lp = 0.0
        for k in names:
            lp = lp + jax.scipy.stats.uniform.logpdf(params[k], -w, 2 * w)
        return lp

    def loglik_fn(params):
        x = jnp.array([params[k] for k in names])
        return -0.5 * jnp.sum(x**2)

    return logprior_fn, loglik_fn, names


def _make_particles(key, names, w, n_live):
    """Create initial particles from uniform prior."""

    def prior_sample(k):
        ks = jax.random.split(k, len(names))
        return {
            name: jax.random.uniform(ks[i], minval=-w, maxval=w) for i, name in enumerate(names)
        }

    init_keys = jax.random.split(key, n_live)
    return jax.vmap(prior_sample)(init_keys)


# ── Parity tests ─────────────────────────────────────────────────────


class TestNSSParity:
    """Verify bitwise-identical results between local implementation and handley fork."""

    def test_identical_init(self):
        """init() produces identical states."""
        from tengri.inference.backends.nested.nss import as_top_level_api as tengri_api

        logprior_fn, loglik_fn, names = _make_functions()

        bj_algo = bj_as_top_level_api(
            logprior_fn, loglik_fn, NUM_INNER_STEPS, num_delete=NUM_DELETE
        )
        tengri_algo = tengri_api(logprior_fn, loglik_fn, NUM_INNER_STEPS, num_delete=NUM_DELETE)

        key = jax.random.PRNGKey(42)
        particles = _make_particles(key, names, PRIOR_HALF_WIDTH, N_LIVE)

        bj_state = bj_algo.init(particles)
        tengri_state = tengri_algo.init(particles)

        # Compare loglikelihoods
        np.testing.assert_array_equal(
            np.array(bj_state.particles.loglikelihood),
            np.array(tengri_state.particles.loglikelihood),
        )
        # Compare logdensities (priors)
        np.testing.assert_array_equal(
            np.array(bj_state.particles.logdensity),
            np.array(tengri_state.particles.logdensity),
        )
        # Compare integrator
        np.testing.assert_array_equal(
            np.array(bj_state.integrator.logZ),
            np.array(tengri_state.integrator.logZ),
        )

    def test_identical_steps(self):
        """step() produces identical results for the same key."""
        from tengri.inference.backends.nested.nss import as_top_level_api as tengri_api

        logprior_fn, loglik_fn, names = _make_functions()

        bj_algo = bj_as_top_level_api(
            logprior_fn, loglik_fn, NUM_INNER_STEPS, num_delete=NUM_DELETE
        )
        tengri_algo = tengri_api(logprior_fn, loglik_fn, NUM_INNER_STEPS, num_delete=NUM_DELETE)

        key = jax.random.PRNGKey(42)
        particles = _make_particles(key, names, PRIOR_HALF_WIDTH, N_LIVE)

        bj_state = bj_algo.init(particles)
        tengri_state = tengri_algo.init(particles)

        # Run N_ITER steps with the same keys
        key = jax.random.PRNGKey(0)
        for _i in range(N_ITER):
            key, subkey = jax.random.split(key)
            bj_state, bj_info = bj_algo.step(subkey, bj_state)
            tengri_state, tengri_info = tengri_algo.step(subkey, tengri_state)

            # Compare live particle loglikelihoods
            np.testing.assert_array_equal(
                np.array(bj_state.particles.loglikelihood),
                np.array(tengri_state.particles.loglikelihood),
            )
            # Compare dead particle loglikelihoods
            np.testing.assert_array_equal(
                np.array(bj_info.particles.loglikelihood),
                np.array(tengri_info.particles.loglikelihood),
            )
            # Compare integrator evidence
            np.testing.assert_allclose(
                float(bj_state.integrator.logZ),
                float(tengri_state.integrator.logZ),
                atol=1e-12,
            )

    def test_identical_evidence(self):
        """Full runs produce identical log-evidence."""
        from tengri.inference.backends.nested.nss import as_top_level_api as tengri_api

        logprior_fn, loglik_fn, names = _make_functions()

        bj_algo = bj_as_top_level_api(
            logprior_fn, loglik_fn, NUM_INNER_STEPS, num_delete=NUM_DELETE
        )
        tengri_algo = tengri_api(logprior_fn, loglik_fn, NUM_INNER_STEPS, num_delete=NUM_DELETE)

        key = jax.random.PRNGKey(42)
        particles = _make_particles(key, names, PRIOR_HALF_WIDTH, N_LIVE)

        bj_state = bj_algo.init(particles)
        tengri_state = tengri_algo.init(particles)

        # Fixed iteration count for reproducibility
        key = jax.random.PRNGKey(0)
        for _ in range(N_ITER):
            key, subkey = jax.random.split(key)
            bj_state, _ = bj_algo.step(subkey, bj_state)
            tengri_state, _ = tengri_algo.step(subkey, tengri_state)

        bj_logZ = float(jnp.logaddexp(bj_state.integrator.logZ, bj_state.integrator.logZ_live))
        tengri_logZ = float(
            jnp.logaddexp(tengri_state.integrator.logZ, tengri_state.integrator.logZ_live)
        )

        np.testing.assert_allclose(bj_logZ, tengri_logZ, atol=1e-12)

    def test_identical_post_processing(self):
        """finalize + sample + ess produce identical results."""
        from tengri.inference.backends.nested.nss import as_top_level_api as tengri_api
        from tengri.inference.backends.nested.utils import (
            ess as tengri_ess,
            finalize as tengri_finalize,
            sample as tengri_sample,
        )

        logprior_fn, loglik_fn, names = _make_functions()

        bj_algo = bj_as_top_level_api(
            logprior_fn, loglik_fn, NUM_INNER_STEPS, num_delete=NUM_DELETE
        )
        tengri_algo = tengri_api(logprior_fn, loglik_fn, NUM_INNER_STEPS, num_delete=NUM_DELETE)

        key = jax.random.PRNGKey(42)
        particles = _make_particles(key, names, PRIOR_HALF_WIDTH, N_LIVE)

        bj_state = bj_algo.init(particles)
        tengri_state = tengri_algo.init(particles)

        bj_dead = []
        tengri_dead = []

        key = jax.random.PRNGKey(0)
        for _ in range(N_ITER):
            key, subkey = jax.random.split(key)
            bj_state, bj_info = bj_algo.step(subkey, bj_state)
            tengri_state, tengri_info = tengri_algo.step(subkey, tengri_state)
            bj_dead.append(bj_info)
            tengri_dead.append(tengri_info)

        # Finalize
        bj_run = bj_finalize(bj_state, bj_dead)
        tengri_run = tengri_finalize(tengri_state, tengri_dead)

        np.testing.assert_array_equal(
            np.array(bj_run.particles.loglikelihood),
            np.array(tengri_run.particles.loglikelihood),
        )

        # ESS
        ess_key = jax.random.PRNGKey(99)
        bj_ess_val = float(bj_ess(ess_key, bj_run))
        tengri_ess_val = float(tengri_ess(ess_key, tengri_run))
        np.testing.assert_allclose(bj_ess_val, tengri_ess_val, atol=1e-6)

        # Sample
        sample_key = jax.random.PRNGKey(123)
        bj_samples = bj_sample(sample_key, bj_run, 200)
        tengri_samples = tengri_sample(sample_key, tengri_run, 200)

        for name in names:
            np.testing.assert_array_equal(
                np.array(bj_samples.position[name]),
                np.array(tengri_samples.position[name]),
            )


class TestNSSPerformanceParity:
    """Verify comparable performance (within 2x)."""

    def test_step_time_comparable(self):
        """Single JIT-compiled step takes comparable time."""
        from tengri.inference.backends.nested.nss import as_top_level_api as tengri_api

        logprior_fn, loglik_fn, names = _make_functions()

        bj_algo = bj_as_top_level_api(
            logprior_fn, loglik_fn, NUM_INNER_STEPS, num_delete=NUM_DELETE
        )
        tengri_algo = tengri_api(logprior_fn, loglik_fn, NUM_INNER_STEPS, num_delete=NUM_DELETE)

        key = jax.random.PRNGKey(42)
        particles = _make_particles(key, names, PRIOR_HALF_WIDTH, N_LIVE)

        bj_state = bj_algo.init(particles)
        tengri_state = tengri_algo.init(particles)

        # Warmup JIT
        key, k1, k2 = jax.random.split(key, 3)
        bj_step = jax.jit(bj_algo.step)
        tengri_step = jax.jit(tengri_algo.step)
        bj_state, _ = bj_step(k1, bj_state)
        tengri_state, _ = tengri_step(k1, tengri_state)

        # Time N steps
        n_timed = 20
        keys = jax.random.split(k2, n_timed)

        t0 = time.perf_counter()
        for k in keys:
            bj_state, _ = bj_step(k, bj_state)
        jax.block_until_ready(bj_state.particles.loglikelihood)
        bj_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        for k in keys:
            tengri_state, _ = tengri_step(k, tengri_state)
        jax.block_until_ready(tengri_state.particles.loglikelihood)
        tengri_time = time.perf_counter() - t0

        ratio = tengri_time / max(bj_time, 1e-6)
        print(f"blackjax: {bj_time:.3f}s, tengri: {tengri_time:.3f}s, ratio: {ratio:.2f}")
        assert ratio < 2.0, f"tengri is {ratio:.1f}x slower than blackjax fork"
