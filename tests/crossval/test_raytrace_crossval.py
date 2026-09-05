# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validation: tengri Ray Tracer vs Behroozi's reference JAX implementation.

Runs both implementations on a known multivariate Gaussian target and
verifies that chains produce statistically identical results.

Requires the reference implementation at /tmp/ray-tracing-sampler/.
If not found, tests are skipped.

Usage:
    pytest -m crossval tests/crossval/test_raytrace_crossval.py -v
"""

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

# Try to import Behroozi's reference implementation
BEHROOZI_PATH = Path("/tmp/ray-tracing-sampler")
HAS_REFERENCE = BEHROOZI_PATH.exists() and (BEHROOZI_PATH / "raytrace_jax.py").exists()

pytestmark = [
    pytest.mark.crossval,
    pytest.mark.skipif(
        not HAS_REFERENCE,
        reason="Behroozi reference not at /tmp/ray-tracing-sampler",
    ),
]


def _load_behroozi_module():
    """Dynamically import Behroozi's raytrace_jax.py."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "raytrace_jax_ref", BEHROOZI_PATH / "raytrace_jax.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gaussian_target():
    """5D correlated Gaussian target for testing."""
    D = 5
    # Fixed covariance (mildly correlated)
    rng = np.random.default_rng(123)
    L = rng.standard_normal((D, D)) * 0.3
    cov = L @ L.T + np.eye(D) * 0.5
    precision = np.linalg.inv(cov)
    mean = np.zeros(D)

    def log_prob(x):
        diff = x - jnp.array(mean)
        return -0.5 * diff @ jnp.array(precision) @ diff

    return {"D": D, "mean": mean, "cov": cov, "log_prob": log_prob}


@pytest.fixture(scope="module")
def tengri_chain(gaussian_target):
    """Run tengri's ray tracer."""
    from tengri.inference.backends.mcmc.raytrace import sample_raytrace

    key = jax.random.PRNGKey(0)
    x0 = jnp.zeros(gaussian_target["D"])
    chain, _log_lik, accept_prob, _n_nonfinite = sample_raytrace(
        key,
        x0,
        gaussian_target["log_prob"],
        n_steps=3000,
        n_leapfrog_steps=20,
        step_size=0.1,
    )
    return {
        "chain": np.array(chain[500:]),  # discard burn-in
        "accept": float(accept_prob[500:].mean()),
    }


@pytest.fixture(scope="module")
def behroozi_chain(gaussian_target):
    """Run Behroozi's reference ray tracer."""
    ref = _load_behroozi_module()

    key = jax.random.PRNGKey(0)
    x0 = jnp.zeros(gaussian_target["D"])
    chain, _log_lik = ref.sample_raytrace(
        key,
        x0,
        gaussian_target["log_prob"],
        n_steps=3000,
        n_leapfrog_steps=20,
        step_size=0.1,
    )
    # Behroozi doesn't return accept_prob separately, compute from chain
    # Check for duplicate consecutive rows (rejections)
    diffs = np.diff(np.array(chain), axis=0)
    n_accept = np.sum(np.any(diffs != 0, axis=1))
    accept_rate = n_accept / (len(chain) - 1)
    return {
        "chain": np.array(chain[500:]),
        "accept": float(accept_rate),
    }


class TestRayTraceCrossValidation:
    """Verify tengri RT matches Behroozi's reference on a Gaussian target."""

    def test_chains_identical(self, tengri_chain, behroozi_chain):
        """With same PRNG key, chains should be numerically identical.

        Both implementations use the same JAX scan structure and PRNG
        splitting, so given the same key they should produce identical output.
        """
        # Allow tiny numerical differences from float64 arithmetic
        np.testing.assert_allclose(
            tengri_chain["chain"],
            behroozi_chain["chain"],
            atol=1e-10,
            rtol=1e-10,
            err_msg="Chains differ — implementations are not identical!",
        )

    def test_acceptance_rates_match(self, tengri_chain, behroozi_chain):
        """Acceptance rates should be nearly identical."""
        assert abs(tengri_chain["accept"] - behroozi_chain["accept"]) < 0.02

    def test_mean_matches_truth(self, tengri_chain, gaussian_target):
        """Chain mean should be near the true mean (zeros)."""
        chain_mean = tengri_chain["chain"].mean(axis=0)
        np.testing.assert_allclose(chain_mean, gaussian_target["mean"], atol=0.15)

    def test_covariance_matches_truth(self, tengri_chain, gaussian_target):
        """Chain covariance should approximate the target covariance."""
        chain_cov = np.cov(tengri_chain["chain"].T)
        np.testing.assert_allclose(
            chain_cov,
            gaussian_target["cov"],
            atol=0.15,
            rtol=0.3,
        )

    def test_autocorrelation_time_reasonable(self, tengri_chain):
        """ACT should be moderate for a well-tuned Gaussian target."""
        from tengri.analysis.diagnostics.autocorrelation import autocorrelation_time

        for d in range(tengri_chain["chain"].shape[1]):
            tau = autocorrelation_time(tengri_chain["chain"][:, d])
            # For a well-tuned sampler on a Gaussian, τ should be < 50
            assert tau < 100, f"dim {d}: τ={tau:.1f} is too high"


class TestHMCCrossValidation:
    """Verify tengri HMC matches Behroozi's reference."""

    def test_hmc_chains_identical(self, gaussian_target):
        """HMC mode should also produce identical chains."""
        from tengri.inference.backends.mcmc.raytrace import sample_hamiltonian

        ref = _load_behroozi_module()
        key = jax.random.PRNGKey(7)
        x0 = jnp.zeros(gaussian_target["D"])

        # Tengri
        chain_t, _, _ = sample_hamiltonian(
            key,
            x0,
            gaussian_target["log_prob"],
            n_steps=500,
            n_leapfrog_steps=20,
            step_size=0.05,
        )

        # Behroozi
        chain_b, _ = ref.sample_hamiltonian(
            key,
            x0,
            gaussian_target["log_prob"],
            n_steps=500,
            n_leapfrog_steps=20,
            step_size=0.05,
        )

        np.testing.assert_allclose(
            np.array(chain_t),
            np.array(chain_b),
            atol=1e-10,
            rtol=1e-10,
            err_msg="HMC chains differ!",
        )
