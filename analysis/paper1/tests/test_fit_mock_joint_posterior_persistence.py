# SPDX-License-Identifier: BSD-3-Clause
"""Test posterior persistence in fit_mock_joint.py.

Validates that sampler runs persist posterior draws, divergence diagnostics,
and per-parameter convergence metrics (rhat, ESS) to both NPZ and JSON files.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from analysis.paper1.fit_mock_joint import _save_sampler_results

pytestmark = pytest.mark.unit


def make_stub_posterior(n_params=5, n_draws=100, n_chains=2):
    """Create a minimal synthetic posterior for testing persistence.

    Args:
        n_params: Number of parameters.
        n_draws: Number of kept samples per chain (n_chains * n_draws total).
        n_chains: Number of MCMC chains.

    Returns:
        A stub Posterior-like object with required attributes.
    """

    class StubPosterior:
        def __init__(self):
            total_draws = n_chains * n_draws
            self.samples = {f"param_{i}": np.random.randn(total_draws) for i in range(n_params)}
            # Create a mask with exactly 2 True values
            divergent_mask = np.zeros(total_draws, dtype=bool)
            divergent_mask[5] = True
            divergent_mask[15] = True
            self.diagnostics = {
                "n_divergent": 2,
                "divergent_mask": divergent_mask,
                "energy": np.random.randn(total_draws),
                "ebfmi_per_chain": [0.8, 0.75],
                "ebfmi_min": 0.75,
            }

        def rhats(self):
            return {f"param_{i}": 1.01 + 0.001 * i for i in range(n_params)}

        def effective_sample_size(self):
            return {f"param_{i}": 50.0 + i for i in range(n_params)}

    return StubPosterior()


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_sampler_results_save_npz_has_draws(temp_output_dir):
    """Test that NPZ contains posterior draws with expected shape."""
    posterior = make_stub_posterior(n_params=5, n_draws=50, n_chains=2)
    out_npz = temp_output_dir / "test_output.npz"
    out_json = temp_output_dir / "test_output.json"

    free = [f"param_{i}" for i in range(5)]
    truth = {k: float(i) for i, k in enumerate(free)}
    fitted = {k: float(i) + 0.1 for i, k in enumerate(free)}
    delta = {k: fitted[k] - truth[k] for k in free}
    method = "mcmc_nuts"
    sampler_kwargs = {
        "n_warmup": 150,
        "n_samples": 300,
        "n_chains": 2,
        "dense_mass_matrix": False,
    }

    _save_sampler_results(
        posterior,
        out_npz,
        out_json,
        free,
        truth,
        fitted,
        delta,
        wall=10.5,
        method=method,
        sampler_kwargs=sampler_kwargs,
    )

    assert out_npz.exists(), "NPZ file should be created"
    data = np.load(out_npz, allow_pickle=True)

    # Check that draws are present for each parameter
    for i in range(5):
        key = f"param_{i}"
        assert key in data, f"Posterior draws for {key} should be in NPZ"
        draws = data[key]
        # Draws should be thinned but present
        assert draws.shape[0] > 0, f"Parameter {key} should have draws"
        assert draws.ndim == 1, f"Draws for {key} should be 1-D array"


def test_sampler_results_save_npz_has_diagnostics(temp_output_dir):
    """Test that NPZ contains diagnostic keys (divergences, rhat, ESS)."""
    posterior = make_stub_posterior(n_params=5, n_draws=50, n_chains=2)
    out_npz = temp_output_dir / "test_output.npz"
    out_json = temp_output_dir / "test_output.json"

    free = [f"param_{i}" for i in range(5)]
    truth = {k: float(i) for i, k in enumerate(free)}
    fitted = {k: float(i) + 0.1 for i, k in enumerate(free)}
    delta = {k: fitted[k] - truth[k] for k in free}

    sampler_kwargs = {
        "n_warmup": 150,
        "n_samples": 300,
        "n_chains": 2,
    }

    _save_sampler_results(
        posterior,
        out_npz,
        out_json,
        free,
        truth,
        fitted,
        delta,
        wall=10.5,
        method="mcmc_nuts",
        sampler_kwargs=sampler_kwargs,
    )

    data = np.load(out_npz, allow_pickle=True)

    # Check divergence metadata
    assert "divergences_count" in data, "Divergence count should be in NPZ"
    assert int(data["divergences_count"]) == 2, "Should have 2 divergences"

    # Check rhat keys
    assert "rhat_max" in data, "rhat_max should be in NPZ"
    assert float(data["rhat_max"]) > 1.0, "rhat_max should be present"

    for i in range(5):
        assert f"rhat_param_{i}" in data, f"rhat for param_{i} should be in NPZ"

    # Check ESS keys
    assert "ess_min" in data, "ess_min should be in NPZ"
    assert float(data["ess_min"]) > 0, "ess_min should be present"

    for i in range(5):
        assert f"ess_param_{i}" in data, f"ESS for param_{i} should be in NPZ"

    # Check energy
    assert "energy" in data, "Energy array should be in NPZ"
    energy = data["energy"]
    assert energy.shape[0] > 0, "Energy array should have samples"


def test_sampler_results_save_json_has_diagnostics(temp_output_dir):
    """Test that JSON sidecar contains diagnostic summary."""
    posterior = make_stub_posterior(n_params=5, n_draws=50, n_chains=2)
    out_npz = temp_output_dir / "test_output.npz"
    out_json = temp_output_dir / "test_output.json"

    free = [f"param_{i}" for i in range(5)]
    truth = {k: float(i) for i, k in enumerate(free)}
    fitted = {k: float(i) + 0.1 for i, k in enumerate(free)}
    delta = {k: fitted[k] - truth[k] for k in free}

    sampler_kwargs = {
        "n_warmup": 150,
        "n_samples": 300,
        "n_chains": 2,
    }

    _save_sampler_results(
        posterior,
        out_npz,
        out_json,
        free,
        truth,
        fitted,
        delta,
        wall=10.5,
        method="mcmc_nuts",
        sampler_kwargs=sampler_kwargs,
    )

    assert out_json.exists(), "JSON sidecar should be created"

    with open(out_json) as f:
        json_data = json.load(f)

    # Check essential keys
    assert "divergences" in json_data, "divergences count should be in JSON"
    assert json_data["divergences"] == 2, "Should record 2 divergences"

    assert "rhat_max" in json_data, "rhat_max should be in JSON"
    assert "ess_min" in json_data, "ess_min should be in JSON"

    assert "ebfmi_per_chain" in json_data, "ebfmi_per_chain should be in JSON"
    assert json_data["ebfmi_per_chain"] == [0.8, 0.75], "ebfmi_per_chain should match"

    assert "ebfmi_min" in json_data, "ebfmi_min should be in JSON"
    assert json_data["ebfmi_min"] == 0.75, "ebfmi_min should match"

    assert "wall_seconds" in json_data, "wall_seconds should be in JSON"
    assert "method" in json_data, "method should be in JSON"
