# SPDX-License-Identifier: BSD-3-Clause
"""Pool chains: synthetic chains, validation, and gate enforcement."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from paper1 import fit_mock_joint, pool_mock_chains
from paper1._posterior_utils import posterior_output_paths

pytestmark = pytest.mark.contract


@pytest.fixture
def synthetic_chains(tmp_path):
    """Create synthetic well-mixed chains for pooling tests.

    Returns a dict with helper functions to add chains to tmp_path.
    """

    def add_chain(
        tag: str,
        seed: int,
        n_draws: int = 500,
        shift: float = 0.0,
        divergences: int = 0,
        seed_repeat: bool = False,
    ):
        """Add a synthetic chain to tmp_path/mock_joint_chains/.

        Args:
            tag: The chain tag (e.g., "0", "1").
            seed: The random seed for this chain.
            n_draws: Number of posterior draws per chain.
            shift: Amount to shift one parameter by (for rhat test).
            divergences: Number of divergences to report.
            seed_repeat: If True, use the same seed as another chain (for validation test).
        """
        np.random.seed(seed if not seed_repeat else 0)

        chains_dir = tmp_path / "mock_joint_chains"
        chains_dir.mkdir(parents=True, exist_ok=True)

        # Synthetic draws for 4 parameters: iid normal (0, 1)
        free_params = ["param_a", "param_b", "param_c", "param_d"]
        n_params = len(free_params)

        npz_data = {}
        samples = {}
        for i, p in enumerate(free_params):
            s = np.random.normal(0, 1, n_draws)
            if i == 0 and shift != 0:
                s = s + shift
            samples[p] = s
            npz_data[p] = s

        # Metadata
        npz_data["free_params"] = np.array(free_params, dtype=object)
        npz_data["truth_values"] = np.zeros(n_params)
        npz_data["fitted_values"] = np.mean([samples[p] for p in free_params], axis=1)
        npz_data["delta"] = npz_data["fitted_values"] - npz_data["truth_values"]
        npz_data["method"] = "mcmc_nuts"
        npz_data["wall_seconds"] = 60.0
        npz_data["n_chains"] = 1
        npz_data["n_warmup"] = 300
        npz_data["n_samples"] = n_draws
        npz_data["divergences_count"] = divergences
        npz_data["seed"] = seed

        # Save NPZ
        npz_path = chains_dir / f"mock_joint_mcmc_nuts_{tag}.npz"
        np.savez(npz_path, **npz_data)

        # Create JSON sidecar
        json_data = {
            "divergences": divergences,
            "rhat": {p: 1.0 for p in free_params},  # Pretend all converged
            "rhat_max": 1.0,
            "ess": {p: 250.0 for p in free_params},  # ESS per chain
            "ess_min": 250.0,
            "wall_seconds": 60.0,
            "n_chains": 1,
            "n_warmup": 300,
            "n_samples": n_draws,
            "target_accept_rate": 0.85,
            "dense_mass_matrix": False,
            "method": "mcmc_nuts",
            "provenance": "abc123def456",
            "seed": seed,
            "chain_tag": tag,
            "dense_cap_override": None,
        }
        json_path = npz_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(json_data, f)

        return npz_path, json_path

    return add_chain


def test_pool_four_iid_chains_pass_gate(synthetic_chains, tmp_path):
    """Four well-mixed iid-normal chains with different seeds pool and pass gate."""
    # Create four chains, no divergences, high ESS
    for i in range(4):
        synthetic_chains(tag=str(i), seed=100 + i, n_draws=1000)

    # Run pool_mock_chains
    ret = pool_mock_chains.main(
        [
            "--chains-dir",
            str(tmp_path / "mock_joint_chains"),
            "--results-dir",
            str(tmp_path),
            "--method",
            "mcmc_nuts",
        ]
    )

    assert ret == 0, "pooling should succeed"

    # Verify canonical files exist
    canonical_npz, _canonical_json = posterior_output_paths(tmp_path, "mcmc_nuts")
    assert canonical_npz.exists()
    assert _canonical_json.exists()

    # Verify pooled n_chains == 4
    pooled_npz = np.load(canonical_npz, allow_pickle=True)
    assert int(pooled_npz["n_chains"]) == 4

    # Independent chains add their effective samples: the pooled ESS is the SUM
    # of each chain's own, recomputed here from the chain files rather than read
    # back from the pooler. A minimum or a mean would understate it fourfold.
    from tengri.analysis.diagnostics.autocorrelation import effective_sample_size

    chain_files = sorted((tmp_path / "mock_joint_chains").glob("mock_joint_mcmc_nuts_*.npz"))
    assert len(chain_files) == 4
    for param in [str(p) for p in pooled_npz["free_params"]]:
        per_chain = [
            effective_sample_size({param: np.load(f)[param]})[param]["ess"] for f in chain_files
        ]
        assert float(pooled_npz[f"ess_{param}"]) == pytest.approx(sum(per_chain), rel=1e-9)


def test_pool_with_high_rhat_fails_gate(synthetic_chains, tmp_path):
    """One chain shifted by 5 sd gives rhat_max > 1.01, gate fails, canonical not written."""
    synthetic_chains(tag="0", seed=100, n_draws=1000)
    synthetic_chains(tag="1", seed=101, n_draws=1000)
    synthetic_chains(tag="2", seed=102, n_draws=1000, shift=5.0)  # Shifted!
    synthetic_chains(tag="3", seed=103, n_draws=1000)

    ret = pool_mock_chains.main(
        [
            "--chains-dir",
            str(tmp_path / "mock_joint_chains"),
            "--results-dir",
            str(tmp_path),
            "--method",
            "mcmc_nuts",
        ]
    )

    assert ret == 1, "pooling should fail due to rhat"

    # Canonical should NOT exist
    canonical_npz, _canonical_json = posterior_output_paths(tmp_path, "mcmc_nuts")
    assert not canonical_npz.exists()
    assert not _canonical_json.exists()

    # Candidate should exist
    candidate_npz = tmp_path / "mock_joint_chains" / "pooled_candidate.npz"
    assert candidate_npz.exists()


def test_pool_with_divergences_fails_gate(synthetic_chains, tmp_path):
    """One chain reports 1 divergence, gate fails because max divergences is 0."""
    synthetic_chains(tag="0", seed=100, n_draws=1000)
    synthetic_chains(tag="1", seed=101, n_draws=1000)
    synthetic_chains(tag="2", seed=102, n_draws=1000, divergences=1)  # Divergent!
    synthetic_chains(tag="3", seed=103, n_draws=1000)

    ret = pool_mock_chains.main(
        [
            "--chains-dir",
            str(tmp_path / "mock_joint_chains"),
            "--results-dir",
            str(tmp_path),
            "--method",
            "mcmc_nuts",
        ]
    )

    assert ret == 1, "pooling should fail due to divergences"

    # Canonical should NOT exist
    canonical_npz, __canonical_json = posterior_output_paths(tmp_path, "mcmc_nuts")
    assert not canonical_npz.exists()


def test_pool_mismatched_free_params_rejected(synthetic_chains, tmp_path):
    """Mismatched free_params between chains causes nonzero exit."""
    chains_dir = tmp_path / "mock_joint_chains"
    chains_dir.mkdir(parents=True, exist_ok=True)

    # Chain 0: params a, b, c, d
    synthetic_chains(tag="0", seed=100)
    synthetic_chains(tag="1", seed=101)

    # Chain 2 with different params
    npz_path = chains_dir / "mock_joint_mcmc_nuts_2.npz"
    different_params = ["param_x", "param_y"]
    npz_data = {p: np.random.normal(0, 1, 500) for p in different_params}
    npz_data["free_params"] = np.array(different_params, dtype=object)
    npz_data["truth_values"] = np.zeros(2)
    npz_data["fitted_values"] = np.zeros(2)
    npz_data["delta"] = np.zeros(2)
    npz_data["method"] = "mcmc_nuts"
    npz_data["wall_seconds"] = 60.0
    npz_data["n_chains"] = 1
    npz_data["n_warmup"] = 300
    npz_data["n_samples"] = 500
    npz_data["divergences_count"] = 0
    np.savez(npz_path, **npz_data)

    json_path = npz_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({"free_params": different_params, "provenance": "abc123"}, f)

    # Run pool: should fail
    ret = pool_mock_chains.main(
        [
            "--chains-dir",
            str(chains_dir),
            "--results-dir",
            str(tmp_path),
            "--method",
            "mcmc_nuts",
        ]
    )

    assert ret == 1


def test_pool_duplicate_seed_rejected(synthetic_chains, tmp_path):
    """Duplicate seed between chains causes nonzero exit."""
    synthetic_chains(tag="0", seed=100)
    synthetic_chains(tag="1", seed=100, seed_repeat=True)  # Same seed!

    ret = pool_mock_chains.main(
        [
            "--chains-dir",
            str(tmp_path / "mock_joint_chains"),
            "--results-dir",
            str(tmp_path),
            "--method",
            "mcmc_nuts",
        ]
    )

    assert ret == 1


def test_chain_output_paths_rejects_invalid_tag(tmp_path):
    """chain_output_paths validates tag format."""
    from _posterior_utils import chain_output_paths

    # Valid tags
    chain_output_paths(tmp_path, "mcmc_nuts", "tag_1")
    chain_output_paths(tmp_path, "mcmc_nuts", "tag-2")
    chain_output_paths(tmp_path, "mcmc_nuts", "TAG3")

    # Invalid tags
    with pytest.raises(ValueError):
        chain_output_paths(tmp_path, "mcmc_nuts", "tag/invalid")

    # Verify paths never equal canonical
    chain_npz, chain_json = chain_output_paths(tmp_path, "mcmc_nuts", "mytag")
    canonical_npz, _canonical_json = posterior_output_paths(tmp_path, "mcmc_nuts")
    assert chain_npz != canonical_npz
    assert chain_json != _canonical_json


def test_fit_mock_joint_parser_chain_tag_validation():
    """--chain-tag with --n-chains 4 causes SystemExit."""
    with pytest.raises(SystemExit):
        fit_mock_joint.main(
            [
                "--chain-tag",
                "test",
                "--n-chains",
                "4",
                "--method",
                "mcmc_nuts",
            ]
        )

    # --chain-tag without sampler method causes SystemExit
    with pytest.raises(SystemExit):
        fit_mock_joint.main(
            [
                "--chain-tag",
                "test",
                "--n-chains",
                "1",
                "--method",
                "map",
            ]
        )


def test_fit_mock_joint_parser_dense_above_cap_validation():
    """--dense-above-cap without --dense-mass or without --n-chains 1 causes SystemExit."""
    # Without --dense-mass
    with pytest.raises(SystemExit):
        fit_mock_joint.main(
            [
                "--dense-above-cap",
                "--n-chains",
                "1",
                "--method",
                "mcmc_nuts",
            ]
        )

    # Without --n-chains 1
    with pytest.raises(SystemExit):
        fit_mock_joint.main(
            [
                "--dense-above-cap",
                "--dense-mass",
                "--n-chains",
                "4",
                "--method",
                "mcmc_nuts",
            ]
        )


def test_apply_dense_mass_override_raises_constant():
    """apply_dense_mass_override raises DENSE_MASS_MAX_DIM and verifies gate passes."""
    import tengri.inference.backends.mcmc.nuts as nuts_mod

    # Store original value
    original_cap = nuts_mod.DENSE_MASS_MAX_DIM

    try:
        # Verify the library cap is 30
        assert nuts_mod.DENSE_MASS_MAX_DIM == 30

        # Call the override helper with D=36
        result = fit_mock_joint.apply_dense_mass_override(36, "mcmc_nuts")

        # Should return override info
        assert result is not None
        assert result["library_cap"] == 30
        assert result["raised_to"] == 36

        # Verify the gate now passes at D=36
        gate_passes = nuts_mod.resolve_dense_mass_gate(True, 36, method="mcmc_nuts", verbose=False)
        assert gate_passes

    finally:
        # Restore the original constant
        nuts_mod.DENSE_MASS_MAX_DIM = original_cap


def test_dense_above_cap_refuses_to_run_without_a_chain_tag(capsys):
    """A lone dense chain must not be able to write the canonical posterior."""
    with pytest.raises(SystemExit) as exc:
        fit_mock_joint.main(
            ["--dense-above-cap", "--dense-mass", "--n-chains", "1", "--method", "mcmc_nuts"]
        )
    assert exc.value.code == 2
    assert "--dense-above-cap requires --chain-tag" in capsys.readouterr().err


def test_the_cap_refusal_filter_matches_the_samplers_real_warning():
    """The filter that makes a refusal fatal must match the warning actually emitted.

    Emit the real warning from the unpatched gate and check the driver's pattern
    catches it; a pattern that drifted from the message would let a dense run
    fall back to diagonal silently again.
    """
    import warnings

    import tengri.inference.backends.mcmc.nuts as nuts_mod

    assert nuts_mod.DENSE_MASS_MAX_DIM == 30
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.filterwarnings("error", message=fit_mock_joint.CAP_REFUSAL_WARNING)
        with pytest.raises(UserWarning):
            nuts_mod.resolve_dense_mass_gate(True, 36, method="mcmc_nuts", verbose=False)
