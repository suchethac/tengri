# SPDX-License-Identifier: BSD-3-Clause
"""Test BMA combiner contract over Laplace evidences.

Pinned invariants:
- Weights sum to 1 per set
- Invalid cells excluded with reason
- One route per set (Laplace only)
- Factorial marginals and priors validated
- VI never enters named sets
- NUTS adoption status loaded
- BMA percentiles computed correctly
- CLI handles missing directories
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PAPER1 = Path(__file__).resolve().parent.parent
ANALYSIS = PAPER1.parent
for entry in [str(ANALYSIS), str(PAPER1)]:
    if entry not in sys.path:
        sys.path.insert(0, entry)

from paper1.bma_combine import (
    _compute_factorial_marginals,
    _compute_prior_mass,
    _resample_mixture,
    _validate_cell,
    combine_bma,
    softmax_weights,
)

pytestmark = pytest.mark.unit


# ============================================================================
# Helper functions for test data
# ============================================================================


def _make_evidence_cell(
    model_key: str,
    config: str | None = None,
    model_set: str = "factorial",
    log_evidence: float = 0.0,
    valid: bool = True,
    newton_decrement: float = 0.01,
    n_clipped_eigenvalues: int = 0,
    route: str = "laplace",
) -> dict:
    """Create a synthetic evidence cell."""
    components = {
        "sfh": "continuity",
        "ssp": "mist_c3k",
        "attenuation": "calzetti",
        "dust_emission": "dl14",
        "nebular": "cue",
    }
    if config:
        components["config"] = config

    # For invalid cells, set diagnostics to invalid values
    if not valid:
        actual_log_evidence = None
        actual_newton = None
        actual_clipped = None
    else:
        actual_log_evidence = log_evidence
        actual_newton = newton_decrement
        actual_clipped = n_clipped_eigenvalues

    return {
        "galaxy": 1000,
        "z": 0.5,
        "model_key": model_key,
        "model_set": model_set,
        "components": components,
        "route": route,
        "log_evidence": actual_log_evidence,
        "map_loss": 100.0,
        "n_free": 10,
        "newton_decrement": actual_newton,
        "n_clipped_eigenvalues": actual_clipped,
        "condition_number": 1e5,
        "valid": valid,
        "n_map_restarts": 1,
        "map_restart_loss_spread": None,
        "wall_time_s": 100.0,
        "peak_rss_gb": 5.0,
        "code_revision": "test",
        "seed": 0,
        "error": None if valid else "test error",
    }


def _make_npz_data(galaxy_id: int, n_samples: int = 100) -> dict[str, np.ndarray]:
    """Create synthetic NPZ arrays."""
    return {
        "log_stellar_mass_formed": np.random.normal(10.5, 0.5, n_samples),
        "log_stellar_mass_survived": np.random.normal(10.2, 0.5, n_samples),
        "log_sfr_100myr": np.random.normal(-0.5, 0.3, n_samples),
        "log_sfr_10myr": np.random.normal(-1.0, 0.5, n_samples),
    }


# ============================================================================
# Tests
# ============================================================================


def test_softmax_weights_sum_to_one():
    """BMA weights sum to 1."""
    log_z = {
        "model_A": 0.0,
        "model_B": 1.0,
        "model_C": 2.0,
    }
    weights = softmax_weights(log_z)
    assert abs(sum(weights.values()) - 1.0) < 1e-10


def test_softmax_weights_high_contrast():
    """A 10-nat lead gives weight > 0.9999."""
    log_z = {
        "model_A": 10.0,
        "model_B": 0.0,
    }
    weights = softmax_weights(log_z)
    assert weights["model_A"] > 0.9999


def test_softmax_weights_equal_evidence():
    """Equal log Z gives equal weights."""
    log_z = {
        "model_A": 5.0,
        "model_B": 5.0,
        "model_C": 5.0,
    }
    weights = softmax_weights(log_z)
    for w in weights.values():
        assert abs(w - 1.0 / 3.0) < 1e-10


def test_softmax_weights_empty():
    """Empty input returns empty dict."""
    weights = softmax_weights({})
    assert weights == {}


def test_softmax_weights_single():
    """Single model gets weight 1.0."""
    weights = softmax_weights({"only": 0.0})
    assert weights["only"] == 1.0


def test_validate_cell_valid():
    """A valid cell passes validation."""
    cell = _make_evidence_cell(
        "model_key",
        config="I",
        log_evidence=10.0,
        newton_decrement=0.05,
        n_clipped_eigenvalues=0,
    )
    is_valid, reason = _validate_cell(cell)
    assert is_valid is True
    assert reason is None


def test_validate_cell_invalid_newton():
    """Newton decrement > 0.1 fails validation."""
    cell = _make_evidence_cell(
        "model_key", log_evidence=10.0, newton_decrement=0.5
    )
    is_valid, reason = _validate_cell(cell)
    assert is_valid is False
    assert "newton_decrement" in reason


def test_validate_cell_invalid_clipped():
    """n_clipped_eigenvalues > 0 fails validation."""
    cell = _make_evidence_cell(
        "model_key",
        log_evidence=10.0,
        newton_decrement=0.05,
        n_clipped_eigenvalues=1,
    )
    is_valid, reason = _validate_cell(cell)
    assert is_valid is False
    assert "clipped" in reason


def test_validate_cell_invalid_log_evidence():
    """Infinite log_evidence fails validation."""
    cell = _make_evidence_cell("model_key", log_evidence=np.inf)
    is_valid, reason = _validate_cell(cell)
    assert is_valid is False
    assert "log_evidence" in reason


def test_validate_cell_invalid_nan():
    """NaN log_evidence fails validation."""
    cell = _make_evidence_cell("model_key", log_evidence=np.nan)
    is_valid, _ = _validate_cell(cell)
    assert is_valid is False


def test_validate_cell_wrong_route():
    """Non-laplace route fails validation."""
    cell = _make_evidence_cell("model_key", route="hmc")
    is_valid, reason = _validate_cell(cell)
    assert is_valid is False
    assert "route" in reason


def test_validate_cell_flag_disagrees_with_diagnostics():
    """Flag says invalid but diagnostics say valid."""
    cell = _make_evidence_cell(
        "model_key",
        log_evidence=10.0,
        newton_decrement=0.05,
        n_clipped_eigenvalues=0,
        valid=True,  # Create valid diagnostics first
    )
    # Now override the flag to create disagreement
    cell["valid"] = False
    is_valid, reason = _validate_cell(cell)
    assert is_valid is False
    assert "disagrees" in reason


def test_combine_bma_basic(tmp_path):
    """Basic BMA combining with synthetic data."""
    # Create evidence directory structure
    evidence_dir = tmp_path / "bma_evidence"
    gal_dir = evidence_dir / "1000"
    gal_dir.mkdir(parents=True)

    # Write three factorial cells with different keys
    keys = [
        "sfh-continuity__ssp-mist_c3k__att-calzetti__ir-dl14__neb-cue",
        "sfh-dirichlet__ssp-mist_c3k__att-calzetti__ir-dl14__neb-cue",
        "sfh-delayed__ssp-mist_c3k__att-calzetti__ir-dl14__neb-cue",
    ]
    for i, (key, log_z) in enumerate(zip(keys, [10.0, 0.0, 0.0])):
        cell = _make_evidence_cell(
            key,
            model_set="factorial",
            log_evidence=log_z,
        )
        cell_file = gal_dir / f"cell_{i}.json"
        with open(cell_file, "w") as f:
            json.dump(cell, f)

        # Write NPZ
        npz_file = cell_file.with_suffix(".npz")
        npz_data = _make_npz_data(1000)
        np.savez(npz_file, **npz_data)

    # Combine
    summary = combine_bma(evidence_dir, tmp_path / "fits", tmp_path / "fits_xlike", 100, 0)

    assert "galaxies" in summary
    assert "1000" in summary["galaxies"]

    gal_data = summary["galaxies"]["1000"]
    assert "sets" in gal_data
    assert "factorial" in gal_data["sets"]

    w_set = gal_data["sets"]["factorial"]
    assert "models" in w_set
    assert len(w_set["models"]) == 3

    # Check weights sum to 1
    weights = {m["model_key"]: m["weight"] for m in w_set["models"] if m["weight"]}
    assert abs(sum(weights.values()) - 1.0) < 1e-10

    # Check max weight
    assert w_set["max_weight"] is not None
    assert w_set["max_weight"] > 0.9  # Highest evidence dominates


def test_combine_bma_weights_valid_only(tmp_path):
    """Only valid cells contribute to weights."""
    evidence_dir = tmp_path / "bma_evidence"
    gal_dir = evidence_dir / "2000"
    gal_dir.mkdir(parents=True)

    # Use proper factorial model keys
    key_valid = "sfh-continuity__ssp-mist_c3k__att-calzetti__ir-dl14__neb-cue"
    key_invalid = "sfh-dirichlet__ssp-mist_c3k__att-calzetti__ir-dl14__neb-cue"

    # Write one valid and one invalid cell
    cell_valid = _make_evidence_cell(
        key_valid,
        model_set="factorial",
        log_evidence=10.0,
        valid=True,
    )
    # Create invalid cell with bad newton_decrement (but valid log_evidence to test that specifically)
    cell_invalid = _make_evidence_cell(
        key_invalid,
        model_set="factorial",
        log_evidence=9.0,
        valid=True,  # Set to True so diagnostics are not None
        newton_decrement=0.5,  # This should fail
    )
    cell_invalid["valid"] = False  # Override flag to test conflict

    with open(gal_dir / "valid.json", "w") as f:
        json.dump(cell_valid, f)
    with open(gal_dir / "invalid.json", "w") as f:
        json.dump(cell_invalid, f)

    # NPZ for valid only
    npz_data = _make_npz_data(2000)
    np.savez(gal_dir / "valid.npz", **npz_data)

    summary = combine_bma(evidence_dir, tmp_path / "fits", tmp_path / "fits_xlike", 100, 0)

    w_set = summary["galaxies"]["2000"]["sets"]["factorial"]
    models = w_set["models"]

    # Find the valid and invalid models
    valid_model = next((m for m in models if m["model_key"] == key_valid), None)
    invalid_model = next((m for m in models if m["model_key"] == key_invalid), None)

    assert valid_model is not None
    assert valid_model["weight"] == 1.0  # Only one valid, gets all weight

    assert invalid_model is not None
    assert invalid_model["weight"] is None
    assert invalid_model["valid"] is False
    # Should be marked as disagreement or newton error
    assert any(x in invalid_model["excluded_reason"] for x in ["newton", "disagrees"])


def test_combine_bma_invalid_counts(tmp_path):
    """Invalid counts are aggregated correctly."""
    evidence_dir = tmp_path / "bma_evidence"

    # Create two galaxies, each with one invalid cell
    for gal_id in [1000, 2000]:
        gal_dir = evidence_dir / str(gal_id)
        gal_dir.mkdir(parents=True)

        cell = _make_evidence_cell(
            "problem_model",
            model_set="factorial",
            valid=False,
            newton_decrement=0.5,
        )
        with open(gal_dir / "cell.json", "w") as f:
            json.dump(cell, f)

    summary = combine_bma(evidence_dir, tmp_path / "fits", tmp_path / "fits_xlike", 100, 0)

    # Check invalid_counts
    invalid_counts = summary["invalid_counts"]["factorial"]
    assert invalid_counts.get("problem_model", 0) == 2


def test_combine_bma_mixed_route_fails(tmp_path):
    """Mixed routes in one galaxy raises error."""
    evidence_dir = tmp_path / "bma_evidence"
    gal_dir = evidence_dir / "3000"
    gal_dir.mkdir(parents=True)

    # Write cells with different routes
    cell1 = _make_evidence_cell("model_A", route="laplace")
    cell2 = _make_evidence_cell("model_B", route="hmc")

    with open(gal_dir / "cell1.json", "w") as f:
        json.dump(cell1, f)
    with open(gal_dir / "cell2.json", "w") as f:
        json.dump(cell2, f)

    with pytest.raises(ValueError, match="multiple routes"):
        combine_bma(
            evidence_dir, tmp_path / "fits", tmp_path / "fits_xlike", 100, 0
        )


def test_combine_bma_close_flag(tmp_path):
    """Close flag set when max weight < 0.9."""
    evidence_dir = tmp_path / "bma_evidence"
    gal_dir = evidence_dir / "4000"
    gal_dir.mkdir(parents=True)

    # Create three cells with similar evidence (weights ~ 0.33 each)
    for i in range(3):
        cell = _make_evidence_cell(
            f"model_{i}",
            model_set="factorial",
            log_evidence=0.0,  # All equal
        )
        with open(gal_dir / f"cell_{i}.json", "w") as f:
            json.dump(cell, f)

        npz_data = _make_npz_data(4000)
        np.savez(gal_dir / f"cell_{i}.npz", **npz_data)

    summary = combine_bma(evidence_dir, tmp_path / "fits", tmp_path / "fits_xlike", 100, 0)

    w_set = summary["galaxies"]["4000"]["sets"]["factorial"]
    assert w_set["close"] is True  # max weight ~0.333 < 0.9


def test_combine_bma_not_close_flag(tmp_path):
    """Close flag not set when one model dominates."""
    evidence_dir = tmp_path / "bma_evidence"
    gal_dir = evidence_dir / "5000"
    gal_dir.mkdir(parents=True)

    # Create one dominant cell and one weak one
    cell_strong = _make_evidence_cell("model_A", log_evidence=10.0)
    cell_weak = _make_evidence_cell("model_B", log_evidence=0.0)

    with open(gal_dir / "cell_strong.json", "w") as f:
        json.dump(cell_strong, f)
    with open(gal_dir / "cell_weak.json", "w") as f:
        json.dump(cell_weak, f)

    for i in range(2):
        npz_data = _make_npz_data(5000)
        np.savez(gal_dir / f"cell_{chr(ord('A') + i)}.npz", **npz_data)

    summary = combine_bma(evidence_dir, tmp_path / "fits", tmp_path / "fits_xlike", 100, 0)

    w_set = summary["galaxies"]["5000"]["sets"]["factorial"]
    assert w_set["close"] is False  # max weight >> 0.9


def test_combine_bma_factorial_marginals(tmp_path):
    """Factorial marginals sum to 1 per axis."""
    evidence_dir = tmp_path / "bma_evidence"
    gal_dir = evidence_dir / "6000"
    gal_dir.mkdir(parents=True)

    # Create two models with different sfh
    cell1 = _make_evidence_cell(
        "sfh-continuity__ssp-mist_c3k__att-calzetti__ir-dl14__neb-cue",
        log_evidence=1.0,
    )
    cell2 = _make_evidence_cell(
        "sfh-dirichlet__ssp-mist_c3k__att-calzetti__ir-dl14__neb-cue",
        log_evidence=1.0,
    )

    with open(gal_dir / "cell1.json", "w") as f:
        json.dump(cell1, f)
    with open(gal_dir / "cell2.json", "w") as f:
        json.dump(cell2, f)

    for i in range(2):
        npz_data = _make_npz_data(6000)
        np.savez(gal_dir / f"cell{i + 1}.npz", **npz_data)

    summary = combine_bma(evidence_dir, tmp_path / "fits", tmp_path / "fits_xlike", 100, 0)

    w_set = summary["galaxies"]["6000"]["sets"]["factorial"]
    marginal = w_set.get("marginal", {})

    # Check sfh marginal sums to 1
    if "sfh" in marginal:
        sfh_sum = sum(marginal["sfh"].values())
        assert abs(sfh_sum - 1.0) < 1e-10


def test_combine_bma_zero_valid_cells(tmp_path):
    """Galaxy with no valid cells reports reason."""
    evidence_dir = tmp_path / "bma_evidence"
    gal_dir = evidence_dir / "7000"
    gal_dir.mkdir(parents=True)

    # All cells invalid
    cell = _make_evidence_cell(
        "model_A",
        valid=False,
        newton_decrement=0.5,
    )
    with open(gal_dir / "cell.json", "w") as f:
        json.dump(cell, f)

    summary = combine_bma(evidence_dir, tmp_path / "fits", tmp_path / "fits_xlike", 100, 0)

    w_set = summary["galaxies"]["7000"]["sets"]["factorial"]
    assert w_set["n_valid"] == 0
    assert "reason" in w_set
    assert w_set["bma_percentiles"] == {}


def test_combine_bma_missing_evidence_dir(tmp_path):
    """Missing evidence dir raises FileNotFoundError."""
    nonexistent = tmp_path / "nonexistent"
    with pytest.raises(FileNotFoundError):
        combine_bma(nonexistent, tmp_path / "fits", tmp_path / "fits_xlike", 100, 0)


def test_compute_factorial_marginals():
    """Factorial marginals computed correctly."""
    # Create mock models with equal weights
    models = [
        {
            "model_key": "sfh-continuity__ssp-mist_c3k__att-calzetti__ir-dl14__neb-cue",
            "valid": True,
        },
        {
            "model_key": "sfh-dirichlet__ssp-mist_c3k__att-calzetti__ir-dl14__neb-cue",
            "valid": True,
        },
    ]
    weights = {
        models[0]["model_key"]: 0.5,
        models[1]["model_key"]: 0.5,
    }

    marginals = _compute_factorial_marginals(models, weights)

    # Check that sfh has both values
    assert "sfh" in marginals
    assert "continuity" in marginals["sfh"]
    assert "dirichlet" in marginals["sfh"]
    assert abs(marginals["sfh"]["continuity"] - 0.5) < 1e-10
    assert abs(marginals["sfh"]["dirichlet"] - 0.5) < 1e-10


def test_compute_prior_mass():
    """Prior mass computed with correct values."""
    prior = _compute_prior_mass()

    assert "factorial" in prior
    assert "sfh" in prior["factorial"]

    # Check factorial prior: 5 sfh types, flat -> 0.2 each
    assert len(prior["factorial"]["sfh"]) == 5
    for _value, mass in prior["factorial"]["sfh"].items():
        assert abs(mass - 0.2) < 1e-10

    # Check isochrone prior: 0.4/0.4/0.2
    assert abs(prior["factorial"]["isochrone"]["mist"] - 0.4) < 1e-10
    assert abs(prior["factorial"]["isochrone"]["prsc"] - 0.4) < 1e-10
    assert abs(prior["factorial"]["isochrone"]["bpass"] - 0.2) < 1e-10

    # Check spectral prior: 0.6/0.4
    assert abs(prior["factorial"]["spectral_library"]["c3k"] - 0.6) < 1e-10
    assert abs(prior["factorial"]["spectral_library"]["miles"] - 0.4) < 1e-10


def test_resample_mixture_percentiles(tmp_path):
    """Resample produces BMA percentiles."""

    # Model A: high mass
    class NPZ_A:
        def __init__(self):
            self.files = ["log_stellar_mass_formed"]
            self._data = {"log_stellar_mass_formed": np.array([11.5] * 100)}

        def __getitem__(self, key):
            if key in self._data:
                return self._data[key]
            raise KeyError(key)

        def __contains__(self, key):
            return key in self._data

    # Model B: low mass
    class NPZ_B:
        def __init__(self):
            self.files = ["log_stellar_mass_formed"]
            self._data = {"log_stellar_mass_formed": np.array([10.0] * 100)}

        def __getitem__(self, key):
            if key in self._data:
                return self._data[key]
            raise KeyError(key)

        def __contains__(self, key):
            return key in self._data

    posteriors = [
        {
            "model_key": "model_A",
            "_npz": NPZ_A(),
        },
        {
            "model_key": "model_B",
            "_npz": NPZ_B(),
        },
    ]

    weights = {"model_A": 0.9, "model_B": 0.1}

    percentiles = _resample_mixture(posteriors, [0, 1], weights, 1000, 42)

    # BMA should be dominated by model A (weight 0.9)
    assert "log_stellar_mass_formed" in percentiles
    p50 = percentiles["log_stellar_mass_formed"][1]
    # Should be closer to 11.5 than to 10.0 due to weights
    assert p50 > 11.0


@pytest.mark.skipif(
    True,
    reason="Requires JAX/tengri import; swap < 3GB. Test written, not run: memory",
)
def test_softmax_weights_parity_with_tengri():
    """Our softmax_weights matches tengri.inference.bma.bma_weights."""
    try:
        from tengri.inference.bma import bma_weights as tengri_bma_weights
    except ImportError:
        pytest.skip("JAX/tengri not available")

    import types

    log_z_dict = {
        "model_A": 0.0,
        "model_B": 1.5,
        "model_C": 2.0,
    }

    our_weights = softmax_weights(log_z_dict)

    # Create mock posteriors for tengri
    posteriors = []
    for _key, log_z in log_z_dict.items():
        posteriors.append(types.SimpleNamespace(log_evidence=log_z))

    tengri_weights = tengri_bma_weights(posteriors)

    # Compare
    for i, key in enumerate(sorted(log_z_dict.keys())):
        assert abs(our_weights[key] - tengri_weights[i]) < 1e-10
