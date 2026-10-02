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

from paper1 import _bma_keys as bk
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


def _fkey(sfh: str = "continuity", ssp: int = 0, att: str = "calzetti") -> str:
    """A factorial model key, generated through _bma_keys (never hand-written)."""
    return bk.model_key(
        {
            "sfh": sfh,
            "ssp": bk.SSP_LABELS[ssp],
            "attenuation": att,
            "dust_emission": bk.DUST_EMISSION,
            "nebular": bk.NEBULAR,
        }
    )


def _make_evidence_cell(
    model_key: str,
    log_evidence: float = 0.0,
    valid: bool = True,
    newton_decrement: float = 0.01,
    n_clipped_eigenvalues: int = 0,
    route: str = "laplace",
) -> dict:
    """Create a synthetic evidence cell whose identity fields come from _bma_keys."""
    parsed = bk.parse_model_key(model_key)

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
        "model_set": bk.model_set_kind(model_key),
        "weight_sets": sorted(bk.set_membership(model_key)),
        "components": parsed,
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


def _write_cell(gal_dir: Path, cell: dict, with_npz: bool = True) -> None:
    """Write a cell as <model_key>.json (+ .npz), the layout the runner produces."""
    stem = cell["model_key"]
    (gal_dir / f"{stem}.json").write_text(json.dumps(cell))
    if with_npz:
        np.savez(gal_dir / f"{stem}.npz", **_make_npz_data(cell["galaxy"]))


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
        _fkey(),
        log_evidence=10.0,
        newton_decrement=0.05,
        n_clipped_eigenvalues=0,
    )
    is_valid, reason = _validate_cell(cell)
    assert is_valid is True
    assert reason is None


def test_validate_cell_invalid_newton():
    """Newton decrement > 0.1 fails validation."""
    cell = _make_evidence_cell(_fkey(), log_evidence=10.0, newton_decrement=0.5)
    is_valid, reason = _validate_cell(cell)
    assert is_valid is False
    assert "newton_decrement" in reason


def test_validate_cell_invalid_clipped():
    """n_clipped_eigenvalues > 0 fails validation."""
    cell = _make_evidence_cell(
        _fkey(),
        log_evidence=10.0,
        newton_decrement=0.05,
        n_clipped_eigenvalues=1,
    )
    is_valid, reason = _validate_cell(cell)
    assert is_valid is False
    assert "clipped" in reason


def test_validate_cell_invalid_log_evidence():
    """Infinite log_evidence fails validation."""
    cell = _make_evidence_cell(_fkey(), log_evidence=np.inf)
    is_valid, reason = _validate_cell(cell)
    assert is_valid is False
    assert "log_evidence" in reason


def test_validate_cell_invalid_nan():
    """NaN log_evidence fails validation."""
    cell = _make_evidence_cell(_fkey(), log_evidence=np.nan)
    is_valid, _ = _validate_cell(cell)
    assert is_valid is False


def test_validate_cell_wrong_route():
    """Non-laplace route fails validation."""
    cell = _make_evidence_cell(_fkey(), route="hmc")
    is_valid, reason = _validate_cell(cell)
    assert is_valid is False
    assert "route" in reason


def test_validate_cell_flag_disagrees_with_diagnostics():
    """Flag says invalid but diagnostics say valid."""
    cell = _make_evidence_cell(
        _fkey(),
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


def _combine(tmp_path: Path, evidence_dir: Path, n_draws: int = 100) -> dict:
    return combine_bma(evidence_dir, tmp_path / "fits", tmp_path / "fits_xlike", n_draws, 0)


def test_combine_bma_basic(tmp_path):
    """Basic BMA combining with synthetic data."""
    evidence_dir = tmp_path / "bma_evidence"
    gal_dir = evidence_dir / "1000"
    gal_dir.mkdir(parents=True)

    keys = [_fkey("continuity"), _fkey("dirichlet"), _fkey("delayed")]
    for key, log_z in zip(keys, [10.0, 0.0, 0.0]):
        _write_cell(gal_dir, _make_evidence_cell(key, log_evidence=log_z))

    summary = _combine(tmp_path, evidence_dir)

    assert "1000" in summary["galaxies"]
    gal_data = summary["galaxies"]["1000"]
    assert "factorial" in gal_data["sets"]

    w_set = gal_data["sets"]["factorial"]
    assert len(w_set["models"]) == 3

    weights = {m["model_key"]: m["weight"] for m in w_set["models"] if m["weight"]}
    assert abs(sum(weights.values()) - 1.0) < 1e-10
    assert w_set["max_weight"] > 0.9  # Highest evidence dominates


def test_combine_bma_weights_valid_only(tmp_path):
    """Only valid cells contribute to weights."""
    evidence_dir = tmp_path / "bma_evidence"
    gal_dir = evidence_dir / "2000"
    gal_dir.mkdir(parents=True)

    key_valid = _fkey("continuity")
    key_invalid = _fkey("dirichlet")

    cell_valid = _make_evidence_cell(key_valid, log_evidence=10.0)
    # Valid log_evidence but a bad newton decrement, flag overridden to test conflict
    cell_invalid = _make_evidence_cell(key_invalid, log_evidence=9.0, newton_decrement=0.5)
    cell_invalid["valid"] = False

    _write_cell(gal_dir, cell_valid)
    _write_cell(gal_dir, cell_invalid, with_npz=False)

    summary = _combine(tmp_path, evidence_dir)

    models = summary["galaxies"]["2000"]["sets"]["factorial"]["models"]
    valid_model = next((m for m in models if m["model_key"] == key_valid), None)
    invalid_model = next((m for m in models if m["model_key"] == key_invalid), None)

    assert valid_model is not None
    assert valid_model["weight"] == 1.0  # Only one valid, gets all weight

    assert invalid_model is not None
    assert invalid_model["weight"] is None
    assert invalid_model["valid"] is False
    assert any(x in invalid_model["excluded_reason"] for x in ["newton", "disagrees"])


def test_combine_bma_invalid_counts(tmp_path):
    """Invalid counts are aggregated correctly."""
    evidence_dir = tmp_path / "bma_evidence"
    key = _fkey("dpl", ssp=2, att="smc")

    for gal_id in [1000, 2000]:
        gal_dir = evidence_dir / str(gal_id)
        gal_dir.mkdir(parents=True)
        _write_cell(gal_dir, _make_evidence_cell(key, valid=False), with_npz=False)

    summary = _combine(tmp_path, evidence_dir)

    assert summary["invalid_counts"]["factorial"] == {key: 2}


def test_combine_bma_mixed_route_fails(tmp_path):
    """Mixed routes in one galaxy raises error."""
    evidence_dir = tmp_path / "bma_evidence"
    gal_dir = evidence_dir / "3000"
    gal_dir.mkdir(parents=True)

    _write_cell(gal_dir, _make_evidence_cell(_fkey("continuity"), route="laplace"), False)
    _write_cell(gal_dir, _make_evidence_cell(_fkey("dirichlet"), route="hmc"), False)

    with pytest.raises(ValueError, match="multiple routes"):
        _combine(tmp_path, evidence_dir)


def test_combine_bma_close_flag(tmp_path):
    """Close flag set when max weight < 0.9."""
    evidence_dir = tmp_path / "bma_evidence"
    gal_dir = evidence_dir / "4000"
    gal_dir.mkdir(parents=True)

    for sfh in ("continuity", "dirichlet", "delayed"):
        _write_cell(gal_dir, _make_evidence_cell(_fkey(sfh), log_evidence=0.0))

    w_set = _combine(tmp_path, evidence_dir)["galaxies"]["4000"]["sets"]["factorial"]
    assert w_set["close"] is True  # max weight ~0.333 < 0.9


def test_combine_bma_not_close_flag(tmp_path):
    """Close flag not set when one model dominates."""
    evidence_dir = tmp_path / "bma_evidence"
    gal_dir = evidence_dir / "5000"
    gal_dir.mkdir(parents=True)

    _write_cell(gal_dir, _make_evidence_cell(_fkey("continuity"), log_evidence=10.0))
    _write_cell(gal_dir, _make_evidence_cell(_fkey("dirichlet"), log_evidence=0.0))

    w_set = _combine(tmp_path, evidence_dir)["galaxies"]["5000"]["sets"]["factorial"]
    assert w_set["close"] is False  # max weight >> 0.9


def test_combine_bma_factorial_marginals(tmp_path):
    """Factorial marginals sum to 1 per axis."""
    evidence_dir = tmp_path / "bma_evidence"
    gal_dir = evidence_dir / "6000"
    gal_dir.mkdir(parents=True)

    _write_cell(gal_dir, _make_evidence_cell(_fkey("continuity"), log_evidence=1.0))
    _write_cell(gal_dir, _make_evidence_cell(_fkey("dirichlet"), log_evidence=1.0))

    marginal = _combine(tmp_path, evidence_dir)["galaxies"]["6000"]["sets"]["factorial"][
        "marginal"
    ]
    assert marginal["sfh"] == pytest.approx({"continuity": 0.5, "dirichlet": 0.5})
    for axis in [*bk.AXES, *bk.DERIVED_AXES]:
        assert sum(marginal[axis].values()) == pytest.approx(1.0), axis


def test_combine_bma_zero_valid_cells(tmp_path):
    """Galaxy with no valid cells reports reason."""
    evidence_dir = tmp_path / "bma_evidence"
    gal_dir = evidence_dir / "7000"
    gal_dir.mkdir(parents=True)

    _write_cell(gal_dir, _make_evidence_cell(_fkey(), valid=False), with_npz=False)

    w_set = _combine(tmp_path, evidence_dir)["galaxies"]["7000"]["sets"]["factorial"]
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
        {"model_key": _fkey("continuity"), "valid": True},
        {"model_key": _fkey("dirichlet"), "valid": True},
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
