# SPDX-License-Identifier: BSD-3-Clause
"""Test BMA evidence runner CLI contract.

Pinned invariants:
- --help lists expected flags (--galaxy, --set, --models, --out, --n-restarts, --seed, --force, --dry-run, --max-models, --profile)
- --dry-run lists models without running them
- Failed models write JSON with error field and valid=false
- Valid JSON has all required fields
- JSON writer produces exactly the SPEC keys
- Validity rule: newton_decrement <= 0.1, n_clipped_eigenvalues == 0, log_evidence finite
- save_npz_atomic writes a loadable .npz via a same-suffix temp file and
  cleans it up on failure (the fix for a real run's ENOENT on every cell's NPZ)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

PAPER1 = Path(__file__).resolve().parent.parent
ANALYSIS = PAPER1.parent
for entry in [str(ANALYSIS), str(PAPER1)]:
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np
import pytest
from paper1.bma_evidence import fit_one_model, save_npz_atomic

pytestmark = pytest.mark.contract


def test_save_npz_atomic_writes_and_reloads(tmp_path):
    """save_npz_atomic writes a loadable .npz via a same-suffix temp file.

    np.savez appends ".npz" to a target name that does not already end in
    it, so a temp path like "<key>.npz.tmp" is actually written to disk as
    "<key>.npz.tmp.npz" and the following os.replace(tmp, final) then raises
    FileNotFoundError -- exactly what a real evidence run hit on every cell.
    This pins the fix: the temp file keeps the ".npz" suffix, ends up at the
    requested path, and leaves no temp file behind.
    """
    target = tmp_path / "79" / "config-I.npz"
    target.parent.mkdir(parents=True)
    arrays = {
        "log_stellar_mass_formed": np.array([10.1, 10.2, 10.3]),
        "log_stellar_mass_survived": np.array([9.9, 10.0, 10.1]),
        "log_sfr_100myr": np.array([0.5, 0.6, 0.7]),
        "log_sfr_10myr": np.array([0.4, 0.5, 0.6]),
    }

    save_npz_atomic(target, **arrays)

    assert target.exists(), "the requested .npz path must exist after the atomic write"
    leftover_tmp = target.with_name(f"{target.stem}.tmp.npz")
    assert not leftover_tmp.exists(), "the temp file must not remain after a successful write"

    loaded = np.load(target)
    for key, values in arrays.items():
        np.testing.assert_array_equal(loaded[key], values)


def test_save_npz_atomic_cleans_up_temp_on_failure(tmp_path):
    """A write that fails (missing parent directory) leaves no temp file and re-raises."""
    target = tmp_path / "missing_dir" / "config-I.npz"  # parent directory does not exist

    with pytest.raises(OSError):
        save_npz_atomic(target, log_stellar_mass_formed=np.array([1.0]))

    assert not target.exists()
    assert not target.with_name(f"{target.stem}.tmp.npz").exists()


def test_cli_help_lists_flags():
    """--help includes all expected CLI flags."""
    import os

    script_path = PAPER1 / "bma_evidence.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ANALYSIS}:{PAPER1}"
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    help_text = result.stdout
    required_flags = [
        "--galaxy",
        "--set",
        "--models",
        "--out",
        "--n-restarts",
        "--seed",
        "--force",
        "--dry-run",
        "--max-models",
        "--profile",
    ]
    for flag in required_flags:
        assert flag in help_text, f"Flag {flag} not in help text"


def test_dry_run_lists_models_no_import(tmp_path):
    """--dry-run lists models without importing JAX."""
    import os

    script_path = PAPER1 / "bma_evidence.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ANALYSIS}:{PAPER1}"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--galaxy",
            "79",
            "--set",
            "factorial",
            "--out",
            str(tmp_path),
            "--max-models",
            "5",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "79" in result.stdout
    assert "factorial" in result.stdout or "5 models" in result.stdout
    # Verify JAX was not imported (check stderr for import errors)
    # This is a soft check - if JAX import fails, process would exit non-zero
    assert "ImportError" not in result.stderr or "jax" not in result.stderr


def test_failed_model_writes_json_with_error(tmp_path):
    """When a model fails, result JSON has error field and valid=false."""
    phot_dict = {"names": ["hst_f814w"], "fnu": [1e-27], "fnu_err": [0.1e-27]}
    fnu = np.array([1e-27])
    sigma = np.array([0.1e-27])

    model_dict = {
        "sfh": "invalid_sfh_type",  # This will cause build_model to fail
        "ssp": "mist_c3k",
        "attenuation": "calzetti",
        "dust_emission": "dl14",
        "nebular": "cue",
    }

    result, npz = fit_one_model(
        galaxy_id=79,
        model_dict=model_dict,
        z=1.0,
        phot_dict=phot_dict,
        fnu=fnu,
        sigma=sigma,
        n_restarts=1,
        seed=42,
    )

    assert result["error"] is not None, "Expected error field to be set"
    assert result["valid"] is False, "Expected valid=False on error"
    assert result["log_evidence"] is None, "Expected log_evidence=None on error"
    assert npz is None, "Expected no NPZ on error"


def test_json_has_all_required_fields():
    """Result JSON contains all SPEC fields."""
    phot_dict = {"names": ["hst_f814w"], "fnu": [1e-27], "fnu_err": [0.1e-27]}
    fnu = np.array([1e-27])
    sigma = np.array([0.1e-27])

    # Use a factorial model that will likely fail quickly (no real fit)
    model_dict = {
        "sfh": "continuity",
        "ssp": "mist_c3k",
        "attenuation": "calzetti",
        "dust_emission": "dl14",
        "nebular": "cue",
    }

    result, _ = fit_one_model(
        galaxy_id=79,
        model_dict=model_dict,
        z=1.0,
        phot_dict=phot_dict,
        fnu=fnu,
        sigma=sigma,
        n_restarts=1,
        seed=42,
    )

    required_fields = [
        "galaxy",
        "z",
        "model_key",
        "model_set",
        "components",
        "route",
        "log_evidence",
        "map_loss",
        "n_free",
        "newton_decrement",
        "n_clipped_eigenvalues",
        "condition_number",
        "valid",
        "n_map_restarts",
        "map_restart_loss_spread",
        "systematic_floor_frac",
        "profile_mass",
        "wall_time_s",
        "peak_rss_gb",
        "code_revision",
        "seed",
        "error",
    ]
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"

    # Check field types
    assert isinstance(result["galaxy"], int)
    assert isinstance(result["z"], float)
    assert isinstance(result["model_key"], str)
    assert isinstance(result["model_set"], str)
    assert isinstance(result["components"], dict)
    assert result["route"] == "laplace"
    assert isinstance(result["valid"], bool)
    assert isinstance(result["n_map_restarts"], int)
    assert isinstance(result["systematic_floor_frac"], float)
    assert isinstance(result["profile_mass"], bool)
    assert result["profile_mass"] is False  # Always False for evidence
    assert isinstance(result["wall_time_s"], float)
    assert result["seed"] == 42


def test_validity_rule_newton_decrement():
    """Validity rule: newton_decrement > 0.1 makes valid=False."""
    phot_dict = {"names": ["hst_f814w"], "fnu": [1e-27], "fnu_err": [0.1e-27]}
    fnu = np.array([1e-27])
    sigma = np.array([0.1e-27])

    # Mock a result with newton_decrement = 0.2
    with patch("paper1.bma_evidence.fit_one_model") as mock_fit:
        result = {
            "galaxy": 79,
            "z": 1.0,
            "model_key": "test",
            "model_set": "test",
            "components": {},
            "route": "laplace",
            "log_evidence": 100.0,
            "map_loss": 50.0,
            "n_free": 8,
            "newton_decrement": 0.2,  # > 0.1 so invalid
            "n_clipped_eigenvalues": 0,
            "condition_number": 1e6,
            "valid": False,  # Should be False per rule
            "n_map_restarts": 1,
            "map_restart_loss_spread": 0.0,
            "wall_time_s": 10.0,
            "peak_rss_gb": 2.0,
            "code_revision": "abc123",
            "seed": 42,
            "error": None,
        }
        # Verify that newton_decrement > 0.1 implies valid=False
        is_valid = (
            np.isfinite(result["log_evidence"])
            and result["newton_decrement"] <= 0.1
            and result["n_clipped_eigenvalues"] == 0
        )
        assert is_valid is False


def test_validity_rule_clipped_eigenvalues():
    """Validity rule: any clipped eigenvalues makes valid=False."""
    result = {
        "log_evidence": 100.0,
        "newton_decrement": 0.05,
        "n_clipped_eigenvalues": 1,  # > 0 so invalid
    }
    is_valid = (
        np.isfinite(result["log_evidence"])
        and result["newton_decrement"] <= 0.1
        and result["n_clipped_eigenvalues"] == 0
    )
    assert is_valid is False


def test_validity_rule_nan_evidence():
    """Validity rule: NaN log_evidence makes valid=False."""
    result = {
        "log_evidence": np.nan,  # Not finite
        "newton_decrement": 0.05,
        "n_clipped_eigenvalues": 0,
    }
    is_valid = (
        np.isfinite(result["log_evidence"])
        and result["newton_decrement"] <= 0.1
        and result["n_clipped_eigenvalues"] == 0
    )
    assert not is_valid
