# SPDX-License-Identifier: BSD-3-Clause
"""``fig11_bma.py`` contract tests for BMA figure generation.

The figure reads a BMA summary JSON written by bma_combine.py, validates that
it contains valid models and weight sets with valid weights summing to 1 within
1e-6 tolerance, and refuses if the summary is missing, empty, or has invalid
weights. Excluded/invalid cells are drawn hatched. Close galaxies are read from
the summary set's "close" field. The sidecar JSON lists exactly which galaxies
and models were drawn, and flags which are close. Configuration VI entries are
dropped. Missing summary → non-zero exit, no PDF.

These tests pin the contract between the combiner and the figure.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ANALYSIS_DIR.parents[1]
FIG11 = ANALYSIS_DIR / "fig11_bma.py"

sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(ANALYSIS_DIR.parent))

pytestmark = pytest.mark.contract


def _run_fig11(
    summary_path: Path,
    out_pdf: Path,
    set_name: str = "named_all",
    data_out: Path | None = None,
) -> subprocess.CompletedProcess:
    """Invoke fig11_bma.py exactly as a reader would, never raising on failure."""
    cmd = [
        sys.executable,
        str(FIG11),
        "--summary",
        str(summary_path),
        "--out",
        str(out_pdf),
        "--set",
        set_name,
    ]
    if data_out:
        cmd.extend(["--data-out", str(data_out)])

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(Path.home()),
            "PYTHONPATH": str(REPO_ROOT / "src") + ":" + str(REPO_ROOT / "analysis"),
        },
        timeout=30,
    )


def _build_minimal_summary(tmp_path: Path) -> dict:
    """Build a minimal valid BMA summary structure."""
    return {
        "route": "laplace",
        "code_revision": "test_abc123",
        "rules": "flat prior over valid models",
        "galaxies": {
            "100001": {
                "z": 0.5,
                "sets": {
                    "named_grid": {
                        "models": [
                            {
                                "model_key": "I",
                                "config": "I",
                                "components": {"sfh": "exp", "ssp": "bc03"},
                                "log_evidence": -1234.5,
                                "weight": 0.5,
                                "valid": True,
                                "excluded_reason": None,
                                "nuts_adoption_pass": True,
                                "percentiles": {
                                    "log_stellar_mass_survived": [10.0, 10.5, 11.0],
                                    "log_stellar_mass_formed": [10.1, 10.6, 11.1],
                                    "log_sfr_100myr": [-1.0, -0.5, 0.0],
                                    "log_sfr_10myr": [-1.5, -1.0, -0.5],
                                },
                            },
                            {
                                "model_key": "II",
                                "config": "II",
                                "components": {"sfh": "delayed_tau", "ssp": "bc03"},
                                "log_evidence": -1240.0,
                                "weight": 0.5,
                                "valid": True,
                                "excluded_reason": None,
                                "nuts_adoption_pass": False,
                                "percentiles": {
                                    "log_stellar_mass_survived": [10.1, 10.6, 11.1],
                                    "log_stellar_mass_formed": [10.2, 10.7, 11.2],
                                    "log_sfr_100myr": [-1.1, -0.6, -0.1],
                                    "log_sfr_10myr": [-1.6, -1.1, -0.6],
                                },
                            },
                        ],
                        "max_weight": 0.5,
                        "close": False,
                        "n_valid": 2,
                    }
                },
            }
        },
    }


def test_missing_summary_exits_nonzero_no_pdf(tmp_path):
    """Missing summary file → non-zero exit, no PDF written."""
    missing_summary = tmp_path / "nonexistent_summary.json"
    out_pdf = tmp_path / "fig11_bma.pdf"
    data_out = tmp_path / "data.json"

    result = _run_fig11(missing_summary, out_pdf, data_out=data_out)

    assert result.returncode != 0, (
        "fig11 exited 0 on missing summary; it should refuse.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert not out_pdf.exists(), f"PDF written despite missing summary: {out_pdf}"
    assert not data_out.exists(), f"Sidecar written despite missing summary: {data_out}"
    assert "bma_combine.py" in result.stderr or "Cannot read" in result.stderr, (
        f"Error message did not name bma_combine.py or indicate reading failure.\n"
        f"stderr:\n{result.stderr}"
    )


def test_empty_summary_exits_nonzero_no_pdf(tmp_path):
    """Empty summary (no galaxies) → non-zero exit, no PDF written."""
    summary_path = tmp_path / "bma_summary.json"
    summary_path.write_text(json.dumps({"galaxies": {}}))

    out_pdf = tmp_path / "fig11_bma.pdf"

    result = _run_fig11(summary_path, out_pdf)

    assert result.returncode != 0, (
        "fig11 exited 0 on empty summary; it should refuse.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert not out_pdf.exists(), f"PDF written despite empty summary: {out_pdf}"


def test_valid_summary_writes_pdf_and_sidecar(tmp_path):
    """Valid summary → PDF and sidecar JSON written, exit 0."""
    summary = _build_minimal_summary(tmp_path)
    summary_path = tmp_path / "bma_summary.json"
    summary_path.write_text(json.dumps(summary))

    out_pdf = tmp_path / "fig11_bma.pdf"
    data_out = tmp_path / "data.json"

    result = _run_fig11(summary_path, out_pdf, data_out=data_out)

    assert result.returncode == 0, (
        f"fig11 failed on valid summary.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert out_pdf.exists(), f"PDF not written: {out_pdf}"
    assert data_out.exists(), f"Sidecar not written: {data_out}"

    # Validate sidecar
    with open(data_out) as f:
        sidecar = json.load(f)
    assert "galaxies_drawn" in sidecar
    assert "models_drawn" in sidecar
    assert isinstance(sidecar["galaxies_drawn"], list)
    assert isinstance(sidecar["models_drawn"], list)


def test_excluded_models_marked_in_sidecar(tmp_path):
    """Excluded models listed in sidecar with reason, not drawn as weights."""
    summary = _build_minimal_summary(tmp_path)
    # Add an excluded model
    summary["galaxies"]["100001"]["sets"]["named_grid"]["models"].append(
        {
            "model_key": "III",
            "config": "III",
            "components": {},
            "log_evidence": None,
            "weight": None,
            "valid": False,
            "excluded_reason": "divergences exceeded threshold",
            "nuts_adoption_pass": False,
            "percentiles": {},
        }
    )

    summary_path = tmp_path / "bma_summary.json"
    summary_path.write_text(json.dumps(summary))

    out_pdf = tmp_path / "fig11_bma.pdf"
    data_out = tmp_path / "data.json"

    result = _run_fig11(summary_path, out_pdf, data_out=data_out)

    assert result.returncode == 0
    with open(data_out) as f:
        sidecar = json.load(f)

    assert "excluded_models" in sidecar
    if "named_grid" in sidecar["excluded_models"]:
        # Check that excluded model is listed with reason
        excluded_list = sidecar["excluded_models"]["named_grid"]
        assert any(e[0] == "III" for e in excluded_list), (
            f"III not in excluded models: {excluded_list}"
        )


def test_close_galaxy_flagged(tmp_path):
    """Galaxy with close=True in set is flagged in sidecar."""
    summary = _build_minimal_summary(tmp_path)
    # Mark galaxy as close
    summary["galaxies"]["100001"]["sets"]["named_grid"]["close"] = True
    summary["galaxies"]["100001"]["sets"]["named_grid"]["max_weight"] = 0.6

    summary_path = tmp_path / "bma_summary.json"
    summary_path.write_text(json.dumps(summary))

    out_pdf = tmp_path / "fig11_bma.pdf"
    data_out = tmp_path / "data.json"

    result = _run_fig11(summary_path, out_pdf, data_out=data_out)

    assert result.returncode == 0
    with open(data_out) as f:
        sidecar = json.load(f)

    assert "close_galaxies" in sidecar
    assert 100001 in sidecar["close_galaxies"]


def test_weights_not_summing_to_one_exits_nonzero(tmp_path):
    """Weights that don't sum to 1 within 1e-6 → non-zero exit."""
    summary = _build_minimal_summary(tmp_path)
    # Break weight sum (0.8 instead of 1.0)
    summary["galaxies"]["100001"]["sets"]["named_grid"]["models"][0]["weight"] = 0.4
    summary["galaxies"]["100001"]["sets"]["named_grid"]["models"][1]["weight"] = 0.4

    summary_path = tmp_path / "bma_summary.json"
    summary_path.write_text(json.dumps(summary))

    out_pdf = tmp_path / "fig11_bma.pdf"

    result = _run_fig11(summary_path, out_pdf)

    assert result.returncode != 0, (
        f"fig11 exited 0 on weights summing to 0.8; should refuse.\nstderr:\n{result.stderr}"
    )
    assert not out_pdf.exists(), f"PDF written despite invalid weights: {out_pdf}"


def test_fallback_to_named_grid_when_named_all_absent(tmp_path):
    """When named_all absent, falls back to named_grid."""
    summary = _build_minimal_summary(tmp_path)
    # No named_all, only named_grid
    summary_path = tmp_path / "bma_summary.json"
    summary_path.write_text(json.dumps(summary))

    out_pdf = tmp_path / "fig11_bma.pdf"
    data_out = tmp_path / "data.json"

    result = _run_fig11(summary_path, out_pdf, set_name="named_all", data_out=data_out)

    # Should succeed with fallback
    assert result.returncode == 0, (
        f"fig11 failed to fall back to named_grid.\nstderr:\n{result.stderr}"
    )
    with open(data_out) as f:
        sidecar = json.load(f)
    assert sidecar["selected_set"] == "named_grid"


def test_configuration_vi_is_filtered(tmp_path):
    """Configuration VI entries are dropped from summary with a note."""
    summary = _build_minimal_summary(tmp_path)
    # Add a Configuration VI entry (should be filtered)
    summary["galaxies"]["100001"]["sets"]["named_grid"]["models"].append(
        {
            "model_key": "VI",
            "config": "VI",
            "components": {},
            "log_evidence": -1250.0,
            "weight": 0.0,
            "valid": True,
            "excluded_reason": None,
            "nuts_adoption_pass": True,
            "percentiles": {
                "log_stellar_mass_survived": [10.3, 10.8, 11.3],
                "log_stellar_mass_formed": [10.4, 10.9, 11.4],
                "log_sfr_100myr": [-1.3, -0.8, -0.3],
                "log_sfr_10myr": [-1.8, -1.3, -0.8],
            },
        }
    )

    summary_path = tmp_path / "bma_summary.json"
    summary_path.write_text(json.dumps(summary))

    out_pdf = tmp_path / "fig11_bma.pdf"
    data_out = tmp_path / "data.json"

    result = _run_fig11(summary_path, out_pdf, data_out=data_out)

    assert result.returncode == 0
    # VI should not be in the drawn models (it was filtered)
    with open(data_out) as f:
        sidecar = json.load(f)
    assert "VI" not in sidecar["models_drawn"]
    # Note should be in stderr
    assert "Configuration VI" in result.stderr


def test_xlike_models_included_in_named_all(tmp_path):
    """X-like models (cigale_like, etc.) included in named_all set."""
    summary = _build_minimal_summary(tmp_path)
    # Add named_all with X-like models
    summary["galaxies"]["100001"]["sets"]["named_all"] = {
        "models": [
            {
                "model_key": "I",
                "config": "I",
                "components": {},
                "log_evidence": -1234.5,
                "weight": 0.3,
                "valid": True,
                "excluded_reason": None,
                "nuts_adoption_pass": True,
                "percentiles": {
                    "log_stellar_mass_survived": [10.0, 10.5, 11.0],
                    "log_stellar_mass_formed": [10.1, 10.6, 11.1],
                    "log_sfr_100myr": [-1.0, -0.5, 0.0],
                    "log_sfr_10myr": [-1.5, -1.0, -0.5],
                },
            },
            {
                "model_key": "cigale_like",
                "components": {"sfh": "custom"},
                "log_evidence": -1235.0,
                "weight": 0.7,
                "valid": True,
                "excluded_reason": None,
                "nuts_adoption_pass": None,
                "percentiles": {
                    "log_stellar_mass_survived": [10.1, 10.6, 11.1],
                    "log_stellar_mass_formed": [10.2, 10.7, 11.2],
                    "log_sfr_100myr": [-1.1, -0.6, -0.1],
                    "log_sfr_10myr": [-1.6, -1.1, -0.6],
                },
            },
        ],
        "max_weight": 0.7,
        "close": False,
        "n_valid": 2,
    }

    summary_path = tmp_path / "bma_summary.json"
    summary_path.write_text(json.dumps(summary))

    out_pdf = tmp_path / "fig11_bma.pdf"
    data_out = tmp_path / "data.json"

    result = _run_fig11(summary_path, out_pdf, set_name="named_all", data_out=data_out)

    assert result.returncode == 0
    with open(data_out) as f:
        sidecar = json.load(f)
    # Both I and cigale_like should be in models_drawn
    assert "I" in sidecar["models_drawn"]
    assert "cigale_like" in sidecar["models_drawn"]


def _add_factorial_set(summary: dict, gal_id_str: str, marginal: dict) -> None:
    """Attach a minimal "factorial" weight set with the given marginal to one galaxy."""
    summary["galaxies"][gal_id_str]["sets"]["factorial"] = {
        "models": [],
        "max_weight": None,
        "close": False,
        "n_valid": 0,
        "n_expected": 100,
        "bma_percentiles": {},
        "marginal": marginal,
    }


def test_factorial_marginal_panel_draws_and_records_weight_and_prior(tmp_path):
    """Panel (b2) draws, and the sidecar records weight and prior per axis value."""
    summary = _build_minimal_summary(tmp_path)
    _add_factorial_set(
        summary,
        "100001",
        {
            "sfh": {
                "continuity": {"weight": 0.7, "prior": 0.25},
                "dirichlet": {"weight": 0.3, "prior": 0.75},
            }
        },
    )

    summary_path = tmp_path / "bma_summary.json"
    summary_path.write_text(json.dumps(summary))

    out_pdf = tmp_path / "fig11_bma.pdf"
    data_out = tmp_path / "data.json"

    result = _run_fig11(summary_path, out_pdf, data_out=data_out)

    assert result.returncode == 0, (
        f"fig11 failed to draw the factorial marginal panel.\nstderr:\n{result.stderr}"
    )
    assert out_pdf.exists()

    with open(data_out) as f:
        sidecar = json.load(f)

    assert "factorial_marginals" in sidecar
    sfh = sidecar["factorial_marginals"]["sfh"]
    assert sfh["continuity"]["prior"] == pytest.approx(0.25)
    assert sfh["continuity"]["weight"]["100001"] == pytest.approx(0.7)
    assert sfh["dirichlet"]["prior"] == pytest.approx(0.75)
    assert sfh["dirichlet"]["weight"]["100001"] == pytest.approx(0.3)


def test_factorial_marginal_ratio_uses_prior(tmp_path):
    """log2(weight / prior) actually depends on the prior, not just the weight.

    Two axis values carry the same weight (0.5) but different priors: the recorded
    log2_ratio must differ between them. A third value carries no prior at all, and
    must fall back to the documented flat-prior design constant (0.25 for a 4-value
    attenuation axis) rather than being dropped.
    """
    summary = _build_minimal_summary(tmp_path)
    _add_factorial_set(
        summary,
        "100001",
        {
            "attenuation": {
                "calzetti": {"weight": 0.5, "prior": 0.25},  # ratio 2 -> log2 = 1.0
                "smc": {"weight": 0.5, "prior": 0.5},  # ratio 1 -> log2 = 0.0
                "kriek_conroy_2c": 0.5,  # bare weight, no prior -> falls back to 0.25
            }
        },
    )

    summary_path = tmp_path / "bma_summary.json"
    summary_path.write_text(json.dumps(summary))

    out_pdf = tmp_path / "fig11_bma.pdf"
    data_out = tmp_path / "data.json"

    result = _run_fig11(summary_path, out_pdf, data_out=data_out)
    assert result.returncode == 0, f"stderr:\n{result.stderr}"

    with open(data_out) as f:
        sidecar = json.load(f)

    attenuation = sidecar["factorial_marginals"]["attenuation"]
    assert attenuation["calzetti"]["log2_ratio"]["100001"] == pytest.approx(1.0, abs=1e-9)
    assert attenuation["smc"]["log2_ratio"]["100001"] == pytest.approx(0.0, abs=1e-9)
    # Same weight (0.5) as "smc" but a different prior must give a different ratio.
    assert (
        attenuation["calzetti"]["log2_ratio"]["100001"]
        != attenuation["smc"]["log2_ratio"]["100001"]
    )
    # No prior supplied for kriek_conroy_2c -> falls back to the flat-prior design constant.
    assert attenuation["kriek_conroy_2c"]["prior"] == pytest.approx(0.25)
    assert attenuation["kriek_conroy_2c"]["log2_ratio"]["100001"] == pytest.approx(1.0, abs=1e-9)


def test_galaxy_missing_factorial_set_does_not_crash(tmp_path):
    """A galaxy with no "factorial" weight set is drawn as a hatched column, not a crash."""
    summary = _build_minimal_summary(tmp_path)
    _add_factorial_set(summary, "100001", {"sfh": {"continuity": {"weight": 1.0, "prior": 0.2}}})

    # Second galaxy has only named_grid, no "factorial" key at all.
    summary["galaxies"]["100002"] = {
        "z": 0.6,
        "sets": {
            "named_grid": {
                "models": [
                    {
                        "model_key": "I",
                        "config": "I",
                        "components": {},
                        "log_evidence": -1235.0,
                        "weight": 1.0,
                        "valid": True,
                        "excluded_reason": None,
                        "nuts_adoption_pass": True,
                        "percentiles": {
                            "log_stellar_mass_survived": [10.1, 10.6, 11.1],
                            "log_stellar_mass_formed": [10.2, 10.7, 11.2],
                            "log_sfr_100myr": [-1.1, -0.6, -0.1],
                            "log_sfr_10myr": [-1.6, -1.1, -0.6],
                        },
                    }
                ],
                "max_weight": 1.0,
                "close": False,
                "n_valid": 1,
            }
        },
    }

    summary_path = tmp_path / "bma_summary.json"
    summary_path.write_text(json.dumps(summary))

    out_pdf = tmp_path / "fig11_bma.pdf"
    data_out = tmp_path / "data.json"

    result = _run_fig11(summary_path, out_pdf, data_out=data_out)

    assert result.returncode == 0, (
        f"fig11 crashed on a galaxy missing the factorial set.\nstderr:\n{result.stderr}"
    )
    assert out_pdf.exists()

    with open(data_out) as f:
        sidecar = json.load(f)

    assert sidecar["factorial_marginals"]["galaxies_missing_factorial"] == [100002]


def test_multiple_galaxies_draws_all(tmp_path):
    """Multiple galaxies in summary → all listed in sidecar."""
    summary = _build_minimal_summary(tmp_path)
    # Add a second galaxy
    summary["galaxies"]["100002"] = {
        "z": 0.6,
        "sets": {
            "named_grid": {
                "models": [
                    {
                        "model_key": "I",
                        "config": "I",
                        "components": {},
                        "log_evidence": -1235.0,
                        "weight": 0.5,
                        "valid": True,
                        "excluded_reason": None,
                        "nuts_adoption_pass": True,
                        "percentiles": {
                            "log_stellar_mass_survived": [10.1, 10.6, 11.1],
                            "log_stellar_mass_formed": [10.2, 10.7, 11.2],
                            "log_sfr_100myr": [-1.1, -0.6, -0.1],
                            "log_sfr_10myr": [-1.6, -1.1, -0.6],
                        },
                    },
                    {
                        "model_key": "II",
                        "config": "II",
                        "components": {},
                        "log_evidence": -1241.0,
                        "weight": 0.5,
                        "valid": True,
                        "excluded_reason": None,
                        "nuts_adoption_pass": True,
                        "percentiles": {
                            "log_stellar_mass_survived": [10.2, 10.7, 11.2],
                            "log_stellar_mass_formed": [10.3, 10.8, 11.3],
                            "log_sfr_100myr": [-1.2, -0.7, -0.2],
                            "log_sfr_10myr": [-1.7, -1.2, -0.7],
                        },
                    },
                ],
                "max_weight": 0.5,
                "close": False,
                "n_valid": 2,
            }
        },
    }

    summary_path = tmp_path / "bma_summary.json"
    summary_path.write_text(json.dumps(summary))

    out_pdf = tmp_path / "fig11_bma.pdf"
    data_out = tmp_path / "data.json"

    result = _run_fig11(summary_path, out_pdf, data_out=data_out)

    assert result.returncode == 0
    with open(data_out) as f:
        sidecar = json.load(f)

    assert 100001 in sidecar["galaxies_drawn"]
    assert 100002 in sidecar["galaxies_drawn"]
    assert len(sidecar["galaxies_drawn"]) == 2


def test_every_axis_value_and_named_id_has_a_display_label():
    """The figure labels through ``display_label``; an unlabeled value would raise at draw time."""
    from paper1 import _bma_keys as bk

    for axis in (*bk.AXES, *bk.DERIVED_AXES):
        for value in bk.axis_values(axis):
            assert bk.display_label(axis, value), (axis, value)
    for named in (*bk.GRID_IDS, *bk.XLIKE_IDS):
        assert bk.display_label("named", named), named
    with pytest.raises(ValueError, match="No display label"):
        bk.display_label("ssp", "mist_c3k")
    with pytest.raises(ValueError, match="No display labels for axis"):
        bk.display_label("nonsense", "x")
