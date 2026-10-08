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

import itertools
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


# --- panel (a) fit, panel (b1) aggregation, panel (c2) limits ---------------


def _synthetic_set(set_name: str, n_gal: int = 3, sfr_by_model: dict | None = None) -> dict:
    """A summary whose ``set_name`` holds every expected model with distinct, normalized weights."""
    import numpy as np
    from paper1 import _bma_keys as bk

    keys = bk.expected_keys(set_name)
    rng = np.random.default_rng(7)
    galaxies = {}
    for g in range(n_gal):
        raw = rng.dirichlet(np.ones(len(keys)) * 0.3)
        models = []
        for key, weight in zip(keys, raw, strict=True):
            sfr = (sfr_by_model or {}).get((g, key), [0.5, 1.0, 1.5])
            models.append(
                {
                    "model_key": key,
                    "weight": float(weight),
                    "valid": True,
                    "percentiles": {
                        "log_stellar_mass_survived": [10.0, 10.5, 11.0],
                        "log_sfr_100myr": sfr,
                    },
                }
            )
        galaxies[str(1000 + g)] = {
            "sets": {
                set_name: {
                    "models": models,
                    "n_valid": len(models),
                    "close": True,
                    "bma_percentiles": {
                        "log_stellar_mass_survived": [10.0, 10.5, 11.0],
                        "log_sfr_100myr": [0.5, 1.0, 1.5],
                    },
                }
            }
        }
    return {"galaxies": galaxies}


def _build(summary: dict, set_name: str):
    import fig11_bma
    import matplotlib.pyplot as plt

    fig, drawn = fig11_bma.build_figure(summary, set_name)
    return fig, drawn, plt


def _b1_matrix(fig, n_rows: int):
    """The weight heatmap of panel (b1): the image with ``n_rows`` rows and a 0..1 scale."""
    for ax in fig.axes:
        for im in ax.images:
            if im.get_array().shape[0] == n_rows and im.get_clim() == (0.0, 1.0):
                return ax, im.get_array()
    raise AssertionError(f"no weight heatmap with {n_rows} rows")


def test_flowchart_boxes_fit_inside_figure_and_axes():
    from matplotlib.patches import FancyBboxPatch

    fig, _, plt = _build(_synthetic_set("named_all"), "named_all")
    fig.canvas.draw()
    ax_flow, ax_b1 = fig.axes[0], fig.axes[1]
    boxes = [p for p in ax_flow.patches if isinstance(p, FancyBboxPatch)]
    assert len(boxes) == 5
    fig_box, ax_box = fig.bbox, ax_flow.get_window_extent()
    for patch in boxes:
        ext = patch.get_window_extent()
        assert ext.x0 >= fig_box.x0 and ext.x1 <= fig_box.x1
        assert ext.x0 >= ax_box.x0 and ext.x1 <= ax_box.x1
    centre = 0.5 * (boxes[0].get_window_extent().x0 + boxes[-1].get_window_extent().x1)
    b1_box = ax_b1.get_window_extent()
    assert centre == pytest.approx(0.5 * (b1_box.x0 + b1_box.x1), abs=2.0)
    plt.close(fig)


def test_large_set_aggregates_to_top_rows_plus_other():
    import fig11_bma
    import numpy as np

    summary = _synthetic_set("factorial")
    n_models = 100
    fig, drawn, plt = _build(summary, "factorial")
    k = fig11_bma.B1_TOP_MODELS
    assert len(drawn.models_drawn) == k
    assert drawn.model_labels[-1] == f"Other ({n_models - k} models)"
    assert len(drawn.model_labels) == k + 1
    assert drawn.b1_n_aggregated == n_models - k
    assert len(set(drawn.models_drawn) | set(drawn.b1_aggregated_models)) == n_models
    assert not set(drawn.models_drawn) & set(drawn.b1_aggregated_models)

    totals = {
        key: sum(
            m["weight"]
            for g in summary["galaxies"].values()
            for m in g["sets"]["factorial"]["models"]
            if m["model_key"] == key
        )
        for key in (*drawn.models_drawn, *drawn.b1_aggregated_models)
    }
    assert min(totals[key] for key in drawn.models_drawn) >= max(
        totals[key] for key in drawn.b1_aggregated_models
    )

    _, matrix = _b1_matrix(fig, k + 1)
    np.testing.assert_allclose(np.asarray(matrix).sum(axis=0), 1.0, atol=1e-9)
    plt.close(fig)


def test_large_set_row_labels_do_not_overlap():
    fig, drawn, plt = _build(_synthetic_set("factorial"), "factorial")
    fig.canvas.draw()
    ax, _ = _b1_matrix(fig, len(drawn.model_labels))
    boxes = [t.get_window_extent() for t in ax.get_yticklabels()]
    assert len(boxes) == len(drawn.model_labels)
    for upper, lower in itertools.pairwise(boxes):
        assert upper.y0 >= lower.y1 - 0.5
    plt.close(fig)


@pytest.mark.parametrize("set_name", ["named_all", "named_grid"])
def test_named_sets_keep_one_row_per_model(set_name):
    from paper1 import _bma_keys as bk

    n_models = len(bk.expected_keys(set_name))
    fig, drawn, plt = _build(_synthetic_set(set_name), set_name)
    assert len(drawn.models_drawn) == n_models == len(drawn.model_labels)
    assert drawn.b1_n_aggregated == 0 and drawn.b1_aggregated_models == []
    assert not any(label.startswith("Other") for label in drawn.model_labels)
    _b1_matrix(fig, n_models)
    plt.close(fig)


def test_set_of_exactly_the_threshold_is_not_aggregated():
    import fig11_bma

    models = {f"m{i}": {"model_key": f"m{i}"} for i in range(fig11_bma.B1_MAX_ROWS)}
    weights = {0: {k: 1.0 / len(models) for k in models}}
    shown, folded = fig11_bma._split_rows(models, weights)
    assert len(shown) == fig11_bma.B1_MAX_ROWS and folded == []


def test_sfr_panel_ignores_floor_whisker_and_records_it():
    import fig11_bma
    from paper1 import _bma_keys as bk

    victim = bk.expected_keys("named_all")[1]
    summary = _synthetic_set("named_all", sfr_by_model={(1, victim): [-10.0, 0.4, 1.0]})
    fig, drawn, plt = _build(summary, "named_all")
    ax_sfr = fig.axes[-2]
    lo, hi = ax_sfr.get_ylim()
    assert lo > -10.0 and [lo, hi] == drawn.sfr_limits
    records = [r for r in drawn.sfr_out_of_range if r["model"] == victim]
    assert len(records) == 1
    assert records[0]["galaxy"] == 1001 and records[0]["clipped_low"] is True
    assert records[0]["side"] is None and records[0]["p16"] == -10.0
    assert not fig11_bma._out_of_range_record(1, "x", (0.5, 1.0, 1.5), drawn.sfr_limits)
    plt.close(fig)


def test_sfr_median_outside_limits_is_pinned_and_recorded():
    import fig11_bma
    from paper1 import _bma_keys as bk

    victim = bk.expected_keys("named_all")[2]
    summary = _synthetic_set("named_all", sfr_by_model={(0, victim): [-11.0, -10.0, -9.0]})
    fig, drawn, plt = _build(summary, "named_all")
    record = next(r for r in drawn.sfr_out_of_range if r["model"] == victim)
    assert record["side"] == "bottom" and record["median"] == -10.0
    assert drawn.sfr_limits[0] > -10.0
    assert fig11_bma._out_of_range_record(1, "m", (0.0, 1.0, 2.0), None) is None
    plt.close(fig)
