# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end: runner cells -> combiner summary -> Figure 11, on the combiner's own output.

The figure's unit tests used hand-written summaries with short axis labels
("mist_c3k") while the combiner writes the registered grid names
("fsps_mist_c3k_a_chabrier"), so a figure that restated the axes passed its tests and
mislabeled, misordered and lost the priors of real output. Here every cell comes from
``bma_evidence.build_cell_result`` + ``write_cell`` (fake Laplace results with varied
evidences), ``combine_bma`` writes the summary, and ``fig11_bma.py`` runs on it as a
subprocess, exactly as a reader would. The expected labels and priors are written out in
this file, not derived from ``_bma_keys``, so a mistake there fails here too.

Pinned invariants:
- fig11 exits 0 and writes a PDF and a sidecar on real combiner output;
- the ssp rows are the five display labels in axis order, never raw grid names;
- every marginal row carries the design prior (0.2 per ssp value, 0.4/0.4/0.2 isochrone,
  0.6/0.4 spectral library), read from the combiner's ``prior_mass``;
- the invalid cell is hatched and listed as excluded, and the galaxy with no factorial set
  is a hatched column in every marginal row;
- named rows are I..V then the five X-like labels;
- a summary prior that disagrees with the design prior on a full factorial set is warned.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PAPER1 = Path(__file__).resolve().parents[1]
ANALYSIS = PAPER1.parent
REPO_ROOT = ANALYSIS.parent
for entry in (str(ANALYSIS), str(PAPER1)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from paper1 import _bma_keys as bk
from paper1.bma_combine import combine_bma
from paper1.bma_evidence import build_cell_result, write_cell

pytestmark = pytest.mark.contract

FIG11 = PAPER1 / "fig11_bma.py"
GALAXIES = (101, 102, 103)
INVALID_GALAXY, INVALID_MODEL = 102, "config-III"
NO_FACTORIAL_GALAXY = 103

EXPECTED_SSP_ROWS = ["MIST + C3K", "PARSEC + C3K", "MIST + MILES", "PARSEC + MILES", "BPASS + C3K"]
EXPECTED_NAMED_ROWS = [
    "I",
    "II",
    "III",
    "IV",
    "V",
    "CIGALE-like",
    "Prospector-like",
    "BAGPIPES-like",
    "BEAGLE-like",
    "Dense Basis-like",
]
DESIGN_PRIOR = {
    "sfh": [0.2] * 5,
    "ssp": [0.2] * 5,
    "attenuation": [0.25] * 4,
    "isochrone": [0.4, 0.4, 0.2],
    "spectral_library": [0.6, 0.4],
}


def _laplace(log_z: float, newton: float = 0.01):
    diagnostics = {
        "newton_decrement": newton,
        "n_clipped_eigenvalues": 0,
        "condition_number": 1e4,
    }
    return SimpleNamespace(log_evidence=log_z, diagnostics=diagnostics)


def _npz(seed: int, n: int = 30) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "log_stellar_mass_formed": rng.normal(10.5, 0.1, n),
        "log_stellar_mass_survived": rng.normal(10.3, 0.1, n),
        "log_sfr_100myr": rng.normal(0.5, 0.1, n),
        "log_sfr_10myr": rng.normal(0.4, 0.1, n),
    }


def _write_cells(evidence_dir: Path) -> None:
    """Write every galaxy's cells through the runner's assembler and writer."""
    models = bk.enumerate_named_all() + bk.enumerate_factorial()
    for galaxy in GALAXIES:
        gal_dir = evidence_dir / str(galaxy)
        gal_dir.mkdir(parents=True)
        rng = np.random.default_rng(galaxy)
        for i, model_dict in enumerate(models):
            if galaxy == NO_FACTORIAL_GALAXY and "config" not in model_dict:
                continue
            key = bk.model_key(model_dict)
            invalid = galaxy == INVALID_GALAXY and key == INVALID_MODEL
            log_z = -1000.0 - 6.0 * rng.random() * (1 + i % 5)
            result = build_cell_result(
                galaxy_id=galaxy,
                z=1.0,
                model_dict=model_dict,
                laplace=_laplace(log_z, newton=5.0 if invalid else 0.01),
                started=time.time(),
                seed=0,
            )
            write_cell(gal_dir, result, _npz(1000 * galaxy + i))


def _run_fig11(summary_path: Path, out: Path, sidecar: Path):
    return subprocess.run(
        [
            sys.executable,
            str(FIG11),
            "--summary",
            str(summary_path),
            "--out",
            str(out),
            "--data-out",
            str(sidecar),
        ],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(Path.home()),
            "PYTHONPATH": str(REPO_ROOT / "src") + ":" + str(ANALYSIS),
        },
        timeout=120,
    )


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """(summary dict, summary path, out PDF, sidecar dict, subprocess result)."""
    root = tmp_path_factory.mktemp("bma_pipeline")
    evidence_dir = root / "bma_evidence"
    _write_cells(evidence_dir)
    summary = combine_bma(evidence_dir, root / "fits", root / "fits_xlike", n_draws=200, seed=0)
    summary_path = root / "bma_summary.json"
    summary_path.write_text(json.dumps(summary))
    out, data = root / "fig11_bma.pdf", root / "fig11_bma.json"
    result = _run_fig11(summary_path, out, data)
    sidecar = json.loads(data.read_text()) if data.exists() else {}
    return summary, summary_path, out, sidecar, result


def test_fig11_runs_on_combiner_output(pipeline):
    _, _, out, sidecar, result = pipeline
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert out.exists() and out.stat().st_size > 0
    assert sidecar["galaxies_drawn"] == list(GALAXIES)
    assert sidecar["prior_warnings"] == []


def test_ssp_rows_are_display_labels_in_axis_order(pipeline):
    _, _, _, sidecar, _ = pipeline
    assert sidecar["marginal_row_labels"]["ssp"] == EXPECTED_SSP_ROWS
    for label in sidecar["marginal_row_labels"]["ssp"]:
        assert "fsps" not in label and "_" not in label


def test_every_marginal_row_carries_the_design_prior(pipeline):
    _, _, _, sidecar, _ = pipeline
    marginals = sidecar["factorial_marginals"]
    for axis, expected in DESIGN_PRIOR.items():
        priors = [cell["prior"] for cell in marginals[axis].values()]
        assert priors == pytest.approx(expected), axis
        assert len(marginals[axis]) == len(expected)
        for cell in marginals[axis].values():
            assert set(cell["weight"]) == {"101", "102"}, "weights only for galaxies with a set"


def test_invalid_cell_is_hatched_and_excluded(pipeline):
    _, _, _, sidecar, _ = pipeline
    assert [INVALID_MODEL, INVALID_GALAXY] in sidecar["hatched_weight_cells"]
    assert len(sidecar["hatched_weight_cells"]) == 1
    assert [INVALID_MODEL, "newton_decrement=5.0"] in sidecar["excluded_models"]["named_all"]


def test_galaxy_without_factorial_set_is_a_hatched_column(pipeline):
    _, _, _, sidecar, _ = pipeline
    marginals = sidecar["factorial_marginals"]
    assert marginals["galaxies_missing_factorial"] == [NO_FACTORIAL_GALAXY]
    for axis in DESIGN_PRIOR:
        for value, gals in marginals["hatched_cells"][axis].items():
            assert gals == [NO_FACTORIAL_GALAXY], (axis, value)
        assert len(marginals["hatched_cells"][axis]) == len(marginals[axis])


def test_named_rows_are_configs_then_xlike_labels(pipeline):
    _, _, _, sidecar, _ = pipeline
    assert sidecar["model_labels"] == EXPECTED_NAMED_ROWS
    assert sidecar["models_drawn"][:5] == [f"config-{c}" for c in ("I", "II", "III", "IV", "V")]


def test_disagreeing_summary_prior_is_warned(pipeline, tmp_path):
    summary, _, _, _, _ = pipeline
    tampered = json.loads(json.dumps(summary))
    first = bk.SSP_LABELS[0]
    tampered["prior_mass"]["factorial"]["ssp"][first] = 0.35
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered))
    out, data = tmp_path / "fig.pdf", tmp_path / "fig.json"
    result = _run_fig11(path, out, data)
    assert result.returncode == 0, result.stderr
    warnings = json.loads(data.read_text())["prior_warnings"]
    assert warnings and all(f"ssp/{first}" in w for w in warnings)
    assert "prior mismatch" in result.stderr
