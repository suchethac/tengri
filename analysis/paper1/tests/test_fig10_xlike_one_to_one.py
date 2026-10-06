# SPDX-License-Identifier: BSD-3-Clause
"""Figure 10 X-like one-to-one comparison - contract and integration tests.

This module pins critical behaviors:
1. Every code drawn has at least one adopted cell and a marker in CODE_MARKERS
2. Empty directory exits non-zero with no PDF
3. Non-adopted cells are counted but not drawn
4. Prospector uses formed mass, other codes use surviving mass
5. All codes use sfr_100myr for SFR comparison
6. Missing surviving mass JSON exits non-zero with command in error message
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ANALYSIS_DIR.parents[1]
FIG10_SCRIPT = ANALYSIS_DIR / "fig10_xlike_one_to_one.py"

sys.path.insert(0, str(ANALYSIS_DIR))

from fig10_xlike_one_to_one import CENSUS_COMMAND, CODE_MARKERS, XLIKE_CODE

from analysis.paper1.tests._xlike_cells import surviving_census, write_xlike_cell

pytestmark = pytest.mark.contract


def _run_fig10(
    results_dir: Path,
    surviving_path: Path | None,
    published_path: Path,
    out_dir: Path,
    selected_galaxies_path: Path | None = None,
) -> subprocess.CompletedProcess:
    """Invoke fig10 exactly as a reader would, returning result without raising."""
    cmd = [
        sys.executable,
        str(FIG10_SCRIPT),
        "--results-dir",
        str(results_dir),
        "--published",
        str(published_path),
        "--out",
        str(out_dir / "fig10.pdf"),
        "--data-out",
        str(out_dir / "fig10_data.json"),
    ]
    if surviving_path:
        cmd.extend(["--surviving", str(surviving_path)])
    if selected_galaxies_path:
        cmd.extend(["--selected-galaxies", str(selected_galaxies_path)])
    return subprocess.run(cmd, capture_output=True, text=True)


def test_empty_results_dir_exits_nonzero(tmp_path: Path) -> None:
    """Empty results directory must exit with error and produce no PDF."""
    results_dir = tmp_path / "empty"
    results_dir.mkdir()

    published_csv = tmp_path / "art_sedfitting_z1.csv"
    published_csv.write_text(
        "code,id,logmstar,logmstar_lo,logmstar_hi,logsfr,logsfr_lo,logsfr_hi\n"
    )

    surviving_json = tmp_path / "surviving.json"
    surviving_json.write_text(json.dumps({"cells": {}}))

    result = _run_fig10(results_dir, surviving_json, published_csv, tmp_path)

    assert result.returncode != 0, "Empty results dir should exit non-zero"
    assert not (tmp_path / "fig10.pdf").exists(), "No PDF should be written on error"
    assert "no adopted x-like cells found" in result.stderr.lower()


def test_no_adopted_cells_exits_nonzero(tmp_path: Path) -> None:
    """When all cells are non-adopted, must exit non-zero."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create a non-adopted X-like cell
    gal_id = 21
    xlike_key = "cigale_like"
    write_xlike_cell(results_dir, gal_id, xlike_key, adopted=False)

    # Create CSV with published data
    published_csv = tmp_path / "art_sedfitting_z1.csv"
    published_csv.write_text(
        "code,id,logmstar,logmstar_lo,logmstar_hi,logsfr,logsfr_lo,logsfr_hi\n"
        f"CIGALE,{gal_id},10.0,9.8,10.2,0.5,0.3,0.7\n"
    )

    # Create selected galaxies
    selected_gals = tmp_path / "selected_galaxies_20.json"
    selected_gals.write_text(json.dumps({"selected_galaxies": [{"id": gal_id}]}))

    # Surviving mass JSON
    surviving_json = tmp_path / "surviving.json"
    surviving_json.write_text(json.dumps({"cells": {}}))

    # Override SELECTION_20 and CANONICAL_PUBLISHED_CSV paths
    import sys

    old_argv = sys.argv
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(FIG10_SCRIPT),
                "--results-dir",
                str(results_dir),
                "--published",
                str(published_csv),
                "--selected-galaxies",
                str(selected_gals),
                "--surviving",
                str(surviving_json),
                "--out",
                str(tmp_path / "fig10.pdf"),
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
    finally:
        sys.argv = old_argv

    assert result.returncode != 0, "No adopted cells should exit non-zero"
    assert not (tmp_path / "fig10.pdf").exists()


def test_missing_surviving_mass_json_exits_with_command(tmp_path: Path) -> None:
    """Missing surviving mass JSON must exit non-zero with command in message."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create an adopted cell with survived mass
    gal_id = 21
    xlike_key = "beagle_like"
    write_xlike_cell(results_dir, gal_id, xlike_key)

    # Create CSV
    published_csv = tmp_path / "art_sedfitting_z1.csv"
    published_csv.write_text(
        "code,id,logmstar,logmstar_lo,logmstar_hi,logsfr,logsfr_lo,logsfr_hi\n"
        f"BEAGLE,{gal_id},10.0,9.8,10.2,0.5,0.3,0.7\n"
    )

    # Create selected galaxies
    selected_gals = tmp_path / "selected_galaxies_20.json"
    selected_gals.write_text(json.dumps({"selected_galaxies": [{"id": gal_id}]}))

    # Don't create surviving JSON - this should trigger error
    surviving_json = tmp_path / "surviving.json"

    result = subprocess.run(
        [
            sys.executable,
            str(FIG10_SCRIPT),
            "--results-dir",
            str(results_dir),
            "--published",
            str(published_csv),
            "--selected-galaxies",
            str(selected_gals),
            "--surviving",
            str(surviving_json),
            "--out",
            str(tmp_path / "fig10.pdf"),
            "--data-out",
            str(tmp_path / "fig10_data.json"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    assert result.returncode != 0, "Missing surviving JSON should exit non-zero"
    assert CENSUS_COMMAND in result.stderr, "Error message should name the generating command"
    assert "--suite xlike" in result.stderr
    assert f"{gal_id}_{xlike_key}" in result.stderr


def test_all_codes_use_sfr_100myr(tmp_path: Path) -> None:
    """All codes must use sfr_100myr, not sfr_10myr."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create adopted cells for each code with distinguishable SFR values
    published_csv = tmp_path / "art_sedfitting_z1.csv"
    csv_lines = ["code,id,logmstar,logmstar_lo,logmstar_hi,logsfr,logsfr_lo,logsfr_hi"]

    selected_ids = []
    for xlike_key, code_name in XLIKE_CODE.items():
        gal_id = 20 + len(selected_ids)
        selected_ids.append(gal_id)

        # Distinguishable 100 Myr vs 10 Myr SFR [dex], so a wrong key shows
        sfr_100myr_val = 1.0 + gal_id * 0.1
        write_xlike_cell(
            results_dir,
            gal_id,
            xlike_key,
            log_mass=10.0 + gal_id * 0.01,
            log_sfr=sfr_100myr_val,
            log_sfr_10myr=sfr_100myr_val - 0.5,
        )

        # Published value uses sfr_100myr (bounds are lower and upper, not errors)
        sfr_lo = sfr_100myr_val - 0.2
        sfr_hi = sfr_100myr_val + 0.2
        csv_lines.append(
            f"{code_name},{gal_id},10.0,9.8,10.2,{sfr_100myr_val:.2f},{sfr_lo:.2f},{sfr_hi:.2f}"
        )

    published_csv.write_text("\n".join(csv_lines))

    # Create selected galaxies
    selected_gals = tmp_path / "selected_galaxies_20.json"
    selected_gals.write_text(
        json.dumps({"selected_galaxies": [{"id": gal_id} for gal_id in selected_ids]})
    )

    # Create surviving JSON for non-Prospector codes
    surviving_cells = {}
    for i, (xlike_key, code_name) in enumerate(XLIKE_CODE.items()):
        if code_name != "Prospector":
            gal_id = selected_ids[i]
            surviving_cells[f"{gal_id}_{xlike_key}"] = {
                "galaxy": gal_id,
                "config": xlike_key,
                "log_mass_survived_p50": 10.0,
                "log_mass_survived_p16": 9.9,
                "log_mass_survived_p84": 10.1,
            }

    surviving_json = tmp_path / "surviving.json"
    surviving_json.write_text(json.dumps({"cells": surviving_cells}))

    result = subprocess.run(
        [
            sys.executable,
            str(FIG10_SCRIPT),
            "--results-dir",
            str(results_dir),
            "--published",
            str(published_csv),
            "--selected-galaxies",
            str(selected_gals),
            "--surviving",
            str(surviving_json),
            "--out",
            str(tmp_path / "fig10.pdf"),
            "--data-out",
            str(tmp_path / "fig10_data.json"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert (tmp_path / "fig10_data.json").exists()

    # Verify the sidecar shows sfr_100myr was used
    sidecar = json.loads((tmp_path / "fig10_data.json").read_text())
    for code in XLIKE_CODE.values():
        if code in sidecar["codes"] and sidecar["codes"][code]["n_adopted"] > 0:
            # If offset is very close to zero, it means we used the right SFR
            sfr_offset = sidecar["codes"][code]["sfr_median_offset"]
            assert sfr_offset is not None
            # The offset should be close to zero because published and tengri both use sfr_100myr
            assert abs(sfr_offset) < 0.5, (
                f"{code} SFR offset too large, may be using wrong timescale"
            )


def test_prospector_uses_formed_mass_others_use_surviving(tmp_path: Path) -> None:
    """Prospector must use formed mass; other codes must use surviving mass."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create cells for Prospector and one other code
    gal_id = 21

    # Prospector reports FORMED mass: the NPZ's stellar_mass. BEAGLE reports surviving
    # mass, which no NPZ carries, so it arrives through the census only.
    write_xlike_cell(results_dir, gal_id, "prospector_like", log_mass=10.5)
    write_xlike_cell(results_dir, gal_id, "beagle_like", log_mass=10.5)

    # CSV: Prospector published formed=10.5, BEAGLE published survived=10.2
    published_csv = tmp_path / "art_sedfitting_z1.csv"
    published_csv.write_text(
        "code,id,logmstar,logmstar_lo,logmstar_hi,logsfr,logsfr_lo,logsfr_hi\n"
        f"Prospector,{gal_id},10.5,10.3,10.7,0.5,0.3,0.7\n"
        f"BEAGLE,{gal_id},10.2,10.0,10.4,0.5,0.3,0.7\n"
    )

    selected_gals = tmp_path / "selected_galaxies_20.json"
    selected_gals.write_text(json.dumps({"selected_galaxies": [{"id": gal_id}]}))

    surviving_json = tmp_path / "surviving.json"
    surviving_json.write_text(
        json.dumps({"cells": surviving_census(gal_id, ["beagle_like"], 10.2, half_width=0.2)})
    )

    data_out_path = tmp_path / "fig10_data.json"
    result = subprocess.run(
        [
            sys.executable,
            str(FIG10_SCRIPT),
            "--results-dir",
            str(results_dir),
            "--published",
            str(published_csv),
            "--selected-galaxies",
            str(selected_gals),
            "--surviving",
            str(surviving_json),
            "--out",
            str(tmp_path / "fig10.pdf"),
            "--data-out",
            str(data_out_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    assert result.returncode == 0

    # Verify sidecar shows expected offsets
    sidecar = json.loads(data_out_path.read_text())

    # Prospector should have offset ~0 (formed 10.5 vs published 10.5)
    pro_offset = sidecar["codes"]["Prospector"]["mass_median_offset"]
    assert pro_offset is not None
    assert abs(pro_offset) < 0.1, "Prospector offset should be ~0, indicating formed mass was used"

    # BEAGLE should have offset ~0 (survived 10.2 vs published 10.2)
    bea_offset = sidecar["codes"]["BEAGLE"]["mass_median_offset"]
    assert bea_offset is not None
    assert abs(bea_offset) < 0.1, "BEAGLE offset should be ~0, indicating survived mass was used"


def test_code_markers_present(tmp_path: Path) -> None:
    """Every code drawn must have a marker in CODE_MARKERS."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    published_csv = tmp_path / "art_sedfitting_z1.csv"
    csv_lines = ["code,id,logmstar,logmstar_lo,logmstar_hi,logsfr,logsfr_lo,logsfr_hi"]

    for i, (xlike_key, code_name) in enumerate(XLIKE_CODE.items()):
        gal_id = 21 + i

        write_xlike_cell(results_dir, gal_id, xlike_key)

        csv_lines.append(f"{code_name},{gal_id},10.0,9.8,10.2,0.5,0.3,0.7")

    published_csv.write_text("\n".join(csv_lines))

    selected_gals = tmp_path / "selected_galaxies_20.json"
    selected_gals.write_text(
        json.dumps({"selected_galaxies": [{"id": 21 + i} for i in range(len(XLIKE_CODE))]})
    )

    surviving_cells = {}
    for i, (xlike_key, code_name) in enumerate(XLIKE_CODE.items()):
        if code_name != "Prospector":
            surviving_cells.update(surviving_census(21 + i, [xlike_key], 10.0))

    surviving_json = tmp_path / "surviving.json"
    surviving_json.write_text(json.dumps({"cells": surviving_cells}))

    result = subprocess.run(
        [
            sys.executable,
            str(FIG10_SCRIPT),
            "--results-dir",
            str(results_dir),
            "--published",
            str(published_csv),
            "--selected-galaxies",
            str(selected_gals),
            "--surviving",
            str(surviving_json),
            "--out",
            str(tmp_path / "fig10.pdf"),
            "--data-out",
            str(tmp_path / "fig10_data.json"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    assert result.returncode == 0

    # Every code in XLIKE_CODE should have a marker
    for code_name in XLIKE_CODE.values():
        assert code_name in CODE_MARKERS, f"{code_name} missing from CODE_MARKERS"


def test_non_adopted_cells_counted_not_drawn(tmp_path: Path) -> None:
    """Non-adopted cells must be counted but not drawn."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create one adopted and one non-adopted cell for the same code/galaxy
    gal_id = 21
    adopted_key = "cigale_like"
    non_adopted_key = "prospector_like"

    write_xlike_cell(results_dir, gal_id, adopted_key, log_mass=10.0, log_sfr=0.5)
    write_xlike_cell(
        results_dir, gal_id, non_adopted_key, adopted=False, log_mass=10.2, log_sfr=0.6
    )

    # Published data
    published_csv = tmp_path / "art_sedfitting_z1.csv"
    published_csv.write_text(
        "code,id,logmstar,logmstar_lo,logmstar_hi,logsfr,logsfr_lo,logsfr_hi\n"
        f"CIGALE,{gal_id},10.0,9.8,10.2,0.5,0.3,0.7\n"
        f"Prospector,{gal_id},10.2,10.0,10.4,0.6,0.4,0.8\n"
    )

    selected_gals = tmp_path / "selected_galaxies_20.json"
    selected_gals.write_text(json.dumps({"selected_galaxies": [{"id": gal_id}]}))

    surviving_json = tmp_path / "surviving.json"
    surviving_json.write_text(json.dumps({"cells": surviving_census(gal_id, [adopted_key], 10.0)}))

    result = subprocess.run(
        [
            sys.executable,
            str(FIG10_SCRIPT),
            "--results-dir",
            str(results_dir),
            "--published",
            str(published_csv),
            "--selected-galaxies",
            str(selected_gals),
            "--surviving",
            str(surviving_json),
            "--out",
            str(tmp_path / "fig10.pdf"),
            "--data-out",
            str(tmp_path / "fig10_data.json"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"

    # Verify sidecar counts both adopted and non-adopted
    sidecar = json.loads((tmp_path / "fig10_data.json").read_text())
    assert sidecar["codes"]["CIGALE"]["n_adopted"] == 1
    assert sidecar["codes"]["Prospector"]["n_non_adopted"] == 1


def test_code_markers_ast_parity() -> None:
    """CODE_MARKERS in fig10 must match fig06_code_overlay.py via AST parse."""
    fig06_path = ANALYSIS_DIR / "fig06_code_overlay.py"
    with open(fig06_path) as f:
        tree = ast.parse(f.read())

    # Find CODE_MARKERS assignment in fig06
    fig06_markers = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CODE_MARKERS":
                    # Extract dict value
                    if isinstance(node.value, ast.Dict):
                        fig06_markers = {}
                        for k, v in zip(node.value.keys, node.value.values):
                            if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                                fig06_markers[k.value] = v.value
                    break

    assert fig06_markers is not None, "Could not parse CODE_MARKERS from fig06_code_overlay.py"
    assert fig06_markers == CODE_MARKERS, (
        f"CODE_MARKERS parity check failed: fig10 has {CODE_MARKERS}, fig06 has {fig06_markers}"
    )


def test_no_jax_import_in_dependencies() -> None:
    """Verify _adoption.py and _figure_style.py don't import jax/tengri."""
    for module_path in [ANALYSIS_DIR / "_adoption.py", ANALYSIS_DIR / "_figure_style.py"]:
        with open(module_path) as f:
            content = f.read()
        assert "import jax" not in content, f"{module_path.name} imports jax"
        assert "import tengri" not in content, f"{module_path.name} imports tengri"
        assert "from jax" not in content, f"{module_path.name} imports from jax"
        assert "from tengri" not in content, f"{module_path.name} imports from tengri"


def test_sidecar_records_the_adoption_bar_per_code():
    """A caption drawing a relaxed code must say so; the sidecar is where it reads that.

    Prospector-like is a continuity-SFH configuration and so on the relaxed bar;
    every other X-like code is judged strictly.
    """
    from fig10_xlike_one_to_one import ADOPTION_BAR

    assert ADOPTION_BAR == {
        "CIGALE": "strict",
        "Prospector": "relaxed",
        "BAGPIPES": "strict",
        "BEAGLE": "strict",
        "Dense_Basis": "strict",
    }
