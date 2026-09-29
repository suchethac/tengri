# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end: cells written by the runner's own code path land in the right weight sets.

The combiner's unit tests used fixtures in the combiner's own key format, so they
passed while none of its keys matched what the runner wrote. Here every cell is
produced by ``bma_evidence.build_cell_result`` + ``write_cell`` (the runner's
result assembly and atomic writers, fed a fake Laplace result) and read back by
``combine_bma``. The expected weight sets are written out in this file, not
derived from ``_bma_keys``, so a mistake in the shared module still fails here.

Pinned invariants:
- every one of the 110 models lands in exactly the weight sets it belongs to;
- an invalid cell is counted invalid in each of its sets and never weighted;
- a cell whose ``weight_sets`` field or file name disagrees with its key is
  excluded with a reason; a key outside the space is reported, not dropped.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PAPER1 = Path(__file__).resolve().parents[1]
ANALYSIS = PAPER1.parent
for entry in (str(ANALYSIS), str(PAPER1)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from paper1 import _bma_keys as bk
from paper1.bma_combine import combine_bma
from paper1.bma_evidence import build_cell_result, write_cell

pytestmark = pytest.mark.unit

GALAXY = 4242
GRID_SET = {"named_grid", "named_all"}


def _laplace(log_z: float, newton: float | None = 0.01, clipped: int | None = 0):
    diagnostics = {"condition_number": 1e4}
    if newton is not None:
        diagnostics["newton_decrement"] = newton
    if clipped is not None:
        diagnostics["n_clipped_eigenvalues"] = clipped
    return SimpleNamespace(log_evidence=log_z, diagnostics=diagnostics)


def _npz(seed: int, n: int = 30) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "log_stellar_mass_formed": rng.normal(10.5, 0.1, n),
        "log_stellar_mass_survived": rng.normal(10.3, 0.1, n),
        "log_sfr_100myr": rng.normal(0.5, 0.1, n),
        "log_sfr_10myr": rng.normal(0.4, 0.1, n),
    }


def _write_runner_cell(out_dir: Path, model: dict, laplace, seed: int = 0) -> dict:
    result = build_cell_result(
        galaxy_id=GALAXY,
        z=1.0,
        model_dict=model,
        laplace=laplace,
        map_loss=10.0,
        n_free=8,
        n_map_restarts=2,
        started=time.time(),
        stage_times={},
        seed=seed,
    )
    write_cell(out_dir, result, _npz(seed))
    return result


def _combine(tmp_path: Path) -> dict:
    return combine_bma(tmp_path / "ev", tmp_path / "fits", tmp_path / "fits_x", 40, 0)


def _models_in(set_summary: dict) -> dict[str, dict]:
    return {m["model_key"]: m for m in set_summary["models"]}


def test_named_xlike_and_factorial_cells_land_in_their_sets(tmp_path: Path) -> None:
    ev = tmp_path / "ev" / str(GALAXY)
    ev.mkdir(parents=True)
    grid_i = {"config": "I"}
    xlike = {"config": "cigale_like"}
    fact_a = {
        "sfh": "dpl",
        "ssp": "fsps_prsc_miles_chabrier",
        "attenuation": "smc",
        "dust_emission": "dl14",
        "nebular": "cue",
    }
    fact_b = dict(fact_a, sfh="lnorm")
    written = {
        "config-I": _write_runner_cell(ev, grid_i, _laplace(5.0), 1),
        "xlike-cigale_like": _write_runner_cell(ev, xlike, _laplace(4.0), 2),
        "fa": _write_runner_cell(ev, fact_a, _laplace(3.0), 3),
        "fb": _write_runner_cell(ev, fact_b, _laplace(3.0), 4),
    }
    assert written["config-I"]["model_key"] == "config-I"
    assert written["xlike-cigale_like"]["model_key"] == "xlike-cigale_like"
    assert written["config-I"]["weight_sets"] == ["named_all", "named_grid"]
    assert written["xlike-cigale_like"]["weight_sets"] == ["named_all"]
    assert written["fa"]["weight_sets"] == ["factorial"]

    sets = _combine(tmp_path)["galaxies"][str(GALAXY)]["sets"]

    grid = _models_in(sets["named_grid"])
    assert set(grid) == {"config-I"}
    assert grid["config-I"]["valid"] and grid["config-I"]["weight"] == pytest.approx(1.0)

    named_all = _models_in(sets["named_all"])
    assert set(named_all) == {"config-I", "xlike-cigale_like"}
    assert sum(m["weight"] for m in named_all.values()) == pytest.approx(1.0)
    assert named_all["config-I"]["weight"] > named_all["xlike-cigale_like"]["weight"] > 0.0

    factorial = _models_in(sets["factorial"])
    assert set(factorial) == {written["fa"]["model_key"], written["fb"]["model_key"]}
    assert all(m["weight"] == pytest.approx(0.5) for m in factorial.values())
    assert sets["factorial"]["marginal"]["isochrone"] == pytest.approx({"prsc": 1.0})
    assert sets["factorial"]["marginal"]["spectral_library"] == pytest.approx({"miles": 1.0})
    assert sets["factorial"]["marginal"]["sfh"] == pytest.approx({"dpl": 0.5, "lnorm": 0.5})


def test_every_model_in_the_space_lands_in_exactly_its_sets(tmp_path: Path) -> None:
    ev = tmp_path / "ev" / str(GALAXY)
    ev.mkdir(parents=True)
    models = bk.enumerate_factorial() + bk.enumerate_named_all()
    assert len(models) == 110
    for i, model in enumerate(models):
        _write_runner_cell(ev, model, _laplace(float(i % 7)), seed=i)

    summary = _combine(tmp_path)
    sets = summary["galaxies"][str(GALAXY)]["sets"]

    assert summary["galaxies"][str(GALAXY)]["unrecognized_cells"] == []
    assert {name: s["n_valid"] for name, s in sets.items()} == {
        "named_grid": 5,
        "named_all": 10,
        "factorial": 100,
    }
    assert {name: s["n_expected"] for name, s in sets.items()} == {
        "named_grid": 5,
        "named_all": 10,
        "factorial": 100,
    }
    assert {k for k in _models_in(sets["named_grid"])} == {
        f"config-{c}" for c in ["I", "II", "III", "IV", "V"]
    }
    assert {k for k in _models_in(sets["named_all"]) if k.startswith("xlike-")} == {
        f"xlike-{k}"
        for k in (
            "cigale_like",
            "prospector_like",
            "bagpipes_like",
            "beagle_like",
            "dense_basis_like",
        )
    }
    assert not any(k.startswith(("config-", "xlike-")) for k in _models_in(sets["factorial"]))
    for name, s in sets.items():
        assert sum(m["weight"] for m in s["models"]) == pytest.approx(1.0), name
    assert sum(summary["galaxies_per_model"]["factorial"].values()) == 100


def test_invalid_cell_is_counted_in_each_of_its_sets(tmp_path: Path) -> None:
    ev = tmp_path / "ev" / str(GALAXY)
    ev.mkdir(parents=True)
    _write_runner_cell(ev, {"config": "II"}, _laplace(2.0), 1)
    bad = _write_runner_cell(ev, {"config": "III"}, _laplace(9.0, newton=0.5), 2)
    assert bad["valid"] is False and bad["log_evidence"] == 9.0

    summary = _combine(tmp_path)
    assert summary["invalid_counts"]["named_grid"] == {"config-III": 1}
    assert summary["invalid_counts"]["named_all"] == {"config-III": 1}
    assert summary["galaxies_per_model"]["named_grid"] == {"config-II": 1}
    grid = _models_in(summary["galaxies"][str(GALAXY)]["sets"]["named_grid"])
    assert grid["config-III"]["weight"] is None
    assert "newton_decrement" in grid["config-III"]["excluded_reason"]
    assert grid["config-II"]["weight"] == pytest.approx(1.0)


def test_missing_diagnostic_makes_the_cell_invalid(tmp_path: Path) -> None:
    ev = tmp_path / "ev" / str(GALAXY)
    ev.mkdir(parents=True)
    result = _write_runner_cell(ev, {"config": "I"}, _laplace(1.0, newton=None))
    assert result["valid"] is False and result["newton_decrement"] is None


def test_weight_sets_field_disagreement_excludes_the_cell(tmp_path: Path) -> None:
    ev = tmp_path / "ev" / str(GALAXY)
    ev.mkdir(parents=True)
    _write_runner_cell(ev, {"config": "I"}, _laplace(2.0), 1)
    _write_runner_cell(ev, {"config": "IV"}, _laplace(2.0), 2)
    path = ev / "config-IV.json"
    cell = json.loads(path.read_text())
    cell["weight_sets"] = ["named_all"]
    path.write_text(json.dumps(cell))

    grid = _models_in(_combine(tmp_path)["galaxies"][str(GALAXY)]["sets"]["named_grid"])
    assert grid["config-IV"]["valid"] is False
    assert "weight_sets" in grid["config-IV"]["excluded_reason"]
    assert grid["config-I"]["weight"] == pytest.approx(1.0)


def test_file_name_disagreement_excludes_the_cell(tmp_path: Path) -> None:
    ev = tmp_path / "ev" / str(GALAXY)
    ev.mkdir(parents=True)
    _write_runner_cell(ev, {"config": "I"}, _laplace(2.0), 1)
    _write_runner_cell(ev, {"config": "V"}, _laplace(2.0), 2)
    (ev / "config-V.json").rename(ev / "renamed.json")

    grid = _models_in(_combine(tmp_path)["galaxies"][str(GALAXY)]["sets"]["named_grid"])
    assert grid["config-V"]["valid"] is False
    assert "file name" in grid["config-V"]["excluded_reason"]


def test_unknown_key_is_reported_not_silently_averaged(tmp_path: Path) -> None:
    ev = tmp_path / "ev" / str(GALAXY)
    ev.mkdir(parents=True)
    _write_runner_cell(ev, {"config": "I"}, _laplace(2.0), 1)
    stale = json.loads((ev / "config-I.json").read_text())
    stale["model_key"] = "X-like-cigale_like"
    (ev / "X-like-cigale_like.json").write_text(json.dumps(stale))

    gal = _combine(tmp_path)["galaxies"][str(GALAXY)]
    assert [u["model_key"] for u in gal["unrecognized_cells"]] == ["X-like-cigale_like"]
    assert set(_models_in(gal["sets"]["named_all"])) == {"config-I"}
