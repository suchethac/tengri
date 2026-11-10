# SPDX-License-Identifier: BSD-3-Clause
"""The combiner reads one evidence route per weight set.

``SET_ROUTE`` is the single source: named_grid and named_all take nested-sampling
cells from the NSS directory, the factorial takes Laplace cells from the Laplace
directory. A cell of the wrong route is excluded with a reason, never used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PAPER1 = Path(__file__).resolve().parents[1]
ANALYSIS = PAPER1.parent
for entry in (str(ANALYSIS), str(PAPER1)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from paper1 import _bma_keys as bk
from paper1.bma_combine import (
    NSS_MAX_LOG_EVIDENCE_ERR,
    SET_ROUTE,
    _validate_cell,
    combine_bma,
    weight_std_mc,
)

pytestmark = pytest.mark.unit

GAL = 11
FACT = bk.model_key(
    {
        "sfh": "dpl",
        "ssp": bk.SSP_LABELS[0],
        "attenuation": "calzetti",
        "dust_emission": bk.DUST_EMISSION,
        "nebular": bk.NEBULAR,
    }
)


def _cell(key: str, route: str, log_z: float, err: float | None = 0.1) -> dict:
    cell = {
        "galaxy": GAL,
        "model_key": key,
        "weight_sets": sorted(bk.set_membership(key)),
        "components": bk.parse_model_key(key),
        "route": route,
        "log_evidence": log_z,
        "code_revision": "test",
    }
    if route == "laplace":
        cell.update(newton_decrement=0.01, n_clipped_eigenvalues=0, valid=True)
    else:
        cell.update(
            log_evidence_err=err,
            newton_decrement=None,
            n_clipped_eigenvalues=None,
            condition_number=None,
            valid=True,
        )
    return cell


def _put(root: Path, cell: dict) -> None:
    gal_dir = root / str(GAL)
    gal_dir.mkdir(parents=True, exist_ok=True)
    (gal_dir / f"{cell['model_key']}.json").write_text(json.dumps(cell))
    rng = np.random.default_rng(0)
    np.savez(
        gal_dir / f"{cell['model_key']}.npz",
        log_stellar_mass_formed=rng.normal(10, 0.1, 40),
        log_stellar_mass_survived=rng.normal(10, 0.1, 40),
        log_sfr_100myr=rng.normal(0, 0.1, 40),
        log_sfr_10myr=rng.normal(0, 0.1, 40),
    )


def _combine(tmp_path: Path) -> dict:
    return combine_bma(
        tmp_path / "lap",
        tmp_path / "fits",
        tmp_path / "fits_x",
        50,
        0,
        nss_evidence_dir=tmp_path / "nss",
    )


def _sets(summary: dict) -> dict:
    return summary["galaxies"][str(GAL)]["sets"]


def _models(w_set: dict) -> dict[str, dict]:
    return {m["model_key"]: m for m in w_set["models"]}


def test_set_route_mapping_is_the_documented_one():
    assert SET_ROUTE == {"named_grid": "nss", "named_all": "nss", "factorial": "laplace"}


def test_each_set_reads_only_its_routes_directory(tmp_path):
    lap, nss = tmp_path / "lap", tmp_path / "nss"
    # Named: the NSS cells say I is far better than II; the Laplace cells say the opposite.
    _put(nss, _cell("config-I", "nss", 10.0))
    _put(nss, _cell("config-II", "nss", 0.0))
    _put(lap, _cell("config-I", "laplace", 0.0))
    _put(lap, _cell("config-II", "laplace", 10.0))
    # Factorial: a Laplace cell is used; the NSS directory's factorial cell is not.
    _put(lap, _cell(FACT, "laplace", 3.0))
    _put(nss, _cell(FACT, "nss", 3.0))

    sets = _sets(_combine(tmp_path))
    grid = _models(sets["named_grid"])
    assert set(grid) == {"config-I", "config-II"}
    assert grid["config-I"]["weight"] > 0.99 and grid["config-I"]["route"] == "nss"
    assert set(_models(sets["named_all"])) == {"config-I", "config-II"}
    fact = _models(sets["factorial"])
    assert set(fact) == {FACT} and fact[FACT]["route"] == "laplace"
    assert sets["named_grid"]["route"] == "nss" and sets["factorial"]["route"] == "laplace"


def test_laplace_cell_in_the_nss_directory_is_excluded_with_a_reason(tmp_path):
    _put(tmp_path / "lap", _cell(FACT, "laplace", 1.0))  # keeps the Laplace dir non-empty
    _put(tmp_path / "nss", _cell("config-I", "nss", 1.0))
    _put(tmp_path / "nss", _cell("config-II", "laplace", 50.0))

    grid = _models(_sets(_combine(tmp_path))["named_grid"])
    assert grid["config-II"]["valid"] is False and grid["config-II"]["weight"] is None
    assert "route=laplace" in grid["config-II"]["excluded_reason"]
    assert grid["config-I"]["weight"] == pytest.approx(1.0)


def test_nss_cell_in_the_laplace_directory_is_excluded_with_a_reason(tmp_path):
    other = bk.model_key({**bk.parse_model_key(FACT), "sfh": "lnorm"})
    _put(tmp_path / "lap", _cell(FACT, "laplace", 1.0))
    _put(tmp_path / "lap", _cell(other, "nss", 50.0))

    fact = _models(_sets(_combine(tmp_path))["factorial"])
    assert fact[other]["valid"] is False and "route=nss" in fact[other]["excluded_reason"]
    assert fact[FACT]["weight"] == pytest.approx(1.0)


def test_missing_nss_directory_leaves_the_named_sets_empty_not_laplace_filled(tmp_path):
    _put(tmp_path / "lap", _cell("config-I", "laplace", 1.0))
    _put(tmp_path / "lap", _cell(FACT, "laplace", 1.0))

    sets = _sets(_combine(tmp_path))
    assert sets["named_grid"]["n_valid"] == 0 and sets["named_grid"]["models"] == []
    assert "nss" in sets["named_grid"]["reason"]
    assert sets["factorial"]["n_valid"] == 1


@pytest.mark.parametrize(
    ("err", "valid"),
    [
        (0.0, True),
        (NSS_MAX_LOG_EVIDENCE_ERR, True),
        (NSS_MAX_LOG_EVIDENCE_ERR + 1e-6, False),
        (float("nan"), False),
        (float("inf"), False),
        (None, False),
    ],
)
def test_nss_validity_rule_on_the_error(err, valid):
    ok, reason = _validate_cell(_cell("config-I", "nss", 1.0, err=err), "nss")
    assert ok is valid
    assert (reason is None) is valid
    if not valid:
        assert "log_evidence_err" in reason


def test_nss_validity_needs_a_finite_evidence():
    ok, reason = _validate_cell(_cell("config-I", "nss", float("nan")), "nss")
    assert ok is False and "log_evidence" in reason


def test_nss_cell_over_the_threshold_is_excluded_from_the_weights(tmp_path):
    _put(tmp_path / "lap", _cell(FACT, "laplace", 1.0))
    _put(tmp_path / "nss", _cell("config-I", "nss", 1.0, err=0.1))
    _put(tmp_path / "nss", _cell("config-II", "nss", 9.0, err=NSS_MAX_LOG_EVIDENCE_ERR + 0.2))

    summary = _combine(tmp_path)
    grid = _models(_sets(summary)["named_grid"])
    assert grid["config-II"]["valid"] is False and grid["config-I"]["weight"] == pytest.approx(1.0)
    assert summary["invalid_counts"]["named_grid"] == {"config-II": 1}


def test_weight_std_is_zero_without_evidence_error():
    std = weight_std_mc({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 0.0})
    assert std == pytest.approx({"a": 0.0, "b": 0.0}, abs=1e-12)


def test_weight_std_grows_with_the_error_and_is_reproducible():
    log_z = {"a": 0.0, "b": 0.0}
    small = weight_std_mc(log_z, {"a": 0.1, "b": 0.1})
    large = weight_std_mc(log_z, {"a": 1.0, "b": 1.0})
    assert 0.0 < small["a"] < large["a"] <= 0.5
    assert small["a"] == pytest.approx(small["b"], rel=0.2)  # symmetric pair
    assert weight_std_mc(log_z, {"a": 1.0, "b": 1.0}) == large


def test_dominant_model_has_a_tiny_weight_std():
    std = weight_std_mc({"a": 20.0, "b": 0.0}, {"a": 0.2, "b": 0.2})
    assert std["a"] < 1e-6 and std["b"] < 1e-6


def test_summary_carries_weight_std_for_nss_sets_only(tmp_path):
    _put(tmp_path / "nss", _cell("config-I", "nss", 0.0, err=0.3))
    _put(tmp_path / "nss", _cell("config-II", "nss", 0.0, err=0.3))
    other = bk.model_key({**bk.parse_model_key(FACT), "sfh": "lnorm"})
    _put(tmp_path / "lap", _cell(FACT, "laplace", 0.0))
    _put(tmp_path / "lap", _cell(other, "laplace", 0.0))

    sets = _sets(_combine(tmp_path))
    for model in sets["named_grid"]["models"]:
        assert 0.0 < model["weight_std"] < 0.5
        assert model["log_evidence_err"] == pytest.approx(0.3)
    assert all(m["weight_std"] is None for m in sets["factorial"]["models"])


def test_summary_records_routes_rules_and_both_directories_provenance(tmp_path):
    lap_cell = _cell(FACT, "laplace", 1.0)
    lap_cell["code_revision"] = "aaa"
    nss_cell = _cell("config-I", "nss", 1.0)
    nss_cell["code_revision"] = "bbb"
    _put(tmp_path / "lap", lap_cell)
    _put(tmp_path / "nss", nss_cell)

    summary = _combine(tmp_path)
    assert summary["route_by_set"] == SET_ROUTE
    assert str(NSS_MAX_LOG_EVIDENCE_ERR) in summary["validity_rules"]["nss"]
    assert "Newton" in summary["validity_rules"]["laplace"]
    assert "nss" in summary["rules"] and "Laplace approximation" in summary["rules"]
    assert summary["evidence_code_revisions"] == {"aaa": 1, "bbb": 1}
