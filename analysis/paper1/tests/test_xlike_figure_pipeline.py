# SPDX-License-Identifier: BSD-3-Clause
"""Paper-1 X-like figure pipeline: fig10 inputs, the notebook runner, fig11 skips, the census.

The fig10 fixtures are written with the keys the real fit writer stores
(``fit_one.DERIVED_KEYS`` through ``build_npz_payload``), so renaming a key in the
writer breaks these tests rather than silently emptying the figure.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
for _path in (ANALYSIS_DIR, ANALYSIS_DIR.parents[1]):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from analysis.paper1.tests._xlike_cells import surviving_census, write_xlike_cell

GAL = 101
LOG_FORMED = 10.0  # log10(M_formed / Msun) of the fixture draws
LOG_SURVIVED = 9.8  # census survived median [dex]; below formed by construction
LOG_SFR = 0.5  # log10(SFR_100Myr / (Msun/yr))


@pytest.fixture(scope="module")
def fig10():
    import analysis.paper1.fig10_xlike_one_to_one as module

    return module


@pytest.fixture(scope="module")
def fit_one():
    import analysis.paper1.fit_one as module

    return module


def _write_cell(results_dir, fig10, fit_one, xlike_key, *, adopted=True):
    write_xlike_cell(
        results_dir,
        GAL,
        xlike_key,
        adopted=adopted,
        log_mass=LOG_FORMED,
        log_sfr=LOG_SFR,
        scatter=0.05,
        n_draws=200,
    )


def _census(keys) -> dict:
    return surviving_census(GAL, keys, LOG_SURVIVED)


def test_fig10_reads_the_keys_the_fit_writer_stores(fig10, fit_one):
    """fig10's NPZ keys are fields ``fit_one`` really writes, as linear quantities."""
    assert fig10.NPZ_MASS_KEY in fit_one.DERIVED_KEYS
    assert fig10.NPZ_SFR_KEY in fit_one.DERIVED_KEYS


def test_fig10_places_every_code_with_its_own_mass_definition(fig10, fit_one, tmp_path):
    """Formed mass for Prospector, census survived mass for the other four codes."""
    for key in fig10.XLIKE_CODE:
        _write_cell(tmp_path, fig10, fit_one, key)
    surviving_keys = [k for k in fig10.XLIKE_CODE if k != "prospector_like"]

    values = fig10._load_xlike_fits(tmp_path, {GAL}, _census(surviving_keys))

    by_key = {v.xlike_key: v for v in values}
    assert set(by_key) == set(fig10.XLIKE_CODE)
    assert all(v.adopted for v in values)
    assert by_key["prospector_like"].log_mstar_p50 == pytest.approx(LOG_FORMED, abs=0.02)
    for key in surviving_keys:
        assert by_key[key].log_mstar_p50 == pytest.approx(LOG_SURVIVED)
        assert by_key[key].log_mstar_p16 == pytest.approx(LOG_SURVIVED - 0.1)
    for v in values:
        assert v.log_sfr_p50 == pytest.approx(LOG_SFR, abs=0.02)
        assert v.log_sfr_p16 < v.log_sfr_p50 < v.log_sfr_p84


def test_fig10_adopted_cell_missing_from_census_fails_naming_it(fig10, fit_one, tmp_path, capsys):
    for key in fig10.XLIKE_CODE:
        _write_cell(tmp_path, fig10, fit_one, key)
    census = _census(["cigale_like", "beagle_like", "dense_basis_like"])  # no bagpipes

    with pytest.raises(SystemExit) as exc:
        fig10._load_xlike_fits(tmp_path, {GAL}, census)

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert f"{GAL}_bagpipes_like" in err
    assert "--suite xlike" in err


def test_fig10_formed_mass_cell_needs_no_census(fig10, fit_one, tmp_path):
    _write_cell(tmp_path, fig10, fit_one, "prospector_like")

    values = fig10._load_xlike_fits(tmp_path, {GAL}, {})

    assert [v.xlike_key for v in values] == ["prospector_like"]


def test_fig10_no_adopted_cell_exits(fig10, fit_one, tmp_path, capsys):
    _write_cell(tmp_path, fig10, fit_one, "prospector_like", adopted=False)

    with pytest.raises(SystemExit) as exc:
        fig10._load_xlike_fits(tmp_path, {GAL}, {})

    assert exc.value.code == 1
    assert "no adopted X-like cells" in capsys.readouterr().err


def test_fig10_default_census_path_is_the_census_default_out(fig10):
    from analysis.paper1.surviving_mass_census import DEFAULT_OUT_BY_SUITE

    assert DEFAULT_OUT_BY_SUITE["xlike"] == fig10.DEFAULT_SURVIVING
    assert "--suite xlike" in fig10.CENSUS_COMMAND
    assert str(DEFAULT_OUT_BY_SUITE["xlike"].name) in fig10.CENSUS_COMMAND


# --- run_figure ---------------------------------------------------------------


@pytest.fixture
def fake_figure(monkeypatch):
    """Install ``analysis.paper1.<name>`` whose ``main`` exits with a chosen code."""
    monkeypatch.syspath_prepend(str(ANALYSIS_DIR.parents[1] / "paper1_figures"))

    def install(code):
        module = types.ModuleType("analysis.paper1._fake_exit")

        def main():
            raise SystemExit(code)

        module.main = main
        monkeypatch.setitem(sys.modules, "analysis.paper1._fake_exit", module)

    return install


@pytest.mark.parametrize(("code", "status"), [(None, 0), (0, 0), (2, 2)])
def test_run_figure_returns_the_exit_code_of_a_script_that_exits(fake_figure, code, status):
    from _run import run_figure

    fake_figure(code)

    assert run_figure("_fake_exit", []) == status


def test_run_figure_prints_a_string_exit_and_returns_failure(fake_figure, capsys):
    from _run import run_figure

    fake_figure("ERROR: no inputs")

    assert run_figure("_fake_exit", []) == 1
    assert "ERROR: no inputs" in capsys.readouterr().err


# --- fig11 --------------------------------------------------------------------


def _model(key: str, weight: float, valid: bool = True) -> dict:
    return {"model_key": key, "weight": weight, "valid": valid}


def _summary_with_empty_galaxy() -> dict:
    ok = {
        "n_valid": 2,
        "models": [_model("a", 0.25), _model("b", 0.75)],
    }
    empty = {"n_valid": 0, "models": [], "reason": "No cells found for named_all"}
    return {"galaxies": {"1": {"sets": {"s": ok}}, "2": {"sets": {"s": empty}}}}


def test_fig11_skips_a_galaxy_with_no_valid_model_and_reports_it():
    from analysis.paper1.fig11_bma import _collect_weight_set

    out = _collect_weight_set(_summary_with_empty_galaxy(), "s", [1, 2])

    weights_by_gal, skipped = out[1], out[-1]
    assert skipped == [2]
    assert set(weights_by_gal) == {1}
    assert sum(weights_by_gal[1].values()) == pytest.approx(1.0)


def test_fig11_still_rejects_weights_that_do_not_sum_to_one():
    from analysis.paper1.fig11_bma import _collect_weight_set

    summary = _summary_with_empty_galaxy()
    summary["galaxies"]["1"]["sets"]["s"]["models"][1]["weight"] = 0.5

    with pytest.raises(ValueError, match="sum to"):
        _collect_weight_set(summary, "s", [1, 2])


# --- census resolver ------------------------------------------------------------


def test_census_resolves_xlike_and_grid_keys_through_one_resolver():
    import analysis.paper1.configs as configs
    import analysis.paper1.xlike_configs as xlike
    from analysis.paper1.fig06_code_overlay import resolve_configuration

    builder, ssp_loader = resolve_configuration("prospector_like", configs)
    assert builder is xlike.XLIKE_BUILDERS["prospector_like"]
    assert ssp_loader.func is xlike.load_ssp_for_xlike
    assert ssp_loader.args == ("prospector_like",)

    grid_builder, grid_loader = resolve_configuration("II", configs)
    assert grid_builder is configs.config_II
    assert grid_loader.func is configs.load_ssp_for

    with pytest.raises(KeyError, match="neither a grid key"):
        resolve_configuration("not_a_config", configs)


def test_census_suite_keys_come_from_config_metadata():
    from analysis.paper1.config_metadata import XLIKE_KEYS
    from analysis.paper1.surviving_mass_census import suite_config_keys

    assert suite_config_keys("xlike") == list(XLIKE_KEYS)
    assert "prospector_like" not in suite_config_keys("grid")
    with pytest.raises(ValueError, match="unknown suite"):
        suite_config_keys("nope")
