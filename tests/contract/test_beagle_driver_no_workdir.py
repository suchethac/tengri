# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the BEAGLE engine driver works without a workdir and parses the smoke output.

The driver writes parameter files, invokes the BEAGLE binary inside Docker, and
parses the native FITS. None of that needs a BEAGLE installation on the machine
running the tests, so these checks run in CI: the availability guard, the shape
of every built-in run's parameter file and input table, the cache-key stability,
the cache-miss refusal, and the output parser on a local smoke run when one is
present.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest
from reproduction.beagle._drivers import beagle_matrix as M, beagle_output as O, beagle_run as B

pytestmark = pytest.mark.contract

SMOKE_FITS = (
    Path.home() / "beagle" / "BEAGLE-general" / "results" / "smoke_bc03_mock" / "smoke_bc03.fits"
)
TEMPLATE_LINE = re.compile(r"^TEMPLATES( NEBULAR)? = ", re.MULTILINE)


def test_unset_workdir_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BEAGLE_WORKDIR", raising=False)
    assert B.available() is False
    with pytest.raises(RuntimeError) as excinfo:
        B.require_available()
    message = str(excinfo.value)
    assert "BEAGLE_WORKDIR" in message
    assert "docker pull" in message


@pytest.mark.parametrize("run", M.RUN_MATRIX, ids=lambda r: r.section)
def test_matrix_parameter_file_is_well_formed(run: B.MockRun) -> None:
    text = run.param_text()
    assert isinstance(text, str)
    assert "\t" not in text
    assert len(TEMPLATE_LINE.findall(text)) == 1
    assert "RESULTS DIRECTORY" in text
    assert "MOCK CATALOGUE NAME" in text
    assert "HUBBLE_0 = 67.66" in text
    assert "OMEGA_MATTER = 0.30966" in text
    assert "OMEGA_LAMBDA = 0.69034" in text


@pytest.mark.parametrize("run", M.RUN_MATRIX, ids=lambda r: r.section)
def test_matrix_input_table_matches_object_count(run: B.MockRun) -> None:
    table = run.input_table()
    assert len(table) == len(run.rows)
    assert f"N_OBJECTS = {len(table)}" in run.param_text()
    assert set(table.colnames) == {
        line.split("name:")[1].split()[0]
        for line in run.param_text().splitlines()
        if "type:from_file" in line
    }


def test_cache_key_is_stable_across_constructions() -> None:
    first = M.RUN_MATRIX[0]
    second = dataclasses.replace(first)
    assert first.key == second.key
    assert first.name == f"{first.section}_{first.key}"


def test_cache_miss_without_docker_names_the_run_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "Docker_env.list").write_text("BEAGLE_TEMPLATES=/x\n")
    monkeypatch.setenv("BEAGLE_WORKDIR", str(tmp_path))
    run = M.RUN_MATRIX[0]
    with pytest.raises(RuntimeError, match=f"--run {run.section}"):
        B.run(run, allow_docker=False)


def test_param_rejects_tabs() -> None:
    with pytest.raises(ValueError):
        B.Param("redshift", "type:from_file", prior="prior:\tx").render()


def test_parser_round_trips_smoke_fits() -> None:
    if not SMOKE_FITS.is_file():
        pytest.skip(f"no local BEAGLE smoke output at {SMOKE_FITS}")
    parsed = O.parse_mock_output(SMOKE_FITS)
    assert parsed.sed_lnu is not None
    assert parsed.sed_lnu.shape == (3, 2198)
    assert np.all(np.isfinite(parsed.sed_lnu))
    assert parsed.wl_rest_aa is not None
    assert parsed.wl_rest_aa.shape == (2198,)
    assert parsed.galaxy_properties is not None
    assert "luminosity_distance" in parsed.galaxy_properties


def test_cache_round_trip_preserves_arrays(tmp_path: Path) -> None:
    if not SMOKE_FITS.is_file():
        pytest.skip(f"no local BEAGLE smoke output at {SMOKE_FITS}")
    parsed = O.parse_mock_output(SMOKE_FITS, provenance={"image": "test"})
    O.save_cache(tmp_path, "smoke", parsed)
    loaded = O.load_cache(tmp_path, "smoke")
    assert np.array_equal(loaded.sed_lnu, parsed.sed_lnu)
    assert np.array_equal(loaded.wl_rest_aa, parsed.wl_rest_aa)
    assert set(loaded.galaxy_properties) == set(parsed.galaxy_properties)
    assert loaded.continuum_sed is None
    assert loaded.posterior is None
    assert loaded.provenance["image"] == "test"


def test_ssp_section_does_not_print_the_sfh() -> None:
    ssp = M.by_section("s1_ssp_bc03")[0]
    text = ssp.param_text()
    assert "sfh_type" in text and "char_value:ssp" in text
    assert "PRINT SF AND ChE HISTORIES = F" in text
    assert "PRINT SF AND ChE HISTORIES = T" not in text
    delayed = M.by_section("s2_sfh_delayed")[0].param_text()
    assert "PRINT SF AND ChE HISTORIES = T" in delayed


@pytest.mark.parametrize("run", M.RUN_MATRIX, ids=lambda r: r.section)
def test_igm_is_inoue_unless_the_section_is_the_no_igm_control(run: B.MockRun) -> None:
    lines = run.param_text().splitlines()
    igm = [line for line in lines if line.startswith("IGM ABSORPTION")]
    if run.section.startswith("s12_igm_none"):
        assert igm == []
    else:
        assert igm == ["IGM ABSORPTION = Inoue"]


def test_igm_sections_differ_only_in_the_igm_keyword() -> None:
    inoue = M.by_section("s12_igm_inoue_z3")[0]
    none = M.by_section("s12_igm_none_z3")[0]
    assert len(inoue.filters) == 12
    assert len(none.filters) == 12

    def stripped(run: B.MockRun) -> set[str]:
        skip = ("RESULTS DIRECTORY", "MOCK CATALOGUE NAME", "MOCK INPUT", "FILTERS CONFIGURATION")
        return {line for line in run.param_text().splitlines() if not line.startswith(skip)}

    assert stripped(inoue) ^ stripped(none) == {"IGM ABSORPTION = Inoue"}


def test_ccww13_full_is_not_run() -> None:
    assert all(run.section != "s4_ccww13_full" for run in M.RUN_MATRIX)
    assert all(run.dust != "CCWW13_full" for run in M.RUN_MATRIX)


@pytest.mark.parametrize(
    ("section", "law"),
    [
        ("s4_cf00", "CF00"),
        ("s4_calzetti", "Calzetti"),
        ("s4_ccww13_universal", "CCWW13_universal"),
        ("s4_mw", "mw"),
        ("s4_lmc", "lmc"),
        ("s4_smc", "smc"),
    ],
)
def test_attenuation_sections_carry_their_law_and_tauv(section: str, law: str) -> None:
    run = M.by_section(section)[0]
    text = run.param_text()
    assert len(run.rows) == 1
    assert f"PARAMETER = name:attenuation_type mock:type:fixed char_value:{law}" in text
    assert "PARAMETER = name:tauV_eff mock:type:fixed value:1.0" in text


@pytest.mark.parametrize("section", ["s4_cf00", "s4_ccww13_universal"])
def test_mu_is_set_where_the_manual_requires_it(section: str) -> None:
    assert "PARAMETER = name:mu mock:type:fixed value:0.4" in M.by_section(section)[0].param_text()


@pytest.mark.parametrize("section", ["s4_calzetti", "s4_mw", "s4_lmc", "s4_smc"])
def test_mu_is_omitted_where_the_manual_does_not_require_it(section: str) -> None:
    assert "name:mu" not in M.by_section(section)[0].param_text()


def test_capstone_renders_without_docker_and_names_the_four_galaxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reproduction.beagle._drivers import beagle_capstone as C

    monkeypatch.delenv("BEAGLE_WORKDIR", raising=False)
    run = C.capstone_run()
    text = run.param_text()
    assert "# objects: 79 7837 15775 1826" in text
    assert len(run.catalog_table) == 4
    assert len(run.filters) == 17
    assert "TEMPLATES NEBULAR = " in text
    assert "PRIORS CATALOGUE = " in text
    assert (
        "PARAMETER = name:redshift type:fitted prior:distribution:dirac prior:colName:zz" in text
    )
    for line in (
        "PRINT SPECTRUM = T",
        "PRINT PHOTOMETRY = T",
        "PRINT LINE FLUXES = T",
        "PRINT CONTINUUM = T",
    ):
        assert line in text
    assert "PDF SAMPLER FILE = $BEAGLE_PARAM_DIR/MCMC_example.param" in text


def test_run_commands_need_the_allow_docker_flag(capsys: pytest.CaptureFixture[str]) -> None:
    from reproduction.beagle._drivers import beagle_run as R

    assert R.main(["--run", "s1_ssp_bc03"]) == 1
    assert "--allow-docker" in capsys.readouterr().err
    assert R.main(["--fit-capstone"]) == 0
    assert "--allow-docker" in capsys.readouterr().out


def test_capstone_throughputs_are_dimensionless_and_nonnegative() -> None:
    from reproduction.beagle._drivers import beagle_capstone as C

    run = C.capstone_run()
    for spec in run.filters:
        assert np.all(spec.transmission >= 0.0)
        assert spec.transmission.max() <= 1.0
    k_band = next(f for f in run.filters if f.name == "ISAAC_KS")
    assert 0.0 < k_band.transmission.max() < 1.0


def test_capstone_filter_configuration_follows_the_fit_example() -> None:
    from reproduction.beagle._drivers import beagle_capstone as C

    run = C.capstone_run()
    lines = B._filter_config(run.name, run.filters, fitting=True).splitlines()
    assert lines[0] == "units:microJy"
    assert lines[1] == "object_ID:colName:ID"
    band_lines = lines[2:]
    assert len(band_lines) == len(run.filters)
    for line, spec in zip(band_lines, run.filters, strict=True):
        assert line == (
            f"name:{spec.name} fileName:$BEAGLE_FILTERS/{run.name}_{spec.name}.dat "
            f"flux:colName:flux_{spec.name} fluxErr:colName:fluxerr_{spec.name}"
        )
    assert "min_rel_err" not in "\n".join(lines)


def test_photometry_only_configuration_has_no_object_id_line() -> None:
    run = M.by_section("s12_igm_inoue_z3")[0]
    lines = B._filter_config(run.name, run.filters, fitting=False).splitlines()
    assert lines[0] == "units:microJy"
    assert all(not line.startswith("object_ID") for line in lines)
    assert all("flux:colName" not in line for line in lines)


@pytest.mark.parametrize("z", [1, 3, 6])
@pytest.mark.parametrize("label", ["inoue", "none"])
def test_igm_sections_run_at_native_template_resolution(label: str, z: int) -> None:
    run = M.by_section(f"s12_igm_{label}_z{z}")[0]
    assert run.rebin is None
    assert "REBIN TEMPLATES" not in run.param_text()


@pytest.mark.parametrize("label", ["inoue", "none"])
def test_rebinned_z6_pair_is_kept_for_comparison(label: str) -> None:
    run = M.by_section(f"s12_igm_{label}_z6_rebin10")[0]
    assert run.rebin == 10
    assert "REBIN TEMPLATES = 10" in run.param_text()
    native = M.by_section(f"s12_igm_{label}_z6")[0]
    assert run.rows == native.rows
    assert [f.name for f in run.filters] == [f.name for f in native.filters]
    for a, b in zip(run.filters, native.filters, strict=True):
        assert np.array_equal(a.wl_aa, b.wl_aa)
        assert np.array_equal(a.transmission, b.transmission)


def test_module_entry_point_renders_the_fit_configuration(tmp_path: Path) -> None:
    """Run as ``python -m``, the driver's module is ``__main__``; a fit must still be recognised."""
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[2]
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(repo / "src") + ":" + str(repo)}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "reproduction.beagle._drivers.beagle_run",
            "--render-params",
            "--out",
            str(tmp_path),
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    config = next(tmp_path.glob("filters/capstone_paper1_*_config.dat")).read_text().splitlines()
    assert config[1] == "object_ID:colName:ID"
