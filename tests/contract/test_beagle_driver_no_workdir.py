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


def test_ccww13_full_section_carries_the_tuffs_geometry() -> None:
    text = M.by_section("s4_ccww13_full")[0].param_text()
    assert "PARAMETER = name:attenuation_type mock:type:fixed char_value:CCWW13_full" in text
    assert "PARAMETER = name:inclination mock:type:fixed value:60.0" in text
    assert "PARAMETER = name:tau_b_perp mock:type:fixed value:4.0" in text
    assert "PARAMETER = name:tauV_eff mock:type:fixed value:1.0" in text
    assert "PARAMETER = name:mu mock:type:fixed value:0.4" in text
    universal = M.by_section("s4_ccww13_universal")[0].param_text()
    assert "tau_b_perp" not in universal and "inclination" not in universal


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
