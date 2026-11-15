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
import itertools
import re
from pathlib import Path

import numpy as np
import pytest
from reproduction.beagle._drivers import beagle_matrix as M, beagle_output as O, beagle_run as B

pytestmark = pytest.mark.contract

TEMPLATE_LINE = re.compile(r"^TEMPLATES( NEBULAR)? = ", re.MULTILINE)
# Sections with no IGM keyword: the no-IGM controls and the z = 0.001 nebular-physics block.
NO_IGM_PREFIXES = ("s12_igm_none", "s8b_", "s8c_", "s8d_", "s8e_", "s8f_", "s9_")


def _synthetic_mock_file(path: Path, n_obj: int = 3, n_wl: int = 2198) -> Path:
    """A tiny ``--mock`` output FITS with the extensions ``parse_mock_output`` reads."""
    from astropy.io import fits

    wl = np.geomspace(1000.0, 10000.0, n_wl)
    flam = np.full((n_obj, n_wl), 1e-20, dtype=np.float32)
    hdus = [
        fits.PrimaryHDU(),
        fits.BinTableHDU.from_columns(
            [
                fits.Column(name="luminosity_distance", format="1D", array=np.full(n_obj, 1e26)),
                fits.Column(name="M_star", format="1E", array=np.linspace(9.0, 10.0, n_obj)),
            ],
            name="GALAXY PROPERTIES",
        ),
        fits.BinTableHDU.from_columns(
            [fits.Column(name="wl", format=f"{n_wl}D", array=wl[None, :])], name="FULL SED WL"
        ),
        fits.ImageHDU(data=flam, name="FULL SED"),
    ]
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


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


def test_parser_round_trips_mock_fits(tmp_path: Path) -> None:
    parsed = O.parse_mock_output(_synthetic_mock_file(tmp_path / "mock.fits"))
    assert parsed.sed_lnu is not None
    assert parsed.sed_lnu.shape == (3, 2198)
    assert np.all(np.isfinite(parsed.sed_lnu))
    assert parsed.wl_rest_aa is not None
    assert parsed.wl_rest_aa.shape == (2198,)
    assert parsed.galaxy_properties is not None
    assert "luminosity_distance" in parsed.galaxy_properties


def test_cache_round_trip_preserves_arrays(tmp_path: Path) -> None:
    mock = _synthetic_mock_file(tmp_path / "mock.fits")
    parsed = O.parse_mock_output(mock, provenance={"image": "test"})
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
def test_igm_is_inoue_unless_the_section_is_a_no_igm_section(run: B.MockRun) -> None:
    lines = run.param_text().splitlines()
    igm = [line for line in lines if line.startswith("IGM ABSORPTION")]
    if run.section.startswith(NO_IGM_PREFIXES):
        assert run.igm is None
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
            f"min_rel_err:0.02 flux:colName:flux_{spec.name} fluxErr:colName:fluxerr_{spec.name}"
        )
    assert all(spec.min_rel_err == 0.02 for spec in run.filters)


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
    """Run as ``python -m``, the module is ``__main__``; a fit must still be recognized."""
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


def _synthetic_fit_file(path: Path, n_samples: int = 5, n_wl: int = 6) -> None:
    """A tiny ``<ID>_BEAGLE.fits.gz`` with the layout of a real fit (no BEAGLE data)."""
    from astropy.io import fits

    rng = np.random.default_rng(0)
    prob = np.arange(1.0, n_samples + 1.0)
    wl = np.linspace(1000.0, 3000.0, n_wl)
    image = np.tile(np.arange(n_samples, dtype=np.float32)[:, None], (1, n_wl)) + 1.0
    image = image + rng.uniform(0.0, 1e-3, size=image.shape).astype(np.float32)
    cols = [
        fits.Column(name="probability", format="1E", array=prob),
        fits.Column(name="tau", format="1E", array=prob * 0.5),
    ]
    hdus = [
        fits.PrimaryHDU(),
        fits.BinTableHDU.from_columns(cols, name="POSTERIOR PDF"),
        fits.BinTableHDU.from_columns(
            [fits.Column(name="M_star", format="1E", array=prob)], name="GALAXY PROPERTIES"
        ),
        fits.BinTableHDU.from_columns(
            [fits.Column(name="wl", format=f"{n_wl}E", array=wl[None, :])], name="FULL SED WL"
        ),
        fits.ImageHDU(data=image, name="FULL SED"),
    ]
    fits.HDUList(hdus).writeto(path, overwrite=True)


def test_fit_parser_reduces_image_sed_to_weighted_percentiles(tmp_path: Path) -> None:
    _synthetic_fit_file(tmp_path / "7_BEAGLE.fits.gz")
    parsed = O.parse_fit_output(tmp_path, ("7",))
    per = parsed.posterior["7"]
    assert set(per) == {"POSTERIOR PDF", "GALAXY PROPERTIES", "FULL SED WL", "FULL SED"}
    assert np.array_equal(per["POSTERIOR PDF"]["probability"], np.arange(1.0, 6.0))
    sed = per["FULL SED"]
    assert set(sed) == {"wl", "p16", "p50", "p84"}
    assert sed["wl"].shape == (6,)
    # Row k holds value k+1 with weight k+1, so the normalized cumulative weights are
    # 1/15, 3/15, 6/15, 10/15, 1. The first row reaching 0.16 is row 1 (value 2), 0.50 is
    # row 3 (value 4) and 0.84 is row 4 (value 5).
    assert np.allclose(sed["p16"], 2.0, atol=1e-2)
    assert np.allclose(sed["p50"], 4.0, atol=1e-2)
    assert np.allclose(sed["p84"], 5.0, atol=1e-2)


def test_fit_parser_survives_a_cache_round_trip(tmp_path: Path) -> None:
    _synthetic_fit_file(tmp_path / "7_BEAGLE.fits.gz")
    parsed = O.parse_fit_output(tmp_path, ("7",), provenance={"tag": "test"})
    O.save_cache(tmp_path, "synthetic", parsed)
    loaded = O.load_cache(tmp_path, "synthetic")
    assert np.array_equal(
        loaded.posterior["7"]["FULL SED"]["p50"], parsed.posterior["7"]["FULL SED"]["p50"]
    )
    assert loaded.provenance["tag"] == "test"


# ── Nebular-physics and AGN sections ────────────────────────────────────────
AGN_SECTIONS = ("s9_agn", "s9_agn_only")


@pytest.mark.parametrize("section", AGN_SECTIONS)
def test_agn_sections_carry_the_agn_template_and_no_rebin(section: str) -> None:
    text = M.by_section(section)[0].param_text()
    assert f"TEMPLATES AGN = $BEAGLE_TEMPLATES/{B.AGN_LISTING}" in text
    assert "PRINT AGN SPECTRUM = T" in text
    assert "TEMPLATES NEBULAR = " in text
    assert "EMISSION LINES CONFIGURATION = " in text
    assert "REBIN TEMPLATES" not in text


@pytest.mark.parametrize("run", [r for r in M.RUN_MATRIX if not r.agn], ids=lambda r: r.section)
def test_non_agn_sections_carry_no_agn_keywords(run: B.MockRun) -> None:
    text = run.param_text()
    assert "TEMPLATES AGN" not in text
    assert "PRINT AGN SPECTRUM" not in text


def test_agn_grid_covers_the_declared_nodes() -> None:
    grid = M.by_section("s9_agn")[0]
    assert len(grid.rows) == 24
    assert {r["agn_lacc"] for r in grid.rows} == {43.0, 44.0, 45.0, 46.0}
    combos = {(r["agn_logU"], r["agn_Z"]) for r in grid.rows}
    assert combos == set(itertools.product((-3.5, -2.5, -1.5), (-0.5, 0.0)))
    assert "PARAMETER = name:agn_xi mock:type:fixed value:0.3" in grid.param_text()
    only = M.by_section("s9_agn_only")[0]
    assert len(only.rows) == 6
    assert all(r["sfr"] == -3.0 and r["mass"] == 8.0 for r in only.rows)
    assert "PARAMETER = name:agn_lacc mock:type:fixed value:45.0" in only.param_text()


def test_agn_run_needs_nebular_emission_and_no_rebin() -> None:
    row = {"redshift": 0.001, "mass": 10.0}
    with pytest.raises(ValueError, match="AGN"):
        B.MockRun(section="bad", rows=(row,), templates="bc03", agn=True, rebin=None)
    with pytest.raises(ValueError, match="AGN"):
        B.MockRun(section="bad", rows=(row,), templates="ineb_june16", nebular=True, agn=True)


@pytest.mark.parametrize(("section", "flag"), [("s8c_hiidust_on", "T"), ("s8c_hiidust_off", "F")])
def test_hii_dust_keyword_follows_the_section(section: str, flag: str) -> None:
    assert f"ACCOUNT FOR HII DUST = {flag}" in M.by_section(section)[0].param_text()


def test_hii_dust_keyword_absent_when_unset() -> None:
    assert "ACCOUNT FOR HII DUST" not in M.by_section("s8_nebular")[0].param_text()


def test_lyc_escape_fraction_is_a_from_file_parameter() -> None:
    run = M.by_section("s8d_fesc")[0]
    assert "PARAMETER = name:LyC_f_esc type:from_file" in run.param_text()
    assert sorted(r["LyC_f_esc"] for r in run.rows) == [0.0, 0.1, 0.3, 0.6]


def test_episode_section_varies_the_current_timescale_per_row() -> None:
    run = M.by_section("s8b_episode")[0]
    assert len(run.rows) == 7
    assert "SF_PARAMETER = name:current_sfr_timescale type:from_file" in run.param_text()
    off_fiducial = {(r["sfr"], r["current_sfr_timescale"]) for r in run.rows}
    assert {(0.5, 7.5), (0.5, 8.0)} <= off_fiducial
    assert {r["sfr"] for r in run.rows if r["current_sfr_timescale"] == 7.0} == {
        -1.0,
        0.0,
        0.5,
        1.0,
        1.5,
    }


def test_const100_uses_the_s8_node_grid() -> None:
    const = M.by_section("s8e_const100")[0]
    assert const.rows == M.by_section("s8_nebular")[0].rows
    assert "max_stellar_age mock:type:fixed value:8.0" in const.param_text()


@pytest.mark.parametrize(
    ("value", "expected"),
    [(609.789978, "609.789978"), (615.0, "615.000000"), (10829.9, "10829.9004")],
)
def test_lambda_suffix_is_float32_with_nine_significant_digits(
    value: float, expected: str
) -> None:
    assert B._lambda_suffix(value) == expected


def _write_line_list(path: Path, entries: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"wl:{wl} use:T print:T label:{label}\n" for wl, label in entries))


def _write_agn_tables(src: Path) -> None:
    from astropy.io import fits

    src.mkdir(parents=True, exist_ok=True)
    (src / "nlr_list.dat").write_text("nlr_test.fits\n")
    wl_names = ["wl_cont", "Mg10@610", "@615", "@1909"]
    wl_values = [1000.0, 609.789978, 615.0, 1907.7]
    hdus = [
        fits.PrimaryHDU(),
        fits.BinTableHDU.from_columns(
            [fits.Column(name=n, format="1E", array=[v]) for n, v in zip(wl_names, wl_values)],
            name="WAVELENGTHS",
        ),
        fits.BinTableHDU.from_columns(
            [fits.Column(name="xi", format="1E", array=[0.3])], name="PARAMETERS"
        ),
        fits.BinTableHDU.from_columns(
            [
                fits.Column(name="Z", format="1E", array=[0.02]),
                fits.Column(name="Mg10@610", format="1E", array=[1.0]),
                fits.Column(name="@1909", format="1E", array=[2.0]),
            ],
            name="LOGU_1.5",
        ),
    ]
    fits.HDUList(hdus).writeto(src / "nlr_test.fits", overwrite=True)


def test_prepare_agn_templates_names_columns_the_way_beagle_looks_them_up(tmp_path: Path) -> None:
    from astropy.io import fits

    _write_agn_tables(tmp_path / "AGN_NLR")
    _write_line_list(
        tmp_path / B.NEBULAR_TEMPLATES["ineb_june16"][1],
        [("609.79", "Mg10@610"), ("615.00", "@615"), ("1907.70", "X1"), ("1500.00", "Far")],
    )
    out = B.prepare_agn_templates(tmp_path)
    assert out == tmp_path / B.AGN_TEMPLATE_DIR
    assert (out / "nlr_list.dat").read_text() == "nlr_test.fits\n"
    with fits.open(out / "nlr_test.fits") as hdul:
        assert hdul["WAVELENGTHS"].columns.names == [
            "wl_cont",
            "Mg10@610_609.789978",
            "@615_615.000000",
            "@1909_1907.69995",
        ]
        assert hdul["PARAMETERS"].columns.names == ["xi"]
        assert hdul["LOGU_1.5"].columns.names == ["Z", "Mg10@610_609.789978", "@1909_1907.69995"]
        assert hdul["LOGU_1.5"].data["@1909_1907.69995"][0] == pytest.approx(2.0)


def test_prepare_agn_templates_refuses_an_unmatched_column(tmp_path: Path) -> None:
    _write_agn_tables(tmp_path / "AGN_NLR")
    _write_line_list(tmp_path / B.NEBULAR_TEMPLATES["ineb_june16"][1], [("1500.00", "Far")])
    with pytest.raises(ValueError, match="AGN column"):
        B.prepare_agn_templates(tmp_path)


def test_prepare_agn_templates_needs_the_downloaded_tables(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="AGN_NLR"):
        B.prepare_agn_templates(tmp_path)


def _synthetic_agn_mock(path: Path, n_obj: int = 2, n_wl: int = 6) -> Path:
    from astropy.io import fits

    wl = np.linspace(1000.0, 5000.0, n_wl)
    hdus = [
        fits.PrimaryHDU(),
        fits.BinTableHDU.from_columns(
            [fits.Column(name="luminosity_distance", format="1D", array=np.full(n_obj, 1e26))],
            name="GALAXY PROPERTIES",
        ),
        fits.BinTableHDU.from_columns(
            [fits.Column(name="wl", format=f"{n_wl}D", array=wl[None, :])], name="FULL SED WL"
        ),
        fits.ImageHDU(data=np.full((n_obj, n_wl), 1e-20), name="FULL SED"),
        fits.BinTableHDU.from_columns(
            [
                fits.Column(name="agn_present", format="1E", array=np.ones(n_obj)),
                fits.Column(name="Mg10@610_609.789978_lum_obs", format="1E", array=[1.0e5, 2.0e5]),
                fits.Column(name="Mg10@610_609.789978_flux", format="1E", array=[1e-15, 2e-15]),
            ],
            name="AGN EMISSION",
        ),
        fits.ImageHDU(data=np.full((n_obj, n_wl), 3e-21), name="AGN FULL SED"),
    ]
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def test_parser_reads_agn_lines_and_the_agn_sed(tmp_path: Path) -> None:
    parsed = O.parse_mock_output(_synthetic_agn_mock(tmp_path / "agn.fits"))
    assert set(parsed.agn_line_lums_lsun) == {"Mg10@610_609.79_lum_obs"}
    assert np.array_equal(parsed.agn_line_lums_lsun["Mg10@610_609.79_lum_obs"], [1e5, 2e5])
    assert parsed.agn_sed is not None
    assert parsed.agn_sed.shape == (2, 6)
    # Same 4 pi d_L^2 convention as the total SED: the AGN flux is 0.3 of the total here.
    assert np.allclose(parsed.agn_sed, 0.3 * parsed.sed_lnu)


def test_agn_fields_survive_a_cache_round_trip(tmp_path: Path) -> None:
    parsed = O.parse_mock_output(_synthetic_agn_mock(tmp_path / "agn.fits"))
    O.save_cache(tmp_path, "agn", parsed)
    loaded = O.load_cache(tmp_path, "agn")
    assert np.array_equal(loaded.agn_sed, parsed.agn_sed)
    assert set(loaded.agn_line_lums_lsun) == set(parsed.agn_line_lums_lsun)
    assert np.array_equal(
        loaded.agn_line_lums_lsun["Mg10@610_609.79_lum_obs"],
        parsed.agn_line_lums_lsun["Mg10@610_609.79_lum_obs"],
    )


def test_mock_without_agn_extensions_leaves_the_fields_none(tmp_path: Path) -> None:
    parsed = O.parse_mock_output(_synthetic_mock_file(tmp_path / "plain.fits"))
    assert parsed.agn_sed is None
    assert parsed.agn_line_lums_lsun is None
    O.save_cache(tmp_path, "plain", parsed)
    loaded = O.load_cache(tmp_path, "plain")
    assert loaded.agn_sed is None
    assert loaded.agn_line_lums_lsun is None
