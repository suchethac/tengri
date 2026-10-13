# SPDX-License-Identifier: BSD-3-Clause
"""BPASS surviving mass: the registry records the grid's origin and why it has no table (#2800).

``bpss_stars_c3k_a_chabrier`` is ProSpect's ``BPASS_St_Ch300`` repackaging of BPASS v2.2.1
(Eldridge et al. 2017, PASA 34, e058; Stanway & Eldridge 2018, MNRAS 479, 75): binary
populations, ``imf_chab300``, 13 metallicities, 51 ages. tengri's surviving mass is living
stars plus remnants per unit formed mass. BPASS's ``starmass`` column 2 is living stars
only, and column 3 (remnants) is flagged untested by the BPASS manual.

The alternative is mass balance, ``living + remnants = M0 - E(t)`` with ``E`` the wind and
supernova ejecta integrated from the ``yields`` files. It needs the two BPASS outputs to
close. They do not: at the youngest ages, before any supernova, ``living + E`` differs from
``M0`` by up to 1.0 per cent for every placement of ``E`` within a bin, and ``living + E``
exceeds ``M0`` (mass created) by up to 0.8 per cent even when ``E`` is the strict lower
bound. So no table is built, the registry row stays PENDING, and the explicit
``mass_remaining="dsps_fit"`` opt-in remains the only way to load the grid.

Expected values below are re-derived from the committed excerpt of the BPASS files with
plain numpy, following the manual's bin rule: 0.1 dex bins centered on log10(age/yr) =
6.0, 6.1, ..., each ``10**(la + 0.05) - 10**(la - 0.05)`` yr wide; the first bin ends at
10**6.05 yr.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "tests" / "fixtures" / "bpass_chab300" / "mass_balance.csv"
GRID_STEM = "bpss_stars_c3k_a_chabrier"
M0 = 1.0e6
#: Rows for log10 age 6.0 ... 10.2. The supernova columns of the yields files at log10 age
#: 10.3-10.6 hold 30-300 Msun/yr (1e11-1e12 Msun per bin), which is not a physical age
#: of this population, so E is not compared beyond 10.2.
ROWS_TO_10P2 = 43


def _fixture() -> dict[str, np.ndarray]:
    """Rows of the committed excerpt by metallicity code: (n, 5) float arrays."""
    rows: dict[str, list[list[float]]] = {}
    with FIXTURE.open(newline="") as handle:
        reader = csv.reader(line for line in handle if not line.startswith("#"))
        next(reader)
        for row in reader:
            rows.setdefault(row[0], []).append([float(v) for v in row[1:]])
    return {k: np.array(v) for k, v in rows.items()}


def _ejecta(rows: np.ndarray, first_bin_start_yr: float, placement: str) -> np.ndarray:
    """E(t) [Msun] by a plain loop over bins; ``placement`` is 'upper', 'lower' or 'mid'."""
    out = np.zeros(rows.shape[0])
    total = 0.0
    for k, la in enumerate(rows[:, 0]):
        start = first_bin_start_yr if k == 0 else 10.0 ** (la - 0.05)
        mass = (rows[k, 3] + rows[k, 4]) * (10.0 ** (la + 0.05) - start)
        out[k] = {"upper": total + mass, "lower": total, "mid": total + 0.5 * mass}[placement]
        total += mass
    return out


# ---------------------------------------------------------------------------
# The grid's origin and the PENDING reason are recorded where the user meets them.
# ---------------------------------------------------------------------------
def _write_grid(path: Path) -> Path:
    age = np.arange(6.0, 11.01, 0.1) - 9.0
    logz = np.log10([1e-5, 1e-3, 1.4e-2, 4e-2])
    with h5py.File(path, "w") as f:
        f["ssp_wave"] = np.linspace(1000.0, 20000.0, 20)
        f["ssp_flux"] = np.full((len(logz), len(age), 20), 1e-4)
        f["ssp_lg_age_gyr"] = age
        f["ssp_lgmet"] = logz
    return path


def test_refusal_names_the_grid_origin_and_the_mass_balance_finding(tmp_path):
    """The refusal says which BPASS product the grid is and why no table exists."""
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    path = _write_grid(tmp_path / f"{GRID_STEM}.h5")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError) as err:
            load_ssp_data(str(path))
    text = str(err.value)
    assert "BPASS_St_Ch300" in text
    assert "imf_chab300" in text
    assert "mass_remaining='dsps_fit'" in text
    assert "yields" in text and "starmass" in text


def test_doctor_line_for_bpass_records_origin():
    from tengri.components.stellar.sps.mass_remaining_tables import describe_source

    line = describe_source(GRID_STEM)
    assert line.startswith("PENDING")
    assert "BPASS v2.2.1" in line and "imf_chab300" in line


def test_registry_origin_matches_the_grid_it_describes():
    """The recorded origin agrees with the shipped grid's own axes (13 Z, 51 ages)."""
    from tengri.components.stellar.sps import mass_remaining_tables as mrt

    assert mrt.MASS_REMAINING_REGISTRY[GRID_STEM].source == mrt.PENDING
    origin = mrt.GRID_ORIGIN[GRID_STEM]
    for token in ("ProSpect", "BPASS_St_Ch300", "BPASS v2.2.1", "binary", "imf_chab300"):
        assert token in origin
    try:
        with h5py.File(REPO / "data" / f"{GRID_STEM}.h5", "r") as f:
            n_met, n_age = f["ssp_lgmet"].shape[0], f["ssp_lg_age_gyr"].shape[0]
            first, last = float(f["ssp_lg_age_gyr"][0]), float(f["ssp_lg_age_gyr"][-1])
    except FileNotFoundError:
        pytest.skip("data/bpss_stars_c3k_a_chabrier.h5 is not present")
    assert (n_met, n_age) == (13, 51)
    assert abs(first + 9.0 - 6.0) < 1e-6 and abs(last + 9.0 - 11.0) < 1e-6


# ---------------------------------------------------------------------------
# The mass-balance route does not close: re-derived from the committed excerpt.
# ---------------------------------------------------------------------------
@pytest.mark.bounds
def test_excerpt_shape_and_unit_checks():
    """13 metallicities x 50 rows (log10 age 6.0-10.9; the 11.0 row is not physical)."""
    data = _fixture()
    assert len(data) == 13
    for rows in data.values():
        assert rows.shape == (50, 5)
        assert np.allclose(np.diff(rows[:, 0]), 0.1, atol=1e-6)
        assert rows[0, 1] == M0  # initial mass, 1e6 Msun
        assert np.all(rows[:, 1] <= M0) and np.all(rows[:, 3:] >= 0.0)


@pytest.mark.conservation
def test_living_plus_ejecta_exceeds_the_formed_mass_before_the_first_supernovae():
    """z001 at log age 6.3: living is exactly M0 yet the winds before it ejected > 1e-3 M0.

    Strict lower bound on E: bins up to the lower edge of the 6.3 bin, with the first bin
    starting at 10**6.0 yr (so E can only be larger). A mass-conserving pair of files
    could not have living + E above M0.
    """
    rows = _fixture()["z001"]
    k = int(np.argmin(abs(rows[:, 0] - 6.3)))
    assert rows[k, 1] == M0
    lower = _ejecta(rows, 10.0**6.0, "lower")[k]
    assert lower / M0 > 1.0e-3
    assert (rows[k, 1] + lower - M0) / M0 == pytest.approx(1.45e-3, abs=5e-5)


@pytest.mark.conservation
def test_closure_fails_for_every_placement_of_the_ejecta_within_a_bin():
    """Max |living + E - M0| / M0 over log age < 6.35 exceeds the 1e-3 gate at some Z."""
    data = _fixture()
    for first_start in (10.0**6.0, 0.0):
        for placement in ("upper", "lower", "mid"):
            worst = 0.0
            for rows in data.values():
                pre = rows[:, 0] < 6.35
                res = rows[pre, 1] + _ejecta(rows, first_start, placement)[pre] - M0
                worst = max(worst, float(np.max(np.abs(res))) / M0)
            assert worst > 1.0e-3, (first_start, placement, worst)


@pytest.mark.conservation
def test_mass_created_below_10_myr_even_for_the_lower_bound():
    """living + E - M0 > 0 for log age <= 7 at the lower bound: up to 8.1e-3 M0 (z003)."""
    data = _fixture()
    worst, where = 0.0, ""
    for code, rows in data.items():
        young = rows[:, 0] <= 7.0 + 1e-9
        excess = rows[young, 1] + _ejecta(rows, 10.0**6.0, "lower")[young] - M0
        if float(np.max(excess)) / M0 > worst:
            worst, where = float(np.max(excess)) / M0, code
    assert where == "z003"
    assert worst == pytest.approx(8.10e-3, abs=5e-5)


@pytest.mark.regression_bug
def test_remnant_column_is_incomplete_at_10_gyr_for_solar_z():
    """z014 at 10 Gyr: column 3 is 3.55e-2 M0, below the white-dwarf mass alone.

    A Chabrier population older than the turnoff (0.95 Msun) holds about 0.1 of its
    mass in white dwarfs: ((0.109 m + 0.428) / m weighted by the IMF mass over 0.95-8 Msun).
    The estimate below integrates the IMF by a plain loop on a log grid.
    """
    rows = _fixture()["z014"]
    k = int(np.argmin(abs(rows[:, 0] - 10.0)))
    col3 = rows[k, 2] / M0
    lm = np.linspace(np.log10(0.1), np.log10(300.0), 20001)
    m = 10.0**lm
    peak = np.exp(-((lm - np.log10(0.079)) ** 2) / (2 * 0.69**2))
    at_one = np.exp(-(np.log10(0.079) ** 2) / (2 * 0.69**2))
    xi = np.where(m < 1.0, peak, at_one * m**-1.3)
    dm = xi * m
    wd = (m >= 0.95) & (m < 8.0)
    wd_fraction = np.trapezoid(np.where(wd, dm * (0.109 * m + 0.428) / m, 0.0), lm) / np.trapezoid(
        dm, lm
    )
    assert col3 == pytest.approx(0.0355, abs=5e-4)
    assert wd_fraction > 0.09
    assert col3 < 0.5 * wd_fraction


# ---------------------------------------------------------------------------
# The grid converter reproduces the shipped grid's unit conversion.
# ---------------------------------------------------------------------------
@pytest.mark.limit
def test_converter_applies_lambda_squared_over_c_with_prospects_c():
    """L_nu = L_lambda * lambda^2 / 2.998e18 for one node, by hand."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "convert_bpass_to_h5", REPO / "scripts" / "convert_bpass_to_h5.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    wave = np.array([1000.0, 5000.0])
    l_lambda = np.full((1, 2, 2), 3.0)
    out = module.to_grid_arrays(wave, np.array([0.014]), np.array([1e6, 1e9]), l_lambda)
    expected = 3.0 * np.array([1000.0**2, 5000.0**2]) / 2.998e18
    assert out["ssp_flux"].shape == (1, 2, 2) and out["ssp_flux"].dtype == np.float32
    assert np.allclose(out["ssp_flux"][0, 0], expected, rtol=1e-6)
    assert np.allclose(out["ssp_lg_age_gyr"], [-3.0, 0.0], atol=1e-6)
    assert np.allclose(out["ssp_lgmet"], np.log10(0.014), atol=1e-6)


def _load_script(name: str):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # NamedTuple classes resolve their module by name
    spec.loader.exec_module(module)
    return module


@pytest.mark.conservation
def test_diagnostic_script_agrees_with_the_independent_loop_and_fails_the_gate():
    """scripts/check_bpass_mass_balance.py reproduces the plain-loop ejecta in all variants."""
    script = _load_script("check_bpass_mass_balance")
    tracks = script.tracks_from_fixture(FIXTURE)
    data = _fixture()
    starts = {"manual_0_to_6.05": 0.0, "6.0_to_6.05": 10.0**6.0}
    placement = {"incl": "upper", "excl": "lower", "mid": "mid"}
    for track in tracks:
        rows = data[track.code]
        for first_bin, start in starts.items():
            for rule, place in placement.items():
                got = script.ejected_mass(track, first_bin, rule)[:ROWS_TO_10P2]
                want = _ejecta(rows, start, place)[:ROWS_TO_10P2]
                assert np.allclose(got, want, rtol=1e-12, atol=1e-6)
    assert [t.code for t in tracks] == list(data)
    results = script.closure(tracks)
    assert min(c.pre_sn_max_abs.max() for c in results) == pytest.approx(1.0e-2, abs=2e-4)
    assert min(c.pre_sn_max_abs.max() for c in results) > script.GATE_TOL
