"""Surviving-mass tables (``src/tengri/data/ssp_mass_remaining/*.h5``): bounds and pins.

Each table is the fraction of a single-age population's formed mass that is in living
stars plus remnants, taken from the SSP grid's own isochrones and IMF (#2751). This file
is table-only: it imports no loader.

BC03 table: the Bruzual & Charlot (2003) original release (Padova 1994 tracks, STELIB hr
library, Chabrier IMF), column (7) ``M*`` of ``bc2003_hr_m*_chab_ssp.4color``, the same
release as ``data/bc03_pdva_stelib_chabrier.h5``.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

TABLE_DIR = Path(__file__).resolve().parents[2] / "src" / "tengri" / "data" / "ssp_mass_remaining"
BC03 = TABLE_DIR / "mass_remaining_bc03pdva94_chabrier.h5"
H5_TABLES = [BC03]
FSPS_TABLES = sorted(p for p in TABLE_DIR.glob("mass_remaining_*.h5") if p != BC03)

# BC03's own M* is not strictly non-increasing: it rises by up to 0.0195 between adjacent
# nodes at log10 age 6.0-6.5 and by <= 7e-4 above 1 Gyr. Bound the rises, do not smooth.
MAX_RISE_YOUNG = 0.02
MAX_RISE_OLD = 1e-3


def _load(path: Path):
    with h5py.File(path, "r") as f:
        return (
            f["log10_age_yr"][:],
            f["log10_z_abs"][:],
            f["mass_remaining"][:],
            dict(f.attrs),
        )


@pytest.mark.bounds
@pytest.mark.parametrize("path", H5_TABLES, ids=lambda p: p.name)
def test_table_bounds(path):
    """BC03 table: shapes, increasing axes, values in (0, 1], 1 at the youngest age."""
    log_age, log_z, mass, attrs = _load(path)
    assert mass.shape == (log_z.shape[0], log_age.shape[0])
    assert mass.dtype == np.float64
    assert np.all(np.diff(log_age) > 0.0)
    assert np.all(np.diff(log_z) > 0.0)
    assert np.all(mass > 0.0) and np.all(mass <= 1.0)
    assert log_age[0] > 5.0 - 1e-9 and np.allclose(mass[:, 0], 1.0)
    rise = np.diff(mass, axis=1)
    old = log_age[1:] > 9.0
    assert rise[:, ~old].max() <= MAX_RISE_YOUNG
    assert rise[:, old].max() <= MAX_RISE_OLD
    for key in (
        "quantity",
        "isochrones",
        "imf",
        "source",
        "remnant_prescription",
        "citation",
        "generator",
        "generator_args",
        "input_sha256",
    ):
        assert key in attrs, key


@pytest.mark.bounds
def test_bc03_axes_are_the_grid_nodes():
    """Six Z nodes of the BC03 grid; 220 ages equal to the grid's nodes after t = 0."""
    log_age, log_z, mass, _ = _load(BC03)
    assert mass.shape == (6, 220)
    np.testing.assert_allclose(
        log_z, [-4.0, -3.39794, -2.39794, -2.09691, -1.6989701, -1.30103], atol=1e-5
    )
    assert log_age[0] == pytest.approx(5.100002) and log_age[-1] == pytest.approx(10.30103)


# Source values: column (7) M* of bc2003_hr_<m>_chab_ssp.4color inside
# bc03.models.padova_1994_chabrier_imf.tar.gz (sha256 d7b51afa...cd7b, see PROVENANCE.md).
# Row identified by its column (1) log age. The table stores exactly that decimal.
PINS = [
    # (z file key, log10 Z, log10 age [yr], M* from the 4color row)
    ("m22", -4.0, 8.006543, 0.71382),
    ("m62", -1.6989701, 9.006547, 0.60963),
    ("m72", -1.30103, 10.000000, 0.50813),
]


@pytest.mark.bounds
@pytest.mark.parametrize("key,log_z0,log_age0,expected", PINS, ids=[p[0] for p in PINS])
def test_bc03_values_equal_the_source_column(key, log_z0, log_age0, expected):
    """The table equals the 4color column (7) at three (Z, age) rows to 1e-10."""
    log_age, log_z, mass, _ = _load(BC03)
    iz = int(np.argmin(np.abs(log_z - log_z0)))
    ia = int(np.argmin(np.abs(log_age - log_age0)))
    assert abs(log_z[iz] - log_z0) < 1e-5
    assert abs(log_age[ia] - log_age0) < 1e-6
    assert abs(mass[iz, ia] - expected) < 1e-10


def _tau_exponential_fraction(log_age, m_z, t_obs_gyr, n):
    """Surviving fraction of a tau = 1 Gyr declining-exponential SFH [dimensionless]."""
    a = np.concatenate([[1e-12], np.linspace(0.0, t_obs_gyr, n + 1)[1:-1], [t_obs_gyr]])
    m_a = np.interp(np.log10(a * 1e9), log_age, m_z, left=1.0)
    w = np.exp(-(t_obs_gyr - a))
    return np.trapezoid(w * m_a, a) / np.trapezoid(w, a)


@pytest.mark.bounds
def test_bc03_2003_tau_1gyr_surviving_fraction_pin():
    """tau = 1 Gyr declining exponential, T = 4.8939 Gyr, Z = 0.02: BC03 2003 release.

    Pinned as the value of the BC03 *original 2003* release (stars + remnants, column 7
    of ``bc2003_hr_m62_chab_ssp.4color``) derived in this task: 0.545963, converged to
    < 1e-6 between 2000 and 200000 integration points. It is a cross-check of the table
    against the AGNfitter-rX template (0.5464) and is not a target. The 2016 update's
    equivalent column gives 0.592158 and its living-stars-only column 0.479432.
    """
    log_age, log_z, mass, _ = _load(BC03)
    iz = int(np.argmin(np.abs(log_z + 1.6989701)))
    coarse = _tau_exponential_fraction(log_age, mass[iz], 4.8939, 2000)
    fine = _tau_exponential_fraction(log_age, mass[iz], 4.8939, 200000)
    assert abs(coarse - fine) < 1e-6
    assert fine == pytest.approx(0.545963, abs=2e-6)


# FSPS-computed tables (python-fsps, or repackaged from a python-fsps grid). FSPS's own
# output is not bounded by 1 at the youngest nodes: measured maxima 1.0047 (Chabrier,
# Kroupa) and 1.0113 (Salpeter) near log10 age 6.4-6.7, and the youngest node (1e5 yr) is
# 0.983-1.011. Those values are what FSPS returns and are repackaged unaltered; the loader
# is agnostic to them and never clips.
OLD_BOUNDS = {"chabrier": (0.4, 0.8), "kroupa": (0.4, 0.8), "salpeter": (0.6, 0.9)}


@pytest.mark.bounds
@pytest.mark.parametrize("path", FSPS_TABLES, ids=lambda p: p.name)
def test_fsps_table_bounds(path):
    """FSPS tables: axes, non-increasing above 1 Gyr, 13 Gyr window, attributes.

    The 13 Gyr window (0.4-0.8 Chabrier/Kroupa, 0.6-0.9 Salpeter) brackets the measured
    0.54-0.59 / 0.72-0.75 as a sanity bound, not a calibration.
    """
    log_age, log_z, mass, attrs = _load(path)
    assert mass.shape == (log_z.shape[0], log_age.shape[0]) and mass.dtype == np.float64
    assert np.all(np.diff(log_age) > 0.0) and np.all(np.diff(log_z) > 0.0)
    rise = np.diff(mass, axis=1)
    old = log_age[1:] > 9.0
    assert rise[:, old].max() <= 0.0
    lo, hi = OLD_BOUNDS[attrs["imf"]]
    i13 = int(np.argmin(np.abs(log_age - 10.11)))
    assert np.all(mass[:, i13] >= lo) and np.all(mass[:, i13] <= hi)
    for key in (
        "quantity",
        "isochrones",
        "imf",
        "source",
        "remnant_prescription",
        "citation",
        "generator",
        "generator_args",
    ):
        assert key in attrs, key
    assert "source_commit" in attrs or "input_sha256" in attrs


@pytest.mark.bounds
@pytest.mark.parametrize(
    "path",
    [
        pytest.param(
            p,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "FSPS stellar_mass overshoots 1 at young ages (MIST: 1.0113 Salpeter, "
                    "1.0047 Chabrier near log10 age 6.35; BaSTI: 1.085 Salpeter, 1.018 "
                    "Chabrier at log10 age 5.5, Z = 0.008). Accepted as FSPS's convention "
                    "and recorded in PROVENANCE.md. Flips to XPASS (a failure) if the "
                    "overshoot is removed."
                ),
            ),
        )
        if any(tag in p.name for tag in ("mist", "bsti"))
        else p
        for p in FSPS_TABLES
    ],
    ids=lambda p: p.name,
)
def test_fsps_table_is_a_fraction_of_formed_mass(path):
    """Surviving mass is a fraction of the formed mass: values in (0, 1]."""
    _, _, mass, _ = _load(path)
    assert np.all(mass > 0.0) and np.all(mass <= 1.0)
