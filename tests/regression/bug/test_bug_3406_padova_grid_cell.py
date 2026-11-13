"""Regression for #3406: the Padova SSP grids carry a single-cell spike.

Cell lgmet = -3.301 (Z = 0.0005) at age 1.000 Gyr is ~2.8x brighter at 6200 A
than its metallicity and age neighbours in every bundled Padova grid. The
loader repairs that cell (see ``_KNOWN_GRID_DEFECTS`` in ``dsps_wrapper``), so
the loaded grid must be smooth there and a constant-SFH colour sweep must be
monotone in metallicity.
"""

import numpy as np
import pytest

from tengri._data_setup import find_ssp_files
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.components.stellar.sps.grid_defects import repair_known_grid_defects
from tengri.utils.physics_constants import LOG10_ZSUN

SPIKE_LGMET = -3.301
SPIKE_LG_AGE_GYR = 0.0
REF_WAVE_AA = 6200.0
MAX_DEV = 0.10
SFH_AGE_GYR = 10.0


def _grid_path(stem):
    for path in find_ssp_files():
        if path.stem == stem:
            return path
    pytest.skip(f"{stem}.h5 is not in the data directory")


def _cell_index(ssp, lgmet, lg_age_gyr):
    k = int(np.argmin(np.abs(np.asarray(ssp.ssp_lgmet) - lgmet)))
    i = int(np.argmin(np.abs(np.asarray(ssp.ssp_lg_age_gyr) - lg_age_gyr)))
    return k, i


def _band_luminosity(lnu, wave, transmission_path):
    """AB-style band luminosity of ``lnu`` through the filter in ``transmission_path``."""
    fw, ft = np.loadtxt(transmission_path, unpack=True)
    trans = np.interp(wave, fw, ft, left=0.0, right=0.0)
    weight = trans / wave
    return np.trapezoid(lnu * weight, wave) / np.trapezoid(weight, wave)


def _const_sfh_gr(ssp, lgmet, filter_dir):
    """g - r of a constant SFH of 10 Gyr at ``lgmet``, metallicity linear in the grid."""
    lg_age = np.asarray(ssp.ssp_lg_age_gyr, dtype=np.float64)
    wave = np.asarray(ssp.ssp_wave, dtype=np.float64)
    lgmet_grid = np.asarray(ssp.ssp_lgmet, dtype=np.float64)
    flux = np.asarray(ssp.ssp_flux, dtype=np.float64)

    k = int(np.clip(np.searchsorted(lgmet_grid, lgmet) - 1, 0, len(lgmet_grid) - 2))
    frac = (lgmet - lgmet_grid[k]) / (lgmet_grid[k + 1] - lgmet_grid[k])
    flux_at_met = (1.0 - frac) * flux[k] + frac * flux[k + 1]

    age_yr = 10.0**lg_age * 1e9
    keep = age_yr <= SFH_AGE_GYR * 1e9
    weight = np.gradient(age_yr)[keep]
    lnu = weight @ flux_at_met[keep]

    g = _band_luminosity(lnu, wave, filter_dir / "SLOAN_SDSS_g.dat")
    r = _band_luminosity(lnu, wave, filter_dir / "SLOAN_SDSS_r.dat")
    return -2.5 * np.log10(g / r)


@pytest.mark.parametrize("stem", ["fsps_prsc_miles_chabrier"])
def test_repaired_spike_cell_matches_its_neighbours(stem):
    path = _grid_path(stem)
    ssp = load_ssp_data(str(path))
    flux = np.asarray(ssp.ssp_flux)
    wave = np.asarray(ssp.ssp_wave)
    k, i = _cell_index(ssp, SPIKE_LGMET, SPIKE_LG_AGE_GYR)
    j = int(np.argmin(np.abs(wave - REF_WAVE_AA)))

    cell = flux[k, i, j]
    vs_age = cell / np.sqrt(flux[k, i - 1, j] * flux[k, i + 1, j])
    vs_met = cell / np.sqrt(flux[k - 1, i, j] * flux[k + 1, i, j])

    assert abs(vs_age - 1.0) < MAX_DEV, f"age neighbours ratio {vs_age:.3f}"
    assert abs(vs_met - 1.0) < MAX_DEV, f"metallicity neighbours ratio {vs_met:.3f}"
    assert len(ssp.repaired_cells) == 1
    assert ssp.repaired_cells[0][0] == pytest.approx(SPIKE_LGMET, abs=1e-3)
    assert ssp.repaired_cells[0][1] == pytest.approx(SPIKE_LG_AGE_GYR, abs=1e-3)


def test_repair_is_geometric_mean_of_age_neighbours_at_every_wavelength():
    lgmet = np.array([-3.30103, -3.0])
    lg_age = np.array([-0.05, 0.0, 0.05])
    flux = np.ones((2, 3, 4))
    flux[0, 0] = [1.0, 2.0, 4.0, 8.0]
    flux[0, 2] = [4.0, 8.0, 16.0, 32.0]
    flux[0, 1] = [99.0, 99.0, 99.0, 99.0]
    flux[1, 1] = [5.0, 5.0, 5.0, 5.0]

    repaired, cells = repair_known_grid_defects("fsps_prsc_miles_chabrier", flux, lgmet, lg_age)

    np.testing.assert_allclose(repaired[0, 1], [2.0, 4.0, 8.0, 16.0])
    np.testing.assert_allclose(repaired[1], flux[1])
    assert cells == ((-3.30103, 0.0),)
    np.testing.assert_allclose(flux[0, 1], [99.0] * 4)


def test_grid_without_recorded_defect_is_returned_unchanged():
    flux = np.arange(24, dtype=float).reshape(2, 3, 4) + 1.0
    lgmet = np.array([-3.30103, -3.0])
    lg_age = np.array([-0.05, 0.0, 0.05])

    repaired, cells = repair_known_grid_defects("fsps_mist_miles_chabrier", flux, lgmet, lg_age)

    assert cells == ()
    assert repaired is flux


@pytest.mark.parametrize("stem", ["fsps_prsc_miles_chabrier"])
def test_const_sfh_gr_is_monotone_in_metallicity(stem):
    # The sweep stops at -0.5 dex: above about -0.25 dex the 10 Gyr g - r of the
    # upstream grid turns over (MIST node values show the same), which is not the
    # spike this test guards against.
    path = _grid_path(stem)
    ssp = load_ssp_data(str(path))
    logzsol = np.round(np.arange(-2.0, -0.5 + 1e-9, 0.1), 6)
    g_r = np.array(
        [_const_sfh_gr(ssp, lz + LOG10_ZSUN, path.parent / "filters") for lz in logzsol]
    )

    assert np.all(np.isfinite(g_r))
    assert np.all(np.diff(g_r) >= 0.0), f"non-monotone g-r sweep: {np.round(g_r, 4).tolist()}"
