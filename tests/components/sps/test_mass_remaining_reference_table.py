# SPDX-License-Identifier: BSD-3-Clause
"""Contract: SSP mass-remaining tables (#2614).

The reference table (packaged for FSPS MIST + Chabrier) provides
metallicity-dependent surviving-mass fractions. Tests ensure:

* The packaged `.dat` file matches the tracked h5 file exactly;
* Pins from #2614 hold at matching Z nodes and ages;
* Z-dependent behavior is present (max - min over Z > 0.02);
* Matching grids use the table (Z-dependent), non-matching fall back to
  the sigmoid (Z-independent);
* File's own table is preferred over the reference.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

pytestmark = pytest.mark.contract


def _write_tiny_ssp(path: Path, age_nodes, met_nodes, **attrs) -> Path:
    """Write a minimal valid DSPS-format SSP file."""
    n_met = len(met_nodes)
    n_age = len(age_nodes)
    n_wave = 50
    with h5py.File(path, "w") as f:
        f["ssp_wave"] = np.linspace(1000.0, 20000.0, n_wave)
        f["ssp_flux"] = np.full((n_met, n_age, n_wave), 1e-4)
        f["ssp_lg_age_gyr"] = age_nodes
        f["ssp_lgmet"] = met_nodes
        for key, value in attrs.items():
            f.attrs[key] = value
    return path


def test_packaged_dat_matches_h5(tmp_path):
    """Packaged `.dat` == tracked `data/fsps_mass_remaining_chabrier.h5` exactly."""
    # Load from h5 file (repo root via parents)
    repo_root = Path(__file__).resolve().parents[3]
    h5_path = repo_root / "data" / "fsps_mass_remaining_chabrier.h5"
    assert h5_path.exists(), f"Tracked h5 file not found: {h5_path}"

    with h5py.File(h5_path, "r") as f:
        log_age_yr_h5 = f["log_age_yr"][:]
        z_absolute_h5 = f["z_absolute"][:]
        mass_remaining_h5 = f["mass_remaining"][:]

    # Load from packaged .dat via load_ssp_data mechanism
    from tengri.components.stellar.sps.dsps_wrapper import (
        _load_mass_remaining_reference,
    )

    log_age_yr_dat, lgmet_absolute_dat, table_dat = _load_mass_remaining_reference(
        ("mist", "chabrier")
    )

    # Compare arrays exactly
    assert np.array_equal(log_age_yr_h5, log_age_yr_dat)
    assert np.array_equal(np.log10(z_absolute_h5), lgmet_absolute_dat)
    assert np.array_equal(mass_remaining_h5, table_dat)


def test_pins_from_2614(tmp_path):
    """Table at Z☉ and ages 1e8/1e9/1e10 yr match pins from #2614."""
    from tengri.components.stellar.sps.dsps_wrapper import (
        _load_mass_remaining_reference,
    )

    log_age_yr, lgmet, table = _load_mass_remaining_reference(("mist", "chabrier"))

    # Find Z☉ (absolute Z = 0.0142) node, which is log10(0.0142) = -1.848
    z_sun = 0.0142
    lgmet_zsun = np.log10(z_sun)
    i_z_sun = np.argmin(np.abs(lgmet - lgmet_zsun))

    # Find age nodes: 1e8, 1e9, 1e10 yr = log age 8, 9, 10
    ages_target = [8, 9, 10]
    i_ages = [np.argmin(np.abs(log_age_yr - age)) for age in ages_target]

    # Expected pins from #2614
    pins = [0.7769, 0.6716, 0.5742]

    for i_age, age_log, pin in zip(i_ages, ages_target, pins):
        value = table[i_z_sun, i_age]
        assert np.abs(value - pin) < 5e-4, (
            f"Age 1e{age_log} yr: expected {pin}, got {value} (diff {abs(value - pin)})"
        )


def test_pins_min_max_at_1e10_yr(tmp_path):
    """Min and max over Z at 1e10 yr = 0.5544 / 0.5833 from #2614."""
    from tengri.components.stellar.sps.dsps_wrapper import (
        _load_mass_remaining_reference,
    )

    log_age_yr, _, table = _load_mass_remaining_reference(("mist", "chabrier"))

    # 1e10 yr = log age 10
    i_age_1e10 = np.argmin(np.abs(log_age_yr - 10))

    # Min and max over metallicity at that age
    min_val = np.min(table[:, i_age_1e10])
    max_val = np.max(table[:, i_age_1e10])

    assert np.abs(min_val - 0.5544) < 5e-4
    assert np.abs(max_val - 0.5833) < 5e-4


def test_matching_grid_uses_table(tmp_path):
    """A matching FSPS MIST chabrier grid loads with table (Z-dependent)."""
    from tengri.components.stellar.sps.dsps_wrapper import (
        _load_mass_remaining_reference,
    )

    # Get reference ladder
    log_age_yr, lgmet, table_ref = _load_mass_remaining_reference(("mist", "chabrier"))

    # Convert to Gyr for the grid
    lg_age_gyr = (log_age_yr - 9.0).astype(float)
    lgmet_grid = lgmet.astype(float)

    # Write a matching grid
    path = _write_tiny_ssp(
        tmp_path / "fsps_mist_miles_chabrier.h5",
        age_nodes=lg_age_gyr,
        met_nodes=lgmet_grid,
        imf="Chabrier (2003)",
        isochrone="mist",
        spectral_library="miles",
    )

    # Load and check
    ssp = load_ssp_data(str(path))
    assert ssp.ssp_mass_remaining is not None
    assert np.allclose(ssp.ssp_mass_remaining, table_ref, rtol=1e-6)

    # Verify Z-dependence: max - min over Z at oldest age > 0.02
    max_zsun_diff = np.max(ssp.ssp_mass_remaining, axis=0) - np.min(ssp.ssp_mass_remaining, axis=0)
    oldest_age_idx = -1
    assert max_zsun_diff[oldest_age_idx] > 0.02


def test_non_matching_falls_back_to_sigmoid(tmp_path):
    """All rows identical (sigmoid) when filename/IMF/nodes don't match."""
    # Test negative cases

    # (a) Wrong IMF (kroupa instead of chabrier)
    path_wrong_imf = _write_tiny_ssp(
        tmp_path / "fsps_mist_miles_kroupa.h5",
        age_nodes=np.linspace(-1, 1, 107),  # reference ladder
        met_nodes=np.log10(
            np.array(
                [0.0001, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
            )
        ),
        imf="Kroupa (2001)",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ssp_kroupa = load_ssp_data(str(path_wrong_imf))
    # All rows should be identical (Z-independent)
    assert np.allclose(ssp_kroupa.ssp_mass_remaining[0, :], ssp_kroupa.ssp_mass_remaining)

    # (b) Wrong isochrone (pgny instead of mist)
    path_wrong_iso = _write_tiny_ssp(
        tmp_path / "pgny_mist_c3k_chabrier.h5",  # pgny, not fsps/ssp prefix
        age_nodes=np.linspace(-1, 1, 107),
        met_nodes=np.log10(
            np.array(
                [0.0001, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
            )
        ),
        imf="Chabrier (2003)",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ssp_pgny = load_ssp_data(str(path_wrong_iso))
    # Sigmoid: all rows identical
    assert np.allclose(ssp_pgny.ssp_mass_remaining[0, :], ssp_pgny.ssp_mass_remaining)

    # (c) Wrong isochrone token in filename
    path_wrong_token = _write_tiny_ssp(
        tmp_path / "fsps_prsc_miles_chabrier.h5",  # prsc (Parsec), not mist
        age_nodes=np.linspace(-1, 1, 107),
        met_nodes=np.log10(
            np.array(
                [0.0001, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
            )
        ),
        imf="Chabrier (2003)",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ssp_prsc = load_ssp_data(str(path_wrong_token))
    # Sigmoid: all rows identical
    assert np.allclose(ssp_prsc.ssp_mass_remaining[0, :], ssp_prsc.ssp_mass_remaining)

    # (d) Age nodes shifted by +0.01 dex
    path_age_shift = _write_tiny_ssp(
        tmp_path / "fsps_mist_miles_chabrier_aged.h5",
        age_nodes=np.linspace(-1, 1, 107) + 0.01,  # shifted
        met_nodes=np.log10(
            np.array(
                [0.0001, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
            )
        ),
        imf="Chabrier (2003)",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ssp_aged = load_ssp_data(str(path_age_shift))
    # Sigmoid: all rows identical
    assert np.allclose(ssp_aged.ssp_mass_remaining[0, :], ssp_aged.ssp_mass_remaining)

    # (e) One metallicity node shifted by +0.01 dex
    met_shifted = np.log10(
        np.array([0.0001, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64])
    )
    met_shifted[5] += 0.01  # shift one node
    path_met_shift = _write_tiny_ssp(
        tmp_path / "fsps_mist_miles_chabrier_metz.h5",
        age_nodes=np.linspace(-1, 1, 107),
        met_nodes=met_shifted,
        imf="Chabrier (2003)",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ssp_metz = load_ssp_data(str(path_met_shift))
    # Sigmoid: all rows identical
    assert np.allclose(ssp_metz.ssp_mass_remaining[0, :], ssp_metz.ssp_mass_remaining)


def test_file_own_table_preferred(tmp_path):
    """File's own ssp_mass_remaining is preferred over reference."""
    from tengri.components.stellar.sps.dsps_wrapper import (
        _load_mass_remaining_reference,
    )

    # Get reference ladder
    log_age_yr, lgmet, _ = _load_mass_remaining_reference(("mist", "chabrier"))
    lg_age_gyr = (log_age_yr - 9.0).astype(float)
    lgmet_grid = lgmet.astype(float)

    # Write a grid with matching ladder but custom ssp_mass_remaining
    path = tmp_path / "fsps_mist_miles_chabrier.h5"
    n_met = len(lgmet_grid)
    n_age = len(lg_age_gyr)
    n_wave = 50
    with h5py.File(path, "w") as f:
        f["ssp_wave"] = np.linspace(1000.0, 20000.0, n_wave)
        f["ssp_flux"] = np.full((n_met, n_age, n_wave), 1e-4)
        f["ssp_lg_age_gyr"] = lg_age_gyr
        f["ssp_lgmet"] = lgmet_grid
        # Custom table (all 0.5)
        f["ssp_mass_remaining"] = np.full((n_met, n_age), 0.5)
        f.attrs["imf"] = "Chabrier (2003)"
        f.attrs["isochrone"] = "mist"
        f.attrs["spectral_library"] = "miles"

    ssp = load_ssp_data(str(path))
    # Should be 0.5 everywhere (the file's own table)
    assert np.allclose(ssp.ssp_mass_remaining, 0.5)
