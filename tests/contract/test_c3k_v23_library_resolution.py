# SPDX-License-Identifier: BSD-3-Clause
"""C3K v2.3 library resolution tables (FSPS 05b5e55) are wired to their grids.

The c3k_hr and c3k_lr tables are FSPS's own per-node velocity dispersion for the
C3K v2.3 high- and low-resolution libraries. A grid gets its table through the
filename token, and only when its ``ssp_wave`` is that library's reference grid
node for node.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tengri._data_setup import find_data_str
from tengri.components.stellar.sps.dsps_wrapper import (
    _LIBRARY_RESOLUTION_DATA_FILES,
    _detect_library_resolution_key,
    _load_library_resolution_reference,
    _resolve_ssp_resolution,
    load_ssp_data,
)

pytestmark = pytest.mark.contract

#: Grids are located the way the loader locates them (``$TENGRI_DATA_DIR`` and the
#: ancestors of the working directory), so a git worktree finds the parent checkout's data.
HR_DEFAULT_GRID = find_data_str("ssp_mist_c3k_hr_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
LR_GRID = find_data_str("ssp_mist_c3k_lr_chabrier_noNE.h5")
C3K_A_GRID = find_data_str("fsps_mist_c3k_a_chabrier.h5")
#: Alpha-MC lr grid on the total-Z axis (the fixed-[Fe/H] file is refused by the loader).
_ALPHA_MC_LR_PATH = Path(
    "/Users/suchethacooray/Projects/tengri-alpha-grids/formatted/totalz/"
    "fsps_amist_c3k_lr_afe_chabrier_totalz.h5"
)
ALPHA_MC_LR_GRID = str(_ALPHA_MC_LR_PATH) if _ALPHA_MC_LR_PATH.is_file() else None

requires_hr_grid = pytest.mark.skipif(
    HR_DEFAULT_GRID is None, reason="C3K v2.3 hr default grid not found on this machine"
)
requires_lr_grid = pytest.mark.skipif(LR_GRID is None, reason="C3K v2.3 lr grid not found")
requires_c3k_a_grid = pytest.mark.skipif(C3K_A_GRID is None, reason="C3K a grid not found")
requires_alpha_mc_lr = pytest.mark.skipif(
    ALPHA_MC_LR_GRID is None, reason="alpha-MC lr total-Z grid not found"
)


def _sigma_at(ssp, wave_aa: float) -> float:
    """Resolution [km/s] of a loaded grid at the node nearest ``wave_aa``."""
    wave = np.asarray(ssp.ssp_wave)
    idx = int(np.argmin(np.abs(wave - wave_aa)))
    return float(np.asarray(ssp.ssp_resolution_kms)[idx])


class TestKeyDetection:
    """Filename tokens map to tables without cross-matching c3k_a, c3k_hr, c3k_lr."""

    @pytest.mark.parametrize(
        ("stem", "key"),
        [
            ("ssp_mist_c3k_hr_chabrier_wNE_logGasU-3.0_logGasZ0.0", "c3k_hr"),
            ("ssp_mist_c3k_hr_chabrier_noNE", "c3k_hr"),
            ("amist_c3k_hr_kroupa_afe", "c3k_hr"),
            ("ssp_mist_c3k_lr_chabrier_noNE", "c3k_lr"),
            ("fsps_amist_c3k_lr_afe_chabrier", "c3k_lr"),
            ("fsps_amist_c3k_lr_afe_salpeter", "c3k_lr"),
            ("ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0", "c3k_a"),
            ("fsps_mist_c3k_a_kroupa", "c3k_a"),
            ("bpss_stars_c3k_a_chabrier", "c3k_a"),
            ("miles_bsti_kroupa_afe", "miles"),
        ],
    )
    def test_filename_maps_to_key(self, stem, key):
        assert _detect_library_resolution_key(stem) == key

    @pytest.mark.parametrize(
        "stem",
        [
            "pgny_mist_c3k_chabrier",  # bare c3k token, no hr/lr/a suffix
            "fsps_amist_c3k_afe_chabrier",  # afe is not a library token
            "ssp_prsc_bc03_chabrier",
        ],
    )
    def test_unrelated_names_have_no_key(self, stem):
        assert _detect_library_resolution_key(stem) is None

    def test_c3k_a_not_confused_with_hr_or_lr(self):
        assert _detect_library_resolution_key("x_c3k_a_y") == "c3k_a"
        assert _detect_library_resolution_key("x_c3k_hr_y") == "c3k_hr"
        assert _detect_library_resolution_key("x_c3k_lr_y") == "c3k_lr"

    def test_all_keys_have_shipped_tables(self):
        for key in ("c3k_hr", "c3k_lr", "c3k_a", "miles"):
            assert key in _LIBRARY_RESOLUTION_DATA_FILES
            wave, sigma = _load_library_resolution_reference(key)
            assert wave.shape == sigma.shape
            assert np.all(np.diff(wave) > 0)


class TestHrTable:
    """The c3k_hr table carries FSPS's values, read from the packaged file."""

    def test_r3000_core_value(self):
        wave, sigma = _load_library_resolution_reference("c3k_hr")
        core = (wave >= 3001.0) & (wave < 9990.0)
        assert core.any()
        np.testing.assert_allclose(np.abs(sigma[core]), 42.4378, rtol=1e-3)
        assert np.all(sigma[core] > 0)

    def test_approximate_mask_outside_measured_window(self):
        wave, sigma = _load_library_resolution_reference("c3k_hr")
        measured = (wave >= 1001.0) & (wave <= 25000.0)
        assert np.all(sigma[measured] > 0)
        assert np.all(sigma[~measured] < 0)


@requires_hr_grid
class TestHrDefaultGrid:
    """The coming default C3K v2.3 grid receives the hr table."""

    def test_gets_resolution_at_5000(self):
        ssp = load_ssp_data(HR_DEFAULT_GRID)
        assert ssp.ssp_resolution_kms is not None
        assert _sigma_at(ssp, 5000.0) == pytest.approx(42.4378, rel=1e-3)

    def test_approximate_mask_outside_1001_25000(self):
        ssp = load_ssp_data(HR_DEFAULT_GRID)
        wave = np.asarray(ssp.ssp_wave)
        _, approximate = _resolve_ssp_resolution(ssp.ssp_wave, "c3k_hr")
        assert np.all(approximate[wave < 1000.0])
        assert np.all(approximate[wave > 25251.0])
        assert not np.any(approximate[(wave > 1001.0) & (wave < 25000.0)])
        assert _sigma_at(ssp, 500.0) == pytest.approx(636.5667, rel=1e-3)


@requires_lr_grid
class TestLrGrid:
    """A c3k_lr grid gets the lr table: R = 100 over the measured window."""

    def test_gets_lr_resolution(self):
        ssp = load_ssp_data(LR_GRID)
        assert ssp.ssp_resolution_kms is not None
        assert _sigma_at(ssp, 5000.0) == pytest.approx(1273.133, rel=1e-3)


@requires_alpha_mc_lr
class TestAlphaMcLrGrid:
    """The alpha-MC lr grid shares the c3k_lr wavelength grid and gets its table."""

    def test_wave_match_guard_passes_and_table_applies(self):
        ssp = load_ssp_data(ALPHA_MC_LR_GRID)
        assert ssp.ssp_resolution_kms is not None
        assert _sigma_at(ssp, 5000.0) == pytest.approx(1273.133, rel=1e-3)


@requires_c3k_a_grid
class TestLegacyC3kA:
    """The c3k_a token still maps to the c3k_afe0.0 table."""

    def test_c3k_a_grid_keeps_its_table(self):
        ssp = load_ssp_data(C3K_A_GRID)
        assert ssp.ssp_resolution_kms is not None
        ref_wave, ref_sigma = _load_library_resolution_reference("c3k_a")
        assert ref_wave.shape[0] == 11149
        assert _sigma_at(ssp, 5000.0) == pytest.approx(
            abs(ref_sigma[np.argmin(np.abs(ref_wave - 5000.0))]), rel=1e-3
        )
