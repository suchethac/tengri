# SPDX-License-Identifier: BSD-3-Clause
"""C3K v2.3 library resolution tables (FSPS 05b5e55) are wired to their grids.

The c3k_hr and c3k_lr tables are FSPS's own per-node velocity dispersion for the
C3K v2.3 high- and low-resolution libraries. A grid gets its table through the
filename token, and only when its ``ssp_wave`` is that library's reference grid
node for node.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tengri.components.stellar.sps.dsps_wrapper import (
    _LIBRARY_RESOLUTION_DATA_FILES,
    _detect_library_resolution_key,
    _load_library_resolution_reference,
    _resolve_ssp_resolution,
    load_ssp_data,
)

pytestmark = pytest.mark.contract


def _write_grid(tmp_path, stem: str, key: str, *, scale_wave: float = 1.0) -> str:
    """A small synthetic SSP file named like a hosted grid, on the library's own wavelengths.

    ``ssp_wave`` is the packaged reference table's wavelength column (times ``scale_wave``,
    1.0 for a matching grid), so the loader's filename detection and its wavelength-match
    guard run exactly as on the real file, without the multi-GB grid.
    """
    import h5py

    wave, _ = _load_library_resolution_reference(key)
    wave = np.asarray(wave, dtype=float) * scale_wave
    n_met, n_age = 2, 3
    path = tmp_path / f"{stem}.h5"
    with h5py.File(path, "w") as f:
        f["ssp_wave"] = wave
        f["ssp_flux"] = np.full((n_met, n_age, wave.size), 1e-4)
        f["ssp_lg_age_gyr"] = np.linspace(-2.0, 1.0, n_age)
        f["ssp_lgmet"] = np.array([-2.5, -1.8])
    return str(path)


def _load(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_ssp_data(path, mass_remaining="dsps_fit")


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


class TestHrDefaultGrid:
    """A grid named like the coming default C3K v2.3 file, on the c3k_hr wavelengths."""

    STEM = "ssp_mist_c3k_hr_chabrier_wNE_logGasU-3.0_logGasZ0.0"

    def test_gets_resolution_at_5000(self, tmp_path):
        ssp = _load(_write_grid(tmp_path, self.STEM, "c3k_hr"))
        assert ssp.ssp_resolution_kms is not None
        assert _sigma_at(ssp, 5000.0) == pytest.approx(42.4378, rel=1e-3)

    def test_approximate_mask_outside_1001_25000(self, tmp_path):
        ssp = _load(_write_grid(tmp_path, self.STEM, "c3k_hr"))
        wave = np.asarray(ssp.ssp_wave)
        _, approximate = _resolve_ssp_resolution(ssp.ssp_wave, "c3k_hr")
        assert np.all(approximate[wave < 1000.0])
        assert np.all(approximate[wave > 25251.0])
        assert not np.any(approximate[(wave > 1001.0) & (wave < 25000.0)])
        assert _sigma_at(ssp, 500.0) == pytest.approx(636.5667, rel=1e-3)

    def test_mismatched_wavelengths_get_no_table(self, tmp_path):
        ssp = _load(_write_grid(tmp_path, self.STEM, "c3k_hr", scale_wave=1.001))
        assert ssp.ssp_resolution_kms is None


class TestLrGrid:
    """c3k_lr grids, including the alpha-MC lr files, get the lr table: R = 100."""

    @pytest.mark.parametrize(
        "stem", ["ssp_mist_c3k_lr_chabrier_noNE", "fsps_amist_c3k_lr_afe_chabrier_totalz"]
    )
    def test_gets_lr_resolution(self, tmp_path, stem):
        ssp = _load(_write_grid(tmp_path, stem, "c3k_lr"))
        assert ssp.ssp_resolution_kms is not None
        assert _sigma_at(ssp, 5000.0) == pytest.approx(1273.133, rel=1e-3)


class TestLegacyC3kA:
    """The c3k_a token still maps to the c3k_afe0.0 table."""

    def test_c3k_a_grid_keeps_its_table(self, tmp_path):
        ssp = _load(_write_grid(tmp_path, "fsps_mist_c3k_a_chabrier", "c3k_a"))
        assert ssp.ssp_resolution_kms is not None
        ref_wave, ref_sigma = _load_library_resolution_reference("c3k_a")
        assert ref_wave.shape[0] == 11149
        assert _sigma_at(ssp, 5000.0) == pytest.approx(
            abs(ref_sigma[np.argmin(np.abs(ref_wave - 5000.0))]), rel=1e-3
        )
