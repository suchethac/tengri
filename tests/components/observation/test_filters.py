# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for photometric filter management (SVO FPS integration).

Frozen: filter loading (SingleQuote returns FilterCurve), effective wavelength
ordering (g < r < i), transmission bounds, ALMA band definitions and
wavelength ordering, filter integration with SED photometry.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.filters import (
    _C_AA_S,
    FILTER_REGISTRY,
    download_filter,
    load_alma_band,
    load_custom_filter,
    load_filter,
    load_filter_set,
    load_tophat_filter,
)

pytestmark = pytest.mark.bounds
from tengri.observation.photometry import FilterCurve

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cache_dir():
    """Use the project-level cache so tests don't re-download."""
    return "data/filters"


@pytest.fixture(scope="module")
def sample_filter(cache_dir):
    """Load a single well-known filter (SDSS r)."""
    return load_filter("sdss_r", cache_dir=cache_dir)


@pytest.fixture
def custom_filter_path(tmp_path):
    """Write a synthetic two-column filter file."""
    wave = np.linspace(5000.0, 7000.0, 50)
    trans = np.exp(-0.5 * ((wave - 6000.0) / 300.0) ** 2)
    filepath = tmp_path / "custom_test.dat"
    np.savetxt(
        str(filepath),
        np.column_stack([wave, trans]),
        header="wavelength  transmission",
        fmt="%.6e",
    )
    return str(filepath)


# ── Tests ─────────────────────────────────────────────────────────


class TestLoadFilter:
    """load_filter: correct shape, monotonicity, non-negativity."""

    def test_correct_shape(self, sample_filter):
        """Wave and trans are 1-D arrays of same length, >10 points."""
        assert sample_filter.wave.ndim == 1
        assert sample_filter.trans.ndim == 1
        assert len(sample_filter.wave) == len(sample_filter.trans)
        assert len(sample_filter.wave) > 10

    def test_transmission_nonnegative_and_finite(self, sample_filter):
        """Transmission values are non-negative and finite."""
        assert float(jnp.min(sample_filter.trans)) >= 0.0
        assert jnp.all(jnp.isfinite(sample_filter.trans))

    def test_wavelength_monotonically_increasing(self, sample_filter):
        """Wavelengths strictly increasing (required for interpolation)."""
        diffs = jnp.diff(sample_filter.wave)
        assert float(jnp.min(diffs)) > 0.0

    def test_name_preserved(self, sample_filter):
        """Name attribute matches requested filter key."""
        assert sample_filter.name == "sdss_r"

    def test_unknown_filter_raises(self, cache_dir):
        """Unknown filter key raises KeyError."""
        with pytest.raises(KeyError, match="Unknown filter"):
            load_filter("nonexistent_filter_xyz", cache_dir=cache_dir)


class TestLoadFilterSet:
    """load_filter_set: correct count, wavelength ordering."""

    def test_returns_correct_count(self, cache_dir):
        """Returned lists have matching length."""
        names = ["sdss_g", "sdss_r", "sdss_i"]
        waves, trans, curves = load_filter_set(names, cache_dir=cache_dir)
        assert len(waves) == 3
        assert len(trans) == 3
        assert len(curves) == 3

    def test_effective_wavelength_ordering(self, cache_dir):
        """Effective wavelengths increase g < r < i (physics anchor)."""
        names = ["sdss_g", "sdss_r", "sdss_i"]
        waves, trans, _ = load_filter_set(names, cache_dir=cache_dir)
        eff_waves = []
        for w, t in zip(waves, trans):
            eff = float(jnp.sum(w * t) / jnp.sum(t))
            eff_waves.append(eff)
        assert eff_waves[0] < eff_waves[1] < eff_waves[2]


class TestLoadCustomFilter:
    """load_custom_filter: shape, monotonicity, finiteness."""

    def test_loads_correctly(self, custom_filter_path):
        """Custom filter file loads with correct point count."""
        fc = load_custom_filter(custom_filter_path)
        assert isinstance(fc, FilterCurve)
        assert len(fc.wave) == 50

    def test_transmission_nonnegative_and_finite(self, custom_filter_path):
        """Transmission values are non-negative and finite."""
        fc = load_custom_filter(custom_filter_path)
        assert float(jnp.min(fc.trans)) >= 0.0
        assert jnp.all(jnp.isfinite(fc.trans))

    def test_wavelength_monotonically_increasing(self, custom_filter_path):
        """Wavelengths strictly increasing."""
        fc = load_custom_filter(custom_filter_path)
        diffs = jnp.diff(fc.wave)
        assert float(jnp.min(diffs)) > 0.0

    def test_missing_file_raises(self):
        """Missing file path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_custom_filter("/nonexistent/path/filter.dat")


class TestDownloadFilter:
    """download_filter: caching and consistency."""

    def test_caches_to_disk(self, cache_dir):
        """Download creates cached file on disk."""
        svo_id = FILTER_REGISTRY["sdss_r"]
        wave, trans = download_filter(svo_id, cache_dir=cache_dir)
        expected_file = Path(cache_dir) / "SLOAN_SDSS_r.dat"
        assert expected_file.exists()
        assert len(wave) == len(trans)
        assert len(wave) > 0

    def test_loads_from_cache_idempotent(self, cache_dir):
        """Second call loads from disk (consistent result)."""
        svo_id = FILTER_REGISTRY["sdss_r"]
        wave1, trans1 = download_filter(svo_id, cache_dir=cache_dir)
        wave2, trans2 = download_filter(svo_id, cache_dir=cache_dir)
        np.testing.assert_array_equal(wave1, wave2)
        np.testing.assert_array_equal(trans1, trans2)


class TestListAvailableFilters:
    """list_available_filters and registry: not empty."""

    def test_registry_not_empty(self):
        """FILTER_REGISTRY contains >50 filters."""
        assert len(FILTER_REGISTRY) > 50


class TestLoadTophatFilter:
    """load_tophat_filter: center wavelength, width, uniform transmission."""

    def test_center_wavelength(self):
        """Top-hat filter center is at specified wavelength."""
        center = 1.23e7
        width = 5e5
        fc = load_tophat_filter(center, width)
        midpoint = float((fc.wave[0] + fc.wave[-1]) / 2)
        assert midpoint == pytest.approx(center, rel=1e-6)

    def test_width_correct(self):
        """Top-hat filter width equals specified width."""
        center = 1.23e7
        width = 5e5
        fc = load_tophat_filter(center, width)
        actual_width = float(fc.wave[-1] - fc.wave[0])
        assert actual_width == pytest.approx(width, rel=1e-6)

    def test_uniform_transmission(self):
        """Top-hat filter transmission is 1.0 everywhere."""
        fc = load_tophat_filter(1e7, 1e6)
        assert float(jnp.min(fc.trans)) == pytest.approx(1.0)
        assert float(jnp.max(fc.trans)) == pytest.approx(1.0)


class TestLoadAlmaBand:
    """ALMA band filters: definitions, wavelength ordering, monotonicity."""

    def test_all_bands_load(self):
        """Bands 1-10 all load successfully."""
        for b in range(1, 11):
            fc = load_alma_band(b)
            assert isinstance(fc, FilterCurve)

    def test_invalid_band(self):
        """Band 0 and 11 raise ValueError."""
        with pytest.raises(ValueError, match="ALMA band"):
            load_alma_band(0)
        with pytest.raises(ValueError, match="ALMA band"):
            load_alma_band(11)

    def test_band6_center_wavelength(self):
        """Band 6 (211–275 GHz) center ≈ 1.23 mm (frozen from definition)."""
        fc = load_alma_band(6)
        lo_aa = _C_AA_S / (275.0e9)
        hi_aa = _C_AA_S / (211.0e9)
        expected_center = (lo_aa + hi_aa) / 2.0
        actual_center = float((fc.wave[0] + fc.wave[-1]) / 2)
        assert actual_center == pytest.approx(expected_center, rel=1e-4)

    def test_band7_center_wavelength(self):
        """Band 7 (275–373 GHz) center ≈ 0.93 mm (frozen from definition)."""
        fc = load_alma_band(7)
        lo_aa = _C_AA_S / (373.0e9)
        hi_aa = _C_AA_S / (275.0e9)
        expected_center = (lo_aa + hi_aa) / 2.0
        actual_center = float((fc.wave[0] + fc.wave[-1]) / 2)
        assert actual_center == pytest.approx(expected_center, rel=1e-4)

    def test_wavelengths_monotonically_increasing(self):
        """ALMA band wavelengths strictly increasing."""
        fc = load_alma_band(6)
        diffs = jnp.diff(fc.wave)
        assert float(jnp.min(diffs)) > 0.0

    def test_band_wavelength_ordering(self):
        """Higher band number = higher frequency = shorter wavelength."""
        centers = []
        for b in range(3, 11):
            fc = load_alma_band(b)
            centers.append(float((fc.wave[0] + fc.wave[-1]) / 2))
        # Wavelength decreases as band number increases
        for i in range(len(centers) - 1):
            assert centers[i] > centers[i + 1]

    def test_band_definitions_in_expected_range(self):
        """Band 6 center ~1.23 mm, Band 9 center ~0.45 mm (limit check)."""
        fc6 = load_alma_band(6)
        c6_mm = float((fc6.wave[0] + fc6.wave[-1]) / 2) / 1e7
        assert 1.0 < c6_mm < 1.5, f"Band 6 center should be ~1.23 mm, got {c6_mm:.3f}"

        fc9 = load_alma_band(9)
        c9_mm = float((fc9.wave[0] + fc9.wave[-1]) / 2) / 1e7
        assert 0.35 < c9_mm < 0.60, f"Band 9 center should be ~0.45 mm, got {c9_mm:.3f}"


class TestPhotometryIntegration:
    """FilterCurve integration with SED: flux conservation and effective wavelengths."""

    def test_tophat_integration_finite_and_positive(self):
        """Top-hat filter over flat SED produces finite, positive flux."""
        fc = load_tophat_filter(5500.0, 500.0, name="optical_window")
        wave = fc.wave
        sed = jnp.ones_like(wave)
        numerator = jnp.trapezoid(sed * fc.trans * wave, wave)
        denominator = jnp.trapezoid(fc.trans * wave, wave)
        flux = numerator / denominator
        assert jnp.isfinite(flux)
        assert float(flux) > 0.0

    def test_alma_band_integration_finite_and_positive(self):
        """ALMA band filter over power-law SED produces finite, positive flux."""
        fc = load_alma_band(6)
        wave = fc.wave
        sed = wave ** (-1.3)
        numerator = jnp.trapezoid(sed * fc.trans * wave, wave)
        denominator = jnp.trapezoid(fc.trans * wave, wave)
        flux = numerator / denominator
        assert jnp.isfinite(flux)
        assert float(flux) > 0.0

    def test_custom_filter_effective_wavelength(self, tmp_path):
        """Gaussian filter effective wavelength ≈ center wavelength."""
        wave_np = np.linspace(5000.0, 7000.0, 100)
        trans_np = np.exp(-0.5 * ((wave_np - 6000.0) / 300.0) ** 2)
        filepath = tmp_path / "gauss_filter.dat"
        np.savetxt(str(filepath), np.column_stack([wave_np, trans_np]), fmt="%.6e")

        fc = load_custom_filter(str(filepath))
        wave = fc.wave
        # Effective wavelength: λ_eff = ∫T(λ)λ dλ / ∫T(λ) dλ
        numerator = jnp.trapezoid(fc.trans * wave, wave)
        denominator = jnp.trapezoid(fc.trans, wave)
        eff_wave = float(numerator / denominator)
        # Gaussian at 6000 Å should have eff_wave ≈ 6000 Å
        assert abs(eff_wave - 6000.0) < 20.0
