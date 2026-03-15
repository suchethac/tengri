"""Tests for photometric filter management (SVO FPS integration)."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from diffsed.models.observation.filters import (
    FILTER_REGISTRY,
    download_filter,
    list_available_filters,
    load_custom_filter,
    load_filter,
    load_filter_set,
)
from diffsed.models.observation.photometry import FilterCurve

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadFilter:
    """Tests for load_filter (single filter by short name)."""

    def test_returns_filtercurve(self, sample_filter):
        assert isinstance(sample_filter, FilterCurve)

    def test_correct_shape(self, sample_filter):
        """wave and trans must be 1-D arrays of the same length."""
        assert sample_filter.wave.ndim == 1
        assert sample_filter.trans.ndim == 1
        assert len(sample_filter.wave) == len(sample_filter.trans)
        assert len(sample_filter.wave) > 10  # reasonable number of points

    def test_transmission_range(self, sample_filter):
        """Transmission must be in [0, 1] after normalization."""
        assert float(jnp.min(sample_filter.trans)) >= 0.0
        assert float(jnp.max(sample_filter.trans)) <= 1.0 + 1e-7
        # Max should be ~1.0 after normalization
        assert float(jnp.max(sample_filter.trans)) > 0.99

    def test_wavelength_monotonic(self, sample_filter):
        """Wavelengths must be strictly increasing."""
        diffs = jnp.diff(sample_filter.wave)
        assert float(jnp.min(diffs)) > 0.0

    def test_name_preserved(self, sample_filter):
        assert sample_filter.name == "sdss_r"

    def test_unknown_filter_raises(self, cache_dir):
        with pytest.raises(KeyError, match="Unknown filter"):
            load_filter("nonexistent_filter_xyz", cache_dir=cache_dir)


class TestLoadFilterSet:
    """Tests for load_filter_set (multiple filters)."""

    def test_returns_correct_count(self, cache_dir):
        names = ["sdss_g", "sdss_r", "sdss_i"]
        waves, trans, curves = load_filter_set(names, cache_dir=cache_dir)
        assert len(waves) == 3
        assert len(trans) == 3
        assert len(curves) == 3

    def test_all_are_filtercurves(self, cache_dir):
        names = ["sdss_g", "sdss_r"]
        _, _, curves = load_filter_set(names, cache_dir=cache_dir)
        for fc in curves:
            assert isinstance(fc, FilterCurve)

    def test_wavelength_ordering(self, cache_dir):
        """Effective wavelengths should increase g < r < i."""
        names = ["sdss_g", "sdss_r", "sdss_i"]
        waves, trans, _ = load_filter_set(names, cache_dir=cache_dir)
        eff_waves = []
        for w, t in zip(waves, trans):
            eff = float(jnp.sum(w * t) / jnp.sum(t))
            eff_waves.append(eff)
        assert eff_waves[0] < eff_waves[1] < eff_waves[2]


class TestLoadCustomFilter:
    """Tests for load_custom_filter (user-provided files)."""

    def test_loads_correctly(self, custom_filter_path):
        fc = load_custom_filter(custom_filter_path)
        assert isinstance(fc, FilterCurve)
        assert len(fc.wave) == 50

    def test_normalized(self, custom_filter_path):
        fc = load_custom_filter(custom_filter_path)
        assert float(jnp.max(fc.trans)) == pytest.approx(1.0, abs=1e-6)

    def test_wavelength_monotonic(self, custom_filter_path):
        fc = load_custom_filter(custom_filter_path)
        diffs = jnp.diff(fc.wave)
        assert float(jnp.min(diffs)) > 0.0

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_custom_filter("/nonexistent/path/filter.dat")


class TestDownloadFilter:
    """Tests for the low-level download_filter function."""

    def test_caches_to_disk(self, cache_dir):
        svo_id = FILTER_REGISTRY["sdss_r"]
        wave, trans = download_filter(svo_id, cache_dir=cache_dir)
        expected_file = Path(cache_dir) / "SLOAN_SDSS_r.dat"
        assert expected_file.exists()
        assert len(wave) == len(trans)
        assert len(wave) > 0

    def test_loads_from_cache(self, cache_dir):
        """Second call should load from disk (no network needed)."""
        svo_id = FILTER_REGISTRY["sdss_r"]
        wave1, trans1 = download_filter(svo_id, cache_dir=cache_dir)
        wave2, trans2 = download_filter(svo_id, cache_dir=cache_dir)
        np.testing.assert_array_equal(wave1, wave2)
        np.testing.assert_array_equal(trans1, trans2)


class TestListAvailableFilters:
    """Tests for the registry listing function."""

    def test_returns_dict(self):
        result = list_available_filters()
        assert isinstance(result, dict)
        assert len(result) == len(FILTER_REGISTRY)

    def test_registry_not_empty(self):
        assert len(FILTER_REGISTRY) > 50


class TestFilterRegistry:
    """Tests for the FILTER_REGISTRY itself."""

    def test_all_values_are_strings(self):
        for name, svo_id in FILTER_REGISTRY.items():
            assert isinstance(name, str)
            assert isinstance(svo_id, str)
            assert "/" in svo_id, f"SVO ID '{svo_id}' should contain '/'"
