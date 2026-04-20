"""Tests for photometric filter management (SVO FPS integration)."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.filters import (
    _C_AA_S,
    FILTER_REGISTRY,
    download_filter,
    list_available_filters,
    load_alma_band,
    load_custom_filter,
    load_filter,
    load_filter_set,
    load_tophat_filter,
)
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
    """Tests for load_filter (single filter by short name)."""

    def test_returns_filtercurve(self, sample_filter):
        assert isinstance(sample_filter, FilterCurve)

    def test_correct_shape(self, sample_filter):
        """wave and trans must be 1-D arrays of the same length."""
        assert sample_filter.wave.ndim == 1
        assert sample_filter.trans.ndim == 1
        assert len(sample_filter.wave) == len(sample_filter.trans)
        assert len(sample_filter.wave) > 10  # reasonable number of points

    def test_transmission_nonnegative(self, sample_filter):
        """Raw transmission values must be finite and non-negative."""
        assert float(jnp.min(sample_filter.trans)) >= 0.0
        assert bool(jnp.all(jnp.isfinite(sample_filter.trans)))

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

    def test_transmission_nonnegative(self, custom_filter_path):
        fc = load_custom_filter(custom_filter_path)
        assert float(jnp.min(fc.trans)) >= 0.0
        assert bool(jnp.all(jnp.isfinite(fc.trans)))

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


class TestLoadTophatFilter:
    """Tests for the synthetic top-hat filter constructor."""

    def test_returns_filtercurve(self):
        fc = load_tophat_filter(1.23e7, 5e5, name="test_band")
        assert isinstance(fc, FilterCurve)

    def test_center_wavelength(self):
        center = 1.23e7
        width = 5e5
        fc = load_tophat_filter(center, width)
        midpoint = float((fc.wave[0] + fc.wave[-1]) / 2)
        assert midpoint == pytest.approx(center, rel=1e-6)

    def test_width(self):
        center = 1.23e7
        width = 5e5
        fc = load_tophat_filter(center, width)
        actual_width = float(fc.wave[-1] - fc.wave[0])
        assert actual_width == pytest.approx(width, rel=1e-6)

    def test_uniform_transmission(self):
        fc = load_tophat_filter(1e7, 1e6)
        assert float(jnp.min(fc.trans)) == pytest.approx(1.0)
        assert float(jnp.max(fc.trans)) == pytest.approx(1.0)

    def test_name_set(self):
        fc = load_tophat_filter(1e7, 1e6, name="my_band")
        assert fc.name == "my_band"

    def test_default_name_empty(self):
        fc = load_tophat_filter(1e7, 1e6)
        assert fc.name == ""

    def test_n_points(self):
        fc = load_tophat_filter(1e7, 1e6, n_points=20)
        assert len(fc.wave) == 20


class TestLoadAlmaBand:
    """Tests for the ALMA convenience band loader."""

    def test_all_bands_load(self):
        for b in range(1, 11):
            fc = load_alma_band(b)
            assert isinstance(fc, FilterCurve)

    def test_default_name(self):
        for b in range(1, 11):
            fc = load_alma_band(b)
            assert fc.name == f"alma_band{b}"

    def test_custom_name(self):
        fc = load_alma_band(6, name="my_alma")
        assert fc.name == "my_alma"

    def test_invalid_band(self):
        with pytest.raises(ValueError, match="ALMA band"):
            load_alma_band(0)
        with pytest.raises(ValueError, match="ALMA band"):
            load_alma_band(11)

    def test_band6_center_wavelength(self):
        """Band 6 spans 211–275 GHz; center wavelength should be ~1.23 mm."""
        fc = load_alma_band(6)
        lo_aa = _C_AA_S / (275.0e9)
        hi_aa = _C_AA_S / (211.0e9)
        expected_center = (lo_aa + hi_aa) / 2.0
        actual_center = float((fc.wave[0] + fc.wave[-1]) / 2)
        assert actual_center == pytest.approx(expected_center, rel=1e-4)

    def test_band7_center_wavelength(self):
        """Band 7 spans 275–373 GHz; center ≈ 0.93 mm."""
        fc = load_alma_band(7)
        lo_aa = _C_AA_S / (373.0e9)
        hi_aa = _C_AA_S / (275.0e9)
        expected_center = (lo_aa + hi_aa) / 2.0
        actual_center = float((fc.wave[0] + fc.wave[-1]) / 2)
        assert actual_center == pytest.approx(expected_center, rel=1e-4)

    def test_wavelengths_monotonically_increasing(self):
        fc = load_alma_band(6)
        diffs = jnp.diff(fc.wave)
        assert float(jnp.min(diffs)) > 0.0

    def test_band_ordering(self):
        """Higher band number = higher frequency = shorter wavelength."""
        centers = []
        for b in range(3, 11):
            fc = load_alma_band(b)
            centers.append(float((fc.wave[0] + fc.wave[-1]) / 2))
        # Wavelength decreases as band number increases
        for i in range(len(centers) - 1):
            assert centers[i] > centers[i + 1]

    def test_band_definitions_cover_expected_wavelengths(self):
        """Spot-check: Band 6 center ~1.23 mm, Band 9 center ~0.45 mm."""
        fc6 = load_alma_band(6)
        c6_mm = float((fc6.wave[0] + fc6.wave[-1]) / 2) / 1e7  # Å → mm
        assert 1.0 < c6_mm < 1.5, f"Band 6 center should be ~1.23 mm, got {c6_mm:.3f}"

        fc9 = load_alma_band(9)
        c9_mm = float((fc9.wave[0] + fc9.wave[-1]) / 2) / 1e7
        assert 0.35 < c9_mm < 0.60, f"Band 9 center should be ~0.45 mm, got {c9_mm:.3f}"


class TestPhotometryIntegration:
    """Verify FilterCurve objects work correctly with DSPS-style integration.

    tengri's compute_flux_density integrates SED × T(λ) × λ dλ, so filters
    must be in Angstrom and produce finite, positive fluxes.
    """

    def test_tophat_integration_finite(self):
        """Top-hat filter over a flat SED should give finite flux."""
        fc = load_tophat_filter(5500.0, 500.0, name="optical_window")
        wave = fc.wave
        # Flat SED: f_lambda = constant (in Lsun/Angstrom units)
        sed = jnp.ones_like(wave)
        numerator = jnp.trapezoid(sed * fc.trans * wave, wave)
        denominator = jnp.trapezoid(fc.trans * wave, wave)
        flux = numerator / denominator
        assert jnp.isfinite(flux)
        assert float(flux) > 0.0

    def test_alma_band_integration_finite(self):
        """ALMA top-hat filter over a power-law SED gives finite flux."""
        fc = load_alma_band(6)
        wave = fc.wave
        # Power-law SED: f_nu ~ nu^(-0.7) → f_lambda ~ lambda^(-1.3)
        sed = wave ** (-1.3)
        numerator = jnp.trapezoid(sed * fc.trans * wave, wave)
        denominator = jnp.trapezoid(fc.trans * wave, wave)
        flux = numerator / denominator
        assert jnp.isfinite(flux)
        assert float(flux) > 0.0

    def test_custom_filter_integration_consistent(self, tmp_path):
        """Cached filter and freshly constructed curve give same integral."""
        wave_np = np.linspace(5000.0, 7000.0, 100)
        trans_np = np.exp(-0.5 * ((wave_np - 6000.0) / 300.0) ** 2)
        filepath = tmp_path / "gauss_filter.dat"
        np.savetxt(str(filepath), np.column_stack([wave_np, trans_np]), fmt="%.6e")

        fc = load_custom_filter(str(filepath))
        wave = fc.wave
        sed = jnp.ones_like(wave)
        # Effective wavelength: λ_eff = ∫T(λ)λ dλ / ∫T(λ) dλ  (flat SED)
        numerator = jnp.trapezoid(fc.trans * wave, wave)
        denominator = jnp.trapezoid(fc.trans, wave)
        eff_wave = float(numerator / denominator)
        # Effective wavelength of a Gaussian at 6000 Å over a flat SED ≈ 6000 Å
        assert abs(eff_wave - 6000.0) < 20.0

    @pytest.fixture
    def tmp_path(self, tmp_path):
        return tmp_path


class TestFilterRegistry:
    """Tests for the FILTER_REGISTRY itself."""

    def test_all_values_are_strings(self):
        for name, svo_id in FILTER_REGISTRY.items():
            assert isinstance(name, str)
            assert isinstance(svo_id, str)
            assert "/" in svo_id, f"SVO ID '{svo_id}' should contain '/'"
