"""Tests for filter metadata: effective wavelength, FWHM, facility inference, and filter_info.

Tests Feature 1 (Rich filter listing) — the functions added to
``tengri.observation.filters`` for CIGALE-style filter inspection.
"""

import numpy as np
import pytest

from tengri.observation.filters import (
    FILTER_REGISTRY,
    _infer_facility,
    compute_effective_wavelength,
    compute_fwhm,
    filter_info,
    list_available_filters,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cache_dir():
    return "data/filters"


@pytest.fixture()
def gaussian_filter():
    """Gaussian transmission curve centered at 6000 Å, sigma=300 Å."""
    wave = np.linspace(4000.0, 8000.0, 500)
    trans = np.exp(-0.5 * ((wave - 6000.0) / 300.0) ** 2)
    return wave, trans


@pytest.fixture()
def tophat_filter():
    """Uniform (top-hat) transmission from 5000 to 7000 Å."""
    wave = np.linspace(5000.0, 7000.0, 200)
    trans = np.ones_like(wave)
    return wave, trans


# ---------------------------------------------------------------------------
# compute_effective_wavelength
# ---------------------------------------------------------------------------


class TestComputeEffectiveWavelength:
    """Tests for λ_eff = ∫T·λ·dλ / ∫T·dλ."""

    def test_gaussian_centered(self, gaussian_filter):
        """Gaussian transmission → λ_eff ≈ center."""
        wave, trans = gaussian_filter
        lam_eff = compute_effective_wavelength(wave, trans)
        assert lam_eff == pytest.approx(6000.0, abs=5.0)

    def test_tophat_midpoint(self, tophat_filter):
        """Uniform transmission → λ_eff = midpoint."""
        wave, trans = tophat_filter
        lam_eff = compute_effective_wavelength(wave, trans)
        assert lam_eff == pytest.approx(6000.0, abs=2.0)

    def test_zero_transmission_returns_zero(self):
        """All-zero transmission → 0."""
        wave = np.linspace(5000.0, 7000.0, 100)
        trans = np.zeros_like(wave)
        assert compute_effective_wavelength(wave, trans) == 0.0

    def test_delta_function(self):
        """Single-point peak → λ_eff at that point."""
        wave = np.linspace(4000.0, 8000.0, 1000)
        trans = np.zeros_like(wave)
        idx = np.argmin(np.abs(wave - 5500.0))
        trans[idx - 2 : idx + 3] = 1.0
        lam_eff = compute_effective_wavelength(wave, trans)
        assert lam_eff == pytest.approx(5500.0, abs=10.0)

    def test_asymmetric_curve(self):
        """Red-weighted curve → λ_eff > geometric center."""
        wave = np.linspace(5000.0, 7000.0, 200)
        trans = np.where(wave > 6000.0, 1.0, 0.2)
        lam_eff = compute_effective_wavelength(wave, trans)
        assert lam_eff > 6000.0

    def test_returns_float(self, gaussian_filter):
        wave, trans = gaussian_filter
        assert isinstance(compute_effective_wavelength(wave, trans), float)


# ---------------------------------------------------------------------------
# compute_fwhm
# ---------------------------------------------------------------------------


class TestComputeFWHM:
    """Tests for full width at half maximum."""

    def test_gaussian_fwhm(self, gaussian_filter):
        """Gaussian with σ=300 → FWHM ≈ 2.355×σ ≈ 706 Å."""
        wave, trans = gaussian_filter
        fwhm = compute_fwhm(wave, trans)
        expected = 2.355 * 300.0
        assert fwhm == pytest.approx(expected, rel=0.05)

    def test_tophat_fwhm(self, tophat_filter):
        """Top-hat → FWHM = full width."""
        wave, trans = tophat_filter
        fwhm = compute_fwhm(wave, trans)
        assert fwhm == pytest.approx(2000.0, rel=0.02)

    def test_zero_transmission_returns_zero(self):
        """All-zero transmission → FWHM = 0."""
        wave = np.linspace(5000.0, 7000.0, 100)
        trans = np.zeros_like(wave)
        assert compute_fwhm(wave, trans) == 0.0

    def test_narrow_spike(self):
        """Single narrow peak → small FWHM."""
        wave = np.linspace(5000.0, 7000.0, 1000)
        trans = np.exp(-0.5 * ((wave - 6000.0) / 10.0) ** 2)
        fwhm = compute_fwhm(wave, trans)
        assert fwhm < 50.0
        assert fwhm > 0.0

    def test_returns_float(self, gaussian_filter):
        wave, trans = gaussian_filter
        assert isinstance(compute_fwhm(wave, trans), float)


# ---------------------------------------------------------------------------
# _format_wavelength (private but critical for display)
# ---------------------------------------------------------------------------


class TestFormatWavelength:
    """Tests for wavelength display formatting."""

    def test_angstrom_range(self):
        from tengri.observation.filters import _format_wavelength

        result = _format_wavelength(5500.0)
        assert "\u00c5" in result or "Å" in result
        assert "5500" in result

    def test_micron_range(self):
        from tengri.observation.filters import _format_wavelength

        result = _format_wavelength(20000.0)
        assert "\u03bcm" in result or "μm" in result
        assert "2.00" in result

    def test_cm_range(self):
        from tengri.observation.filters import _format_wavelength

        result = _format_wavelength(1e8)
        assert "cm" in result
        assert "1.00" in result

    def test_boundary_angstrom_to_micron(self):
        from tengri.observation.filters import _format_wavelength

        below = _format_wavelength(9999.0)
        at_boundary = _format_wavelength(10000.0)
        assert "\u00c5" in below or "Å" in below
        assert "\u03bcm" in at_boundary or "μm" in at_boundary

    def test_boundary_micron_to_cm(self):
        from tengri.observation.filters import _format_wavelength

        below = _format_wavelength(9.99e6)
        at_boundary = _format_wavelength(1e7)
        assert "\u03bcm" in below or "μm" in below
        assert "cm" in at_boundary


# ---------------------------------------------------------------------------
# _infer_facility
# ---------------------------------------------------------------------------


class TestInferFacility:
    """Tests for facility name inference from filter short names."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("sdss_r", "SDSS"),
            ("jwst_f200w", "JWST/NIRCam"),
            ("hst_f814w", "HST"),
            ("irac_36", "Spitzer/IRAC"),
            ("galex_fuv", "GALEX"),
            ("2mass_j", "2MASS"),
            ("wise_w1", "WISE"),
            ("herschel_pacs70", "Herschel"),
            ("miri_f770w", "JWST/MIRI"),
            ("lsst_g", "LSST/Rubin"),
        ],
    )
    def test_known_prefixes(self, name, expected):
        assert _infer_facility(name) == expected

    def test_unknown_prefix(self):
        assert _infer_facility("totally_unknown_filter") == "Other"

    def test_empty_string(self):
        assert _infer_facility("") == "Other"


# ---------------------------------------------------------------------------
# filter_info
# ---------------------------------------------------------------------------


class TestFilterInfo:
    """Tests for the filter_info() function."""

    def test_returns_all_keys(self, cache_dir):
        info = filter_info("sdss_r", cache_dir=cache_dir)
        expected_keys = {
            "name",
            "svo_id",
            "facility",
            "lambda_eff_aa",
            "fwhm_aa",
            "lambda_eff_str",
            "fwhm_str",
        }
        assert set(info.keys()) == expected_keys

    def test_correct_name(self, cache_dir):
        info = filter_info("sdss_r", cache_dir=cache_dir)
        assert info["name"] == "sdss_r"

    def test_correct_svo_id(self, cache_dir):
        info = filter_info("sdss_r", cache_dir=cache_dir)
        assert info["svo_id"] == "SLOAN/SDSS.r"

    def test_correct_facility(self, cache_dir):
        info = filter_info("sdss_r", cache_dir=cache_dir)
        assert info["facility"] == "SDSS"

    def test_lambda_eff_reasonable(self, cache_dir):
        """SDSS r-band effective wavelength ≈ 6200 Å."""
        info = filter_info("sdss_r", cache_dir=cache_dir)
        assert 5800.0 < info["lambda_eff_aa"] < 6600.0

    def test_fwhm_positive(self, cache_dir):
        info = filter_info("sdss_r", cache_dir=cache_dir)
        assert info["fwhm_aa"] > 0.0

    def test_fwhm_reasonable(self, cache_dir):
        """SDSS r-band FWHM ≈ 1100 Å."""
        info = filter_info("sdss_r", cache_dir=cache_dir)
        assert 500.0 < info["fwhm_aa"] < 2000.0

    def test_formatted_strings_nonempty(self, cache_dir):
        info = filter_info("sdss_r", cache_dir=cache_dir)
        assert len(info["lambda_eff_str"]) > 0
        assert len(info["fwhm_str"]) > 0

    def test_unknown_filter_raises(self):
        with pytest.raises(KeyError, match="Unknown filter"):
            filter_info("nonexistent_filter_xyz")

    def test_ir_filter_in_micron(self, cache_dir):
        """2MASS J-band effective wavelength should be in μm range."""
        info = filter_info("2mass_j", cache_dir=cache_dir)
        assert info["lambda_eff_aa"] > 10000.0
        assert "\u03bcm" in info["lambda_eff_str"] or "μm" in info["lambda_eff_str"]


# ---------------------------------------------------------------------------
# list_available_filters (enhanced)
# ---------------------------------------------------------------------------


class TestListAvailableFiltersEnhanced:
    """Tests for the enhanced list_available_filters()."""

    def test_returns_full_registry(self):
        result = list_available_filters(group_by="none")
        assert len(result) == len(FILTER_REGISTRY)

    def test_grouped_returns_same_count(self):
        result = list_available_filters(group_by="facility")
        assert len(result) == len(FILTER_REGISTRY)

    def test_returns_dict(self):
        result = list_available_filters()
        assert isinstance(result, dict)

    def test_all_registry_keys_present(self):
        result = list_available_filters()
        for name in FILTER_REGISTRY:
            assert name in result

    def test_compute_properties_false_does_not_error(self, capsys):
        list_available_filters(group_by="none", compute_properties=False)
        captured = capsys.readouterr()
        assert "Total:" in captured.out
