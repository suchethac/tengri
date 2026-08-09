# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for filter metadata: effective wavelength, FWHM, facility inference.

Frozen: λ_eff formula (∫T·λ·dλ / ∫T·dλ) reproduces known centers (Gaussian,
top-hat); FWHM formula (width at half-max) matches analytical values; wavelength
formatting (Å/μm/cm ranges); facility inference from filter names; filter_info
returns complete dict with physical values in expected ranges.
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
from tengri.registry import _RegistryTable

pytestmark = pytest.mark.bounds

# ── Fixtures ──────────────────────────────────────────────────────


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


# ── compute_effective_wavelength ──────────────────────────────────


class TestComputeEffectiveWavelength:
    """Frozen: λ_eff = ∫T·λ·dλ / ∫T·dλ reproduces centers."""

    def test_gaussian_centered(self, gaussian_filter):
        """Gaussian transmission → λ_eff ≈ center (6000 Å)."""
        wave, trans = gaussian_filter
        lam_eff = compute_effective_wavelength(wave, trans)
        assert lam_eff == pytest.approx(6000.0, abs=5.0)

    def test_tophat_midpoint(self, tophat_filter):
        """Uniform transmission → λ_eff = midpoint (6000 Å)."""
        wave, trans = tophat_filter
        lam_eff = compute_effective_wavelength(wave, trans)
        assert lam_eff == pytest.approx(6000.0, abs=2.0)

    def test_zero_transmission_returns_zero(self):
        """All-zero transmission → λ_eff = 0."""
        wave = np.linspace(5000.0, 7000.0, 100)
        trans = np.zeros_like(wave)
        assert compute_effective_wavelength(wave, trans) == 0.0

    def test_delta_function_peak(self):
        """Single narrow peak → λ_eff at peak location."""
        wave = np.linspace(4000.0, 8000.0, 1000)
        trans = np.zeros_like(wave)
        idx = np.argmin(np.abs(wave - 5500.0))
        trans[idx - 2 : idx + 3] = 1.0
        lam_eff = compute_effective_wavelength(wave, trans)
        assert lam_eff == pytest.approx(5500.0, abs=10.0)

    def test_red_weighted_curve_asymmetry(self):
        """Red-weighted curve (T=1 for λ>6000, T=0.2 for λ<6000) → λ_eff > 6000."""
        wave = np.linspace(5000.0, 7000.0, 200)
        trans = np.where(wave > 6000.0, 1.0, 0.2)
        lam_eff = compute_effective_wavelength(wave, trans)
        assert lam_eff > 6000.0


# ── compute_fwhm ──────────────────────────────────────────────────


class TestComputeFWHM:
    """Frozen: FWHM (full width at half maximum) matches analytical values."""

    def test_gaussian_fwhm_formula(self, gaussian_filter):
        """Gaussian with σ=300 Å → FWHM = 2.355×σ ≈ 706 Å."""
        wave, trans = gaussian_filter
        fwhm = compute_fwhm(wave, trans)
        expected = 2.355 * 300.0
        assert fwhm == pytest.approx(expected, rel=0.05)

    def test_tophat_fwhm_full_width(self, tophat_filter):
        """Top-hat 5000–7000 Å → FWHM = 2000 Å."""
        wave, trans = tophat_filter
        fwhm = compute_fwhm(wave, trans)
        assert fwhm == pytest.approx(2000.0, rel=0.02)

    def test_zero_transmission_returns_zero(self):
        """All-zero transmission → FWHM = 0."""
        wave = np.linspace(5000.0, 7000.0, 100)
        trans = np.zeros_like(wave)
        assert compute_fwhm(wave, trans) == 0.0

    def test_narrow_spike_small_fwhm(self):
        """Narrow Gaussian peak (σ=10 Å) → FWHM < 50 Å."""
        wave = np.linspace(5000.0, 7000.0, 1000)
        trans = np.exp(-0.5 * ((wave - 6000.0) / 10.0) ** 2)
        fwhm = compute_fwhm(wave, trans)
        assert 0.0 < fwhm < 50.0


# ── _format_wavelength (private but critical for display) ─────────


class TestFormatWavelength:
    """Wavelength display formatting: Å/μm/cm ranges and boundaries."""

    def test_angstrom_range(self):
        """λ < 10000 Å → displays in Angstrom."""
        from tengri.observation.filters import _format_wavelength

        result = _format_wavelength(5500.0)
        assert "Å" in result or "Å" in result
        assert "5500" in result

    def test_micron_range(self):
        """10000 Å ≤ λ < 10^7 Å → displays in μm."""
        from tengri.observation.filters import _format_wavelength

        result = _format_wavelength(20000.0)
        assert "μm" in result or "μm" in result
        assert "2.00" in result

    def test_cm_range(self):
        """λ ≥ 10^7 Å → displays in cm."""
        from tengri.observation.filters import _format_wavelength

        result = _format_wavelength(1e8)
        assert "cm" in result
        assert "1.00" in result

    def test_boundary_angstrom_to_micron(self):
        """At λ=10000 Å boundary: just below is Å, at is μm."""
        from tengri.observation.filters import _format_wavelength

        below = _format_wavelength(9999.0)
        at_boundary = _format_wavelength(10000.0)
        assert "Å" in below or "Å" in below
        assert "μm" in at_boundary or "μm" in at_boundary

    def test_boundary_micron_to_cm(self):
        """At λ=10^7 Å boundary: just below is μm, at is cm."""
        from tengri.observation.filters import _format_wavelength

        below = _format_wavelength(9.99e6)
        at_boundary = _format_wavelength(1e7)
        assert "μm" in below or "μm" in below
        assert "cm" in at_boundary


# ── _infer_facility ───────────────────────────────────────────────


class TestInferFacility:
    """Frozen: facility name inference from filter short-name prefixes."""

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
        """Frozen facility mappings for known filter prefixes."""
        assert _infer_facility(name) == expected

    def test_unknown_prefix(self):
        """Unknown prefix → 'Other'."""
        assert _infer_facility("totally_unknown_filter") == "Other"

    def test_empty_string(self):
        """Empty string → 'Other'."""
        assert _infer_facility("") == "Other"


# ── filter_info ───────────────────────────────────────────────────


class TestFilterInfo:
    """filter_info: dict structure, SDSS r-band physics, IR handling."""

    def test_sdss_r_lambda_eff_physical(self, cache_dir):
        """SDSS r-band λ_eff ≈ 6200 Å (expected from filter design)."""
        info = filter_info("sdss_r", cache_dir=cache_dir)
        assert 5800.0 < info["lambda_eff_aa"] < 6600.0

    def test_sdss_r_fwhm_positive_and_reasonable(self, cache_dir):
        """SDSS r-band FWHM ≈ 1100 Å (expected from filter bandwidth)."""
        info = filter_info("sdss_r", cache_dir=cache_dir)
        assert info["fwhm_aa"] > 0.0
        assert 500.0 < info["fwhm_aa"] < 2000.0

    def test_unknown_filter_raises(self):
        """Unknown filter key raises KeyError."""
        with pytest.raises(KeyError, match="Unknown filter"):
            filter_info("nonexistent_filter_xyz")

    def test_ir_filter_wavelength_range(self, cache_dir):
        """2MASS J-band (IR) λ_eff in μm range (>10000 Å) and formatted as μm."""
        info = filter_info("2mass_j", cache_dir=cache_dir)
        assert info["lambda_eff_aa"] > 10000.0
        assert "μm" in info["lambda_eff_str"] or "μm" in info["lambda_eff_str"]


# ── list_available_filters (enhanced) ─────────────────────────────


class TestListAvailableFiltersEnhanced:
    """Enhanced list_available_filters: grouping and property computation."""

    def test_compute_properties_false_omits_the_expensive_columns(self):
        """Without compute_properties, no curve is loaded, so no λ_eff/FWHM."""
        rows = list_available_filters(group_by="none", compute_properties=False)
        assert len(rows) > 0
        assert set(rows[0]) >= {"name", "facility", "svo_id"}
        assert "lambda_eff" not in rows[0]
        assert "fwhm" not in rows[0]

    def test_it_returns_a_table_and_prints_nothing(self, capsys):
        """Printed 250 rows to stdout as a side effect until #1574."""
        rows = list_available_filters(group_by="none", compute_properties=False)
        assert isinstance(rows, _RegistryTable)
        assert capsys.readouterr().out == ""

    def test_group_by_facility_orders_rows_by_facility(self):
        """Grouping became an ordering + a column when the printing went away."""
        rows = list_available_filters(group_by="facility", compute_properties=False)
        facilities = [r["facility"] for r in rows]
        assert facilities == sorted(facilities)

    def test_to_dict_gives_back_the_pre_1574_mapping(self):
        """The documented migration for callers that wanted the dict."""
        mapping = list_available_filters().to_dict("svo_id")
        assert isinstance(mapping, dict)
        assert mapping["sdss_r"] == FILTER_REGISTRY["sdss_r"]
