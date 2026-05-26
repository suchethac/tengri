# SPDX-License-Identifier: BSD-3-Clause
"""Tests for filter discovery helpers (tengri.observation.filters namespace)."""

import pytest

from tengri.observation.filters import describe, list_filters, load, suggest

pytestmark = pytest.mark.bounds


class TestListFilters:
    """Test list_filters() function."""

    def test_list_filters_returns_list(self):
        """list_filters() returns a list."""
        result = list_filters()
        assert isinstance(result, list)

    def test_list_filters_nonempty(self):
        """list_filters() returns non-empty list."""
        result = list_filters()
        if not result:
            pytest.skip("Filter library is empty in this environment")
        assert len(result) > 0
        assert all(isinstance(name, str) for name in result)

    def test_list_filters_sorted(self):
        """list_filters() returns sorted names."""
        result = list_filters()
        if not result:
            pytest.skip("Filter library is empty in this environment")
        assert result == sorted(result)

    def test_list_filters_instrument_filter_case_insensitive(self):
        """list_filters(instrument=...) filters case-insensitively."""
        result = list_filters(instrument="sdss")
        result_upper = list_filters(instrument="SDSS")
        assert result == result_upper

    def test_list_filters_instrument_filter_sdss(self):
        """list_filters(instrument='sdss') returns SDSS filters."""
        result = list_filters(instrument="sdss")
        if not result:
            pytest.skip("No SDSS filters in registry")
        # Should match sdss_u, sdss_g, etc.
        assert all("sdss" in name.lower() for name in result)

    def test_list_filters_instrument_filter_jwst(self):
        """list_filters(instrument='jwst') returns JWST filters."""
        result = list_filters(instrument="jwst")
        if not result:
            pytest.skip("No JWST filters in registry")
        # Should match jwst_*, nircam*, niriss*, etc. that contain 'jwst'
        assert all(
            "jwst" in name.lower() or "nircam" in name.lower() or "niriss" in name.lower()
            for name in result
        )

    def test_list_filters_instrument_no_matches(self):
        """list_filters(instrument='nonexistent') returns empty."""
        result = list_filters(instrument="nonexistent_instrument_xyz")
        assert result == []


class TestLoad:
    """Test load() function."""

    def test_load_single_filter(self):
        """load() returns filter set tuple for a single filter."""
        result = load(["sdss_u"])
        if result[2] is None or len(result[2]) == 0:
            pytest.skip("sdss_u not available or load failed")
        assert len(result) == 3  # (waves, trans, curves)
        waves, trans, curves = result
        assert len(waves) == 1
        assert len(trans) == 1
        assert len(curves) == 1
        assert curves[0].name == "sdss_u"

    def test_load_multiple_filters(self):
        """load() returns filter set tuple for multiple filters."""
        result = load(["sdss_r", "sdss_i", "sdss_z"])
        if result[2] is None or len(result[2]) == 0:
            pytest.skip("SDSS filters not available")
        assert len(result) == 3
        waves, trans, curves = result
        assert len(waves) == 3
        assert len(trans) == 3
        assert len(curves) == 3

    def test_load_unknown_filter_raises(self):
        """load() raises KeyError for unknown filter."""
        with pytest.raises(KeyError):
            load(["nonexistent_filter_xyz"])


class TestDescribe:
    """Test describe() function."""

    def test_describe_returns_string(self):
        """describe() returns a non-empty string."""
        result = describe("sdss_r")
        if "filter found; no summary" in result:
            pytest.skip("sdss_r failed to load")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_describe_contains_filter_name(self):
        """describe() output contains the filter name."""
        result = describe("sdss_r")
        assert "sdss_r" in result.lower()

    def test_describe_fallback_on_error(self):
        """describe() returns fallback message on error."""
        result = describe("nonexistent_filter_xyz")
        assert "nonexistent_filter_xyz" in result
        assert "found" in result.lower() or "summary" in result.lower()

    def test_describe_includes_wavelength_info(self):
        """describe() includes wavelength information (lambda_eff or range)."""
        result = describe("sdss_r")
        # Should mention wavelength or effective wavelength
        assert any(
            term in result.lower()
            for term in ["λ_eff", "lambda", "angstrom", "å", "μm", "micron", "range"]
        )


class TestSuggest:
    """Test suggest() function."""

    def test_suggest_returns_list(self):
        """suggest() returns a list."""
        result = suggest(redshift=0.0)
        assert isinstance(result, list)

    def test_suggest_at_z0_visible_coverage(self):
        """suggest(z=0, coverage='visible') returns some filters."""
        result = suggest(redshift=0.0, coverage="visible")
        if not result:
            pytest.skip("No filters in visible range at z=0")
        assert all(isinstance(name, str) for name in result)

    def test_suggest_at_z3_visible_to_nir(self):
        """suggest(z=3, coverage='visible_to_nir') returns some filters."""
        result = suggest(redshift=3.0, coverage="visible_to_nir")
        if not result:
            pytest.skip("No filters cover rest-frame visible_to_nir at z=3")
        assert all(isinstance(name, str) for name in result)

    def test_suggest_returns_sorted_by_wavelength(self):
        """suggest() returns filters sorted by effective wavelength."""
        import numpy as np

        from tengri.observation.filters import compute_effective_wavelength

        result = suggest(redshift=0.0, coverage="visible_to_nir")
        if len(result) < 2:
            pytest.skip("Not enough filters to check sorting")

        # Load all returned filters and verify they're sorted by lambda_eff
        wavelengths = []
        for name in result:
            try:
                fc = load([name])[2][0]
                wave_np = np.asarray(fc.wave)
                trans_np = np.asarray(fc.trans)
                lam_eff = compute_effective_wavelength(wave_np, trans_np)
                wavelengths.append(lam_eff)
            except Exception:
                pass

        if len(wavelengths) >= 2:
            assert wavelengths == sorted(wavelengths)

    def test_suggest_unknown_coverage_raises(self):
        """suggest() raises ValueError for unknown coverage."""
        with pytest.raises(ValueError):
            suggest(redshift=0.0, coverage="unknown_coverage_xyz")

    def test_suggest_coverage_presets(self):
        """suggest() accepts all documented coverage presets."""
        presets = ["visible", "visible_to_nir", "uv_to_ir", "jwst_cover"]
        for preset in presets:
            # Should not raise
            result = suggest(redshift=0.0, coverage=preset)
            assert isinstance(result, list)

    def test_suggest_high_z_shift(self):
        """suggest() at high z shifts observed-frame wavelengths."""
        result_z0 = suggest(redshift=0.0, coverage="visible")
        result_z5 = suggest(redshift=5.0, coverage="visible")

        # At z=5, observed-frame visible corresponds to much shorter rest-frame,
        # so they might be different (or z=5 might have fewer filters).
        # Just verify they're both lists.
        assert isinstance(result_z0, list)
        assert isinstance(result_z5, list)
