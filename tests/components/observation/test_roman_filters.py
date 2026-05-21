"""Test Roman Space Telescope WFI filter pack loading and properties."""

from typing import ClassVar

import numpy as np
import pytest

from tengri.observation.filters import (
    FILTER_REGISTRY,
    filter_info,
    load_filter,
    load_filter_set,
)

pytestmark = pytest.mark.bounds


class TestRomanFiltersRegistry:
    """Test Roman filters are registered."""

    ROMAN_FILTERS: ClassVar[list[str]] = [
        "roman_f062",
        "roman_f087",
        "roman_f106",
        "roman_f129",
        "roman_f158",
        "roman_f184",
        "roman_f213",
    ]

    def test_all_roman_filters_registered(self):
        """All Roman filters exist in the registry."""
        for name in self.ROMAN_FILTERS:
            assert name in FILTER_REGISTRY, f"Roman filter {name} not in registry"

    def test_roman_svo_ids(self):
        """Roman filters map to correct SVO IDs."""
        expected_svo = {
            "roman_f062": "Roman/WFI.F062",
            "roman_f087": "Roman/WFI.F087",
            "roman_f106": "Roman/WFI.F106",
            "roman_f129": "Roman/WFI.F129",
            "roman_f158": "Roman/WFI.F158",
            "roman_f184": "Roman/WFI.F184",
            "roman_f213": "Roman/WFI.F213",
        }
        for name, svo_id in expected_svo.items():
            assert FILTER_REGISTRY[name] == svo_id


class TestRomanFilterData:
    """Test Roman filter transmission curves are valid."""

    ROMAN_FILTERS: ClassVar[dict[str, float]] = {
        "roman_f062": 0.62e4,  # F062 ~ 0.62 µm
        "roman_f087": 0.87e4,  # F087 ~ 0.87 µm
        "roman_f106": 1.06e4,  # F106 ~ 1.06 µm
        "roman_f129": 1.29e4,  # F129 ~ 1.29 µm
        "roman_f158": 1.58e4,  # F158 ~ 1.58 µm
        "roman_f184": 1.84e4,  # F184 ~ 1.84 µm
        "roman_f213": 2.13e4,  # F213 ~ 2.13 µm
    }

    @pytest.mark.parametrize("filter_name,expected_pivot_aa", list(ROMAN_FILTERS.items()))
    def test_roman_filter_loads(self, filter_name, expected_pivot_aa):
        """Roman filter can be loaded from cache."""
        curve = load_filter(filter_name)
        assert curve is not None
        assert hasattr(curve, "wave")
        assert hasattr(curve, "trans")

    @pytest.mark.parametrize("filter_name,expected_pivot_aa", list(ROMAN_FILTERS.items()))
    def test_roman_filter_has_min_points(self, filter_name, expected_pivot_aa):
        """Roman filter has at least 50 wavelength points."""
        curve = load_filter(filter_name)
        assert len(curve.wave) >= 50, f"{filter_name} has only {len(curve.wave)} points"

    @pytest.mark.parametrize("filter_name,expected_pivot_aa", list(ROMAN_FILTERS.items()))
    def test_roman_filter_transmission_bounds(self, filter_name, expected_pivot_aa):
        """Roman filter transmission is positive (SVO may normalize > 1.0)."""
        curve = load_filter(filter_name)
        trans_np = np.asarray(curve.trans)
        # SVO includes normalized values that can exceed 1.0
        assert trans_np.min() >= -0.01, f"{filter_name} transmission min {trans_np.min()} < 0"
        # Allow higher bounds for normalized SVO filters
        assert trans_np.max() <= 3.0, f"{filter_name} transmission max {trans_np.max()} > 3.0"

    @pytest.mark.parametrize("filter_name,expected_pivot_aa", list(ROMAN_FILTERS.items()))
    def test_roman_filter_peak_transmission(self, filter_name, expected_pivot_aa):
        """Roman filter peak transmission is significant."""
        curve = load_filter(filter_name)
        trans_np = np.asarray(curve.trans)
        peak = trans_np.max()
        assert peak > 0.3, f"{filter_name} peak transmission {peak} < 0.3"

    @pytest.mark.parametrize("filter_name,expected_pivot_aa", list(ROMAN_FILTERS.items()))
    def test_roman_filter_pivot_wavelength(self, filter_name, expected_pivot_aa):
        """Roman filter pivot wavelength near expected central wavelength."""
        info = filter_info(filter_name)
        pivot = info["lambda_eff_aa"]

        # Allow ±20% tolerance around expected wavelength
        tolerance = expected_pivot_aa * 0.20
        assert abs(pivot - expected_pivot_aa) < tolerance, (
            f"{filter_name} pivot {pivot:.0f} Å far from expected {expected_pivot_aa:.0f} Å"
        )

    @pytest.mark.parametrize("filter_name,_", list(ROMAN_FILTERS.items()))
    def test_roman_filter_metadata(self, filter_name, _):
        """Roman filter has required metadata properties."""
        info = filter_info(filter_name)
        assert "lambda_eff_aa" in info
        assert "fwhm_aa" in info
        assert info["fwhm_aa"] > 0, f"{filter_name} FWHM is zero"


class TestRomanFilterSet:
    """Test loading multiple Roman filters together."""

    def test_load_roman_filter_set(self):
        """Load all Roman filters as a set."""
        filter_names = [
            "roman_f062",
            "roman_f087",
            "roman_f106",
            "roman_f129",
            "roman_f158",
            "roman_f184",
            "roman_f213",
        ]
        _, _, filter_curves = load_filter_set(filter_names)
        assert len(filter_curves) == 7
        for curve in filter_curves:
            assert curve.name in filter_names
            assert curve is not None

    def test_roman_filter_set_ordered(self):
        """Roman filter set maintains wavelength order."""
        filter_names = [
            "roman_f062",
            "roman_f087",
            "roman_f106",
            "roman_f129",
            "roman_f158",
            "roman_f184",
            "roman_f213",
        ]
        _, _, filter_curves = load_filter_set(filter_names)
        pivots = [filter_info(curve.name)["lambda_eff_aa"] for curve in filter_curves]
        # Should be monotonically increasing
        assert pivots == sorted(pivots), "Roman filters not in wavelength order"
