"""Unit tests for preprocessing module."""

from __future__ import annotations

import chex
import numpy as np
import pytest

from tengri.preprocessing.error_floor import add_systematic_floor
from tengri.preprocessing.upper_limits import (
    detect_upper_limits,
    sigma_upper_limit_from_flux,
)
from tengri.preprocessing.zeropoints import (
    ZEROPOINT_REGISTRY,
    ZeropointEntry,
    apply_zeropoints,
    lookup_zeropoints,
)


class TestZeropointEntry:
    """Tests for ZeropointEntry dataclass."""

    def test_zeropoint_entry_frozen(self):
        """Verify ZeropointEntry is immutable."""
        entry = ZeropointEntry(
            survey="JADES",
            release="DR5",
            filter_name="F150W",
            mag_offset=0.1,
        )
        # Attempt to mutate should raise FrozenInstanceError (or AttributeError)
        with pytest.raises((AttributeError, TypeError)):
            entry.mag_offset = 0.2

    def test_zeropoint_entry_defaults(self):
        """Verify default values are correct."""
        entry = ZeropointEntry(
            survey="JADES",
            release="DR5",
            filter_name="F150W",
        )
        assert entry.mag_offset == 0.0
        assert entry.fractional_sys_err == 0.0
        assert entry.source == ""

    def test_zeropoint_entry_with_all_fields(self):
        """Verify all fields can be set."""
        entry = ZeropointEntry(
            survey="CEERS",
            release="v1",
            filter_name="F277W",
            mag_offset=0.05,
            fractional_sys_err=0.03,
            source="test source",
        )
        assert entry.survey == "CEERS"
        assert entry.release == "v1"
        assert entry.filter_name == "F277W"
        assert entry.mag_offset == 0.05
        assert entry.fractional_sys_err == 0.03
        assert entry.source == "test source"


class TestZeropointRegistry:
    """Tests for ZEROPOINT_REGISTRY."""

    def test_registry_nonempty(self):
        """Verify registry contains entries."""
        assert len(ZEROPOINT_REGISTRY) > 0

    def test_registry_all_frozen(self):
        """Verify all registry entries are frozen."""
        for entry in ZEROPOINT_REGISTRY:
            assert isinstance(entry, ZeropointEntry)
            # Attempt mutation should fail
            with pytest.raises((AttributeError, TypeError)):
                entry.mag_offset = 999.0

    def test_registry_jades_dr5_present(self):
        """Verify JADES DR5 entries are in the registry."""
        jades_filters = [
            "F090W",
            "F115W",
            "F150W",
            "F200W",
            "F277W",
            "F356W",
            "F410M",
            "F444W",
        ]
        registry_filters = {
            e.filter_name for e in ZEROPOINT_REGISTRY if e.survey == "JADES" and e.release == "DR5"
        }
        for filt in jades_filters:
            assert filt in registry_filters, f"JADES DR5 {filt} not in registry"

    def test_registry_entries_have_source(self):
        """Verify all entries have a source string (even if placeholder)."""
        for entry in ZEROPOINT_REGISTRY:
            assert isinstance(entry.source, str)
            # Allow "placeholder" entries but verify they are documented
            if "placeholder" in entry.source.lower():
                assert len(entry.source) > 10  # Non-empty documentation


class TestLookupZeropoints:
    """Tests for lookup_zeropoints function."""

    def test_lookup_known_filter(self):
        """Verify lookup works for a known filter."""
        entries = lookup_zeropoints("JADES", "DR5", ["F150W"])
        assert len(entries) == 1
        assert entries[0].filter_name == "F150W"
        assert entries[0].survey == "JADES"
        assert entries[0].release == "DR5"

    def test_lookup_multiple_filters(self):
        """Verify lookup works for multiple filters."""
        entries = lookup_zeropoints("JADES", "DR5", ["F150W", "F277W"])
        assert len(entries) == 2
        filter_names = [e.filter_name for e in entries]
        assert "F150W" in filter_names
        assert "F277W" in filter_names

    def test_lookup_unknown_filter_raises(self):
        """Verify KeyError is raised for unknown filter."""
        with pytest.raises(KeyError):
            lookup_zeropoints("JADES", "DR5", ["F999W"])

    def test_lookup_preserves_order(self):
        """Verify filter order is preserved in results."""
        entries = lookup_zeropoints("JADES", "DR5", ["F444W", "F150W", "F277W"])
        filter_names = [e.filter_name for e in entries]
        assert filter_names == ["F444W", "F150W", "F277W"]


class TestApplyZeropoints:
    """Tests for apply_zeropoints function."""

    def test_apply_zeropoints_shape_preserved(self):
        """Verify flux and error arrays keep their shape."""
        entries = lookup_zeropoints("JADES", "DR5", ["F150W", "F277W"])
        flux = np.array([[100.0, 50.0], [200.0, 75.0]])
        err = np.array([[5.0, 3.0], [10.0, 4.0]])

        flux_c, err_c = apply_zeropoints(flux, err, entries)

        chex.assert_equal_shape([flux_c, flux])
        chex.assert_equal_shape([err_c, err])

    def test_apply_zeropoints_zero_offset_unchanged(self):
        """Verify zero offset and zero sys err leave values unchanged."""
        entry = ZeropointEntry(
            survey="TEST",
            release="v1",
            filter_name="F150W",
            mag_offset=0.0,
            fractional_sys_err=0.0,
            source="test",
        )
        flux = np.array([100.0])
        err = np.array([5.0])

        flux_c, err_c = apply_zeropoints(flux, err, [entry])

        np.testing.assert_array_almost_equal(flux_c, flux)
        np.testing.assert_array_almost_equal(err_c, err)

    def test_apply_zeropoints_adds_floor(self):
        """Verify fractional_sys_err increases errors."""
        entry = ZeropointEntry(
            survey="TEST",
            release="v1",
            filter_name="F150W",
            mag_offset=0.0,
            fractional_sys_err=0.1,
            source="test",
        )
        flux = np.array([100.0])
        err = np.array([1.0])

        _, err_c = apply_zeropoints(flux, err, [entry])

        # err_corrected = sqrt(1.0^2 + (0.1 * 100.0)^2) = sqrt(1 + 100) ~ 10.05
        expected_err = np.sqrt(1.0 + 10.0**2)
        np.testing.assert_almost_equal(err_c[0], expected_err)

    def test_apply_zeropoints_mag_offset(self):
        """Verify magnitude offset is applied correctly."""
        # mag_offset = 0.1 means flux is multiplied by 10^(-0.4 * 0.1) ~ 0.905
        entry = ZeropointEntry(
            survey="TEST",
            release="v1",
            filter_name="F150W",
            mag_offset=0.1,
            fractional_sys_err=0.0,
            source="test",
        )
        flux = np.array([100.0])
        err = np.array([5.0])

        flux_c, err_c = apply_zeropoints(flux, err, [entry])

        mag_factor = 10.0 ** (-0.4 * 0.1)
        expected_flux = 100.0 * mag_factor
        expected_err = 5.0 * mag_factor

        np.testing.assert_almost_equal(flux_c[0], expected_flux)
        np.testing.assert_almost_equal(err_c[0], expected_err)

    def test_apply_zeropoints_wrong_count_raises(self):
        """Verify ValueError if number of entries doesn't match filters."""
        entries = lookup_zeropoints("JADES", "DR5", ["F150W"])
        flux = np.array([100.0, 50.0])  # 2 filters
        err = np.array([5.0, 3.0])

        with pytest.raises(ValueError):
            apply_zeropoints(flux, err, entries)


class TestSystematicFloor:
    """Tests for add_systematic_floor function."""

    def test_systematic_floor_in_quadrature(self):
        """Verify systematic floor is added in quadrature."""
        flux = np.array([100.0])
        err = np.array([5.0])
        fractional = 0.02

        err_total = add_systematic_floor(flux, err, fractional=fractional)

        expected = np.sqrt(5.0**2 + (0.02 * 100.0) ** 2)
        np.testing.assert_almost_equal(err_total[0], expected)

    def test_systematic_floor_zero_floor(self):
        """Verify zero floor leaves error unchanged."""
        flux = np.array([100.0])
        err = np.array([5.0])

        err_total = add_systematic_floor(flux, err, fractional=0.0)

        np.testing.assert_array_almost_equal(err_total, err)

    def test_systematic_floor_negative_flux(self):
        """Verify absolute value is used for negative flux."""
        flux = np.array([-100.0, 100.0])
        err = np.array([5.0, 5.0])

        err_total = add_systematic_floor(flux, err, fractional=0.02)

        # Both should have the same error (absolute value)
        expected = np.sqrt(5.0**2 + (0.02 * 100.0) ** 2)
        np.testing.assert_array_almost_equal(err_total, [expected, expected])

    def test_systematic_floor_default_fractional(self):
        """Verify default fractional value is 0.02."""
        flux = np.array([100.0])
        err = np.array([5.0])

        err_total = add_systematic_floor(flux, err)

        expected = np.sqrt(5.0**2 + (0.02 * 100.0) ** 2)
        np.testing.assert_almost_equal(err_total[0], expected)

    def test_systematic_floor_array_shapes(self):
        """Verify function works with various array shapes."""
        # 1D
        flux_1d = np.array([100.0, 50.0])
        err_1d = np.array([5.0, 3.0])
        err_1d_total = add_systematic_floor(flux_1d, err_1d, fractional=0.02)
        chex.assert_equal_shape([err_1d_total, err_1d])

        # 2D
        flux_2d = np.array([[100.0, 50.0], [200.0, 75.0]])
        err_2d = np.array([[5.0, 3.0], [10.0, 4.0]])
        err_2d_total = add_systematic_floor(flux_2d, err_2d, fractional=0.02)
        chex.assert_equal_shape([err_2d_total, err_2d])


class TestDetectUpperLimits:
    """Tests for detect_upper_limits function."""

    def test_detect_upper_limits_basic(self):
        """Verify upper limits are detected below threshold."""
        flux = np.array([10.0, 0.5, 100.0])
        err = np.array([2.0, 1.0, 5.0])
        sn_threshold = 2.0

        mask = detect_upper_limits(flux, err, sn_threshold=sn_threshold)

        # S/N: [5.0, 0.5, 20.0]
        # Below 2.0: [False, True, False]
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(mask, expected)

    def test_detect_upper_limits_default_threshold(self):
        """Verify default threshold is 1.0."""
        flux = np.array([1.5, 0.5, 100.0])
        err = np.array([1.0, 1.0, 5.0])

        mask = detect_upper_limits(flux, err)

        # S/N: [1.5, 0.5, 20.0], threshold 1.0
        # Below 1.0: [False, True, False]
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(mask, expected)

    def test_detect_upper_limits_negative_flux(self):
        """Verify absolute value is used for flux."""
        flux = np.array([-10.0, 10.0])
        err = np.array([2.0, 2.0])

        mask = detect_upper_limits(flux, err, sn_threshold=3.0)

        # S/N: [5.0, 5.0], threshold 3.0
        # Both above threshold: [False, False]
        expected = np.array([False, False])
        np.testing.assert_array_equal(mask, expected)

    def test_detect_upper_limits_zero_error(self):
        """Verify handling of zero error."""
        flux = np.array([10.0, 0.0, 100.0])
        err = np.array([0.0, 0.0, 5.0])

        mask = detect_upper_limits(flux, err, sn_threshold=2.0)

        # S/N where err=0 defaults to 0 (safe division)
        # [inf, nan, 20.0] -> [False, True, False] after safe handling
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool


class TestSigmaUpperLimit:
    """Tests for sigma_upper_limit_from_flux function."""

    def test_sigma_upper_limit_scales_with_nsigma(self):
        """Verify upper limit scales with n_sigma."""
        err = np.array([1.0, 2.0, 0.5])

        ul_1 = sigma_upper_limit_from_flux(err, n_sigma=1.0)
        ul_3 = sigma_upper_limit_from_flux(err, n_sigma=3.0)

        np.testing.assert_array_almost_equal(ul_1, err)
        np.testing.assert_array_almost_equal(ul_3, 3.0 * err)

    def test_sigma_upper_limit_default_nsigma(self):
        """Verify default n_sigma is 3.0."""
        err = np.array([1.0, 2.0])

        ul = sigma_upper_limit_from_flux(err)

        expected = 3.0 * err
        np.testing.assert_array_almost_equal(ul, expected)

    def test_sigma_upper_limit_array_shape(self):
        """Verify output shape matches input shape."""
        err_1d = np.array([1.0, 2.0, 3.0])
        ul_1d = sigma_upper_limit_from_flux(err_1d)
        chex.assert_equal_shape([ul_1d, err_1d])

        err_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        ul_2d = sigma_upper_limit_from_flux(err_2d)
        chex.assert_equal_shape([ul_2d, err_2d])

    def test_sigma_upper_limit_zero_error(self):
        """Verify zero error produces zero upper limit."""
        err = np.array([0.0, 1.0, 0.0])

        ul = sigma_upper_limit_from_flux(err, n_sigma=3.0)

        expected = np.array([0.0, 3.0, 0.0])
        np.testing.assert_array_almost_equal(ul, expected)
