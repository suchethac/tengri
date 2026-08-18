# SPDX-License-Identifier: BSD-3-Clause
"""Tests for synthetic (top-hat) bandpass filters.

Covers ALMA, X-ray, and submillimeter continuum bands. Frozen output
contracts for existing ALMA bands, and new functionality tests for
added facilities.

Markers:
- bounds: wavelength ordering, transmission non-negativity
- contract: backward compatibility, registry consistency
- regression_paper: band edge accuracy against literature
"""

import numpy as np
import pytest

from tengri.observation.filters import (
    FILTER_REGISTRY,
    SYNTHETIC_BAND_REGISTRY,
    filter_info,
    list_available_filters,
    load_alma_band,
    load_filter,
    load_synthetic_band,
)
from tengri.observation.photometry import FilterCurve
from tengri.observation.photometry_config import Photometry

pytestmark = pytest.mark.bounds


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(params=sorted(SYNTHETIC_BAND_REGISTRY.keys()))
def synthetic_band_name(request):
    """Parameterize over all synthetic band names."""
    return request.param


@pytest.fixture(params=range(1, 11))
def alma_band_number(request):
    """Parameterize over ALMA band numbers 1–10."""
    return request.param


# ── Tests: Synthetic band loading ──────────────────────────────────


class TestLoadSyntheticBand:
    """load_synthetic_band: shape, monotonicity, transmission bounds."""

    @pytest.mark.contract
    def test_synthetic_band_loads(self, synthetic_band_name):
        """Synthetic band loads and returns FilterCurve with correct name."""
        fc = load_synthetic_band(synthetic_band_name)
        assert isinstance(fc, FilterCurve)
        assert fc.name == synthetic_band_name
        assert fc.wave.ndim == 1
        assert fc.trans.ndim == 1
        assert len(fc.wave) == len(fc.trans)

    @pytest.mark.bounds
    def test_wavelength_strictly_ascending(self, synthetic_band_name):
        """Wavelength array strictly increases (required for interp)."""
        fc = load_synthetic_band(synthetic_band_name)
        wave_np = np.asarray(fc.wave)
        diffs = np.diff(wave_np)
        assert np.all(diffs > 0), f"Found non-increasing wavelengths in {synthetic_band_name}"

    @pytest.mark.bounds
    def test_transmission_nonnegative_and_finite(self, synthetic_band_name):
        """Transmission is non-negative and finite (top-hat should be = 1)."""
        fc = load_synthetic_band(synthetic_band_name)
        trans_np = np.asarray(fc.trans)
        assert np.all(trans_np >= 0.0), f"Negative transmission in {synthetic_band_name}"
        assert np.all(np.isfinite(trans_np)), f"Non-finite transmission in {synthetic_band_name}"
        # Top-hat should be close to 1 in the band
        assert np.max(trans_np) > 0.9, f"Max transmission too low in {synthetic_band_name}"

    @pytest.mark.bounds
    def test_transmission_is_rectangular(self, synthetic_band_name):
        """Transmission curve is rectangular (top-hat approximation)."""
        fc = load_synthetic_band(synthetic_band_name)
        trans_np = np.asarray(fc.trans)
        # All transmission values should be close to 0 or 1 (top-hat)
        # Allow small numerical tolerance
        assert np.all((trans_np < 0.01) | (trans_np > 0.99)), (
            f"Top-hat not rectangular in {synthetic_band_name}: {trans_np}"
        )


class TestSyntheticBandViaLoadFilter:
    """load_filter with synthetic band names: integration with main loader."""

    @pytest.mark.contract
    def test_load_filter_accepts_synthetic_names(self, synthetic_band_name):
        """load_filter() should accept synthetic band names."""
        fc = load_filter(synthetic_band_name)
        assert isinstance(fc, FilterCurve)
        assert fc.name == synthetic_band_name

    @pytest.mark.contract
    def test_synthetic_band_in_photometry(self, synthetic_band_name):
        """Photometry.from_names([...]) should load synthetic bands."""
        # This is the critical use case: SED fitters use from_names()
        phot = Photometry.from_names([synthetic_band_name])
        assert phot.filters[0].name == synthetic_band_name


class TestAlmaBandBackwardCompatibility:
    """load_alma_band: frozen contract for existing usage."""

    @pytest.mark.regression_bug
    def test_alma_band_returns_filtercurve(self, alma_band_number):
        """load_alma_band(N) returns FilterCurve with correct name."""
        fc = load_alma_band(alma_band_number)
        assert isinstance(fc, FilterCurve)
        assert fc.name == f"alma_band{alma_band_number}"

    @pytest.mark.bounds
    def test_alma_band_wavelength_ascending(self, alma_band_number):
        """ALMA band wavelengths strictly increase."""
        fc = load_alma_band(alma_band_number)
        wave_np = np.asarray(fc.wave)
        diffs = np.diff(wave_np)
        assert np.all(diffs > 0)

    @pytest.mark.regression_bug
    def test_alma_band_6_is_1p23mm(self):
        """ALMA Band 6 (211–275 GHz) should peak near 1.23 mm.

        Tolerance: 1% (specification bandwidth).
        """
        fc = load_alma_band(6)
        wave_np = np.asarray(fc.wave)
        trans_np = np.asarray(fc.trans)
        # Center wavelength from 211–275 GHz: center = (211+275)/2 = 243 GHz
        # λ = c / ν = 3e18 Å/s / (243e9 Hz) ≈ 12345 Å = 1.2345 mm
        center_idx = len(wave_np) // 2
        center_wave = float(wave_np[center_idx])
        center_mm = center_wave / 1e7  # Å to mm
        assert 1.20 < center_mm < 1.26, f"ALMA Band 6 center {center_mm} mm off spec"

    @pytest.mark.regression_bug
    def test_alma_bands_are_distinct(self):
        """Each ALMA band N < N+1 has center wavelength at shorter λ."""
        prev_center = float("inf")
        for band in range(1, 11):
            fc = load_alma_band(band)
            wave_np = np.asarray(fc.wave)
            center = float(np.mean(wave_np))  # approximate center
            # Higher band number = higher frequency = shorter wavelength
            assert center < prev_center, f"ALMA Band {band} not shorter than Band {band - 1}"
            prev_center = center

    def test_alma_band_invalid_raises(self):
        """load_alma_band(N) for N not in 1–10 raises ValueError."""
        with pytest.raises(ValueError, match="ALMA band must be"):
            load_alma_band(0)
        with pytest.raises(ValueError, match="ALMA band must be"):
            load_alma_band(11)

    def test_alma_band_custom_name(self):
        """load_alma_band accepts custom name parameter."""
        fc = load_alma_band(6, name="my_alma_6")
        assert fc.name == "my_alma_6"


class TestBandEdgeRoundTrip:
    """Synthetic bands: convert back to native units and verify accuracy."""

    @pytest.mark.regression_paper
    def test_alma_band_edges_round_trip(self):
        """ALMA band edges (GHz) round-trip through Å and back (1% tolerance)."""
        from tengri.utils.physics_constants import C_AA

        # Band 3: 84–116 GHz
        lo_ghz, hi_ghz = 84.0, 116.0
        fc = load_alma_band(3)
        wave_np = np.asarray(fc.wave)

        # Convert back: ν = c / λ
        nu_min = C_AA / (float(wave_np[-1]))  # highest index = longest wave
        nu_max = C_AA / (float(wave_np[0]))  # lowest index = shortest wave

        # Should recover the band edges within 1%
        assert abs(nu_min / 1e9 - lo_ghz) / lo_ghz < 0.01, (
            f"Band 3 low edge: {nu_min / 1e9:.1f} GHz vs {lo_ghz} GHz"
        )
        assert abs(nu_max / 1e9 - hi_ghz) / hi_ghz < 0.01, (
            f"Band 3 high edge: {nu_max / 1e9:.1f} GHz vs {hi_ghz} GHz"
        )

    @pytest.mark.regression_paper
    def test_xray_band_edges_round_trip(self):
        """X-ray band edges (keV) round-trip through Å and back (1% tolerance)."""
        from tengri.utils.physics_constants import HC_KEV_ANGSTROM

        # Chandra soft: 0.5–1.2 keV
        lo_kev, hi_kev = 0.5, 1.2
        fc = load_synthetic_band("chandra_soft")
        wave_np = np.asarray(fc.wave)

        # Convert back: E = hc / λ
        e_min = HC_KEV_ANGSTROM / float(wave_np[-1])  # longest wave = lowest energy
        e_max = HC_KEV_ANGSTROM / float(wave_np[0])  # shortest wave = highest energy

        assert abs(e_min - lo_kev) / lo_kev < 0.01, (
            f"Chandra soft low edge: {e_min:.3f} keV vs {lo_kev} keV"
        )
        assert abs(e_max - hi_kev) / hi_kev < 0.01, (
            f"Chandra soft high edge: {e_max:.3f} keV vs {hi_kev} keV"
        )


class TestRegistryConsistency:
    """Synthetic bands are findable and consistent across discovery functions."""

    @pytest.mark.contract
    def test_all_synthetic_bands_in_list(self, synthetic_band_name):
        """Every synthetic band appears in list_available_filters()."""
        table = list_available_filters()
        names = table.to_dict("name").values()
        assert synthetic_band_name in names

    @pytest.mark.contract
    def test_list_filters_marks_kind_correctly(self, synthetic_band_name):
        """list_available_filters() marks synthetic bands kind='synthetic_band'."""
        table = list_available_filters()
        rows = table.to_dict("kind")
        assert rows[synthetic_band_name] == "synthetic_band"

    @pytest.mark.contract
    def test_filter_info_works_on_synthetic(self, synthetic_band_name):
        """filter_info() computes metadata for synthetic bands."""
        info = filter_info(synthetic_band_name)
        assert info["name"] == synthetic_band_name
        assert info["svo_id"] == "synthetic"
        assert info["facility"] in SYNTHETIC_BAND_REGISTRY[synthetic_band_name].facility
        assert info["lambda_eff_aa"] > 0
        assert info["fwhm_aa"] > 0

    @pytest.mark.contract
    def test_no_alias_collision_with_svo(self):
        """Synthetic band names don't collide with SVO aliases."""
        synthetic_names = set(SYNTHETIC_BAND_REGISTRY.keys())
        svo_names = set(FILTER_REGISTRY.keys())
        collision = synthetic_names & svo_names
        assert not collision, f"Name collision: {collision}"

    @pytest.mark.contract
    def test_unknown_name_lists_synthetic_in_error(self):
        """Unknown filter error message mentions synthetic bands."""
        with pytest.raises(KeyError) as exc_info:
            load_filter("nonexistent_xyz")
        error_msg = str(exc_info.value)
        # Should mention synthetic bands as available
        assert "synthetic" in error_msg.lower() or "ALMA" in error_msg or "xray" in error_msg


class TestFacilityInference:
    """_infer_facility correctly attributes synthetic bands."""

    @pytest.mark.contract
    def test_xmm_epic_not_om(self):
        """xmm_epic_* bands are EPIC, not XMM-Newton/OM (OM is Optical Monitor)."""
        from tengri.observation.filters import _infer_facility

        # XMM-Newton/OM filters start with xmm_ (e.g., xmm_u, xmm_v)
        # but EPIC bands should be labeled differently
        facility_epic = _infer_facility("xmm_epic_soft")
        facility_om = _infer_facility("xmm_u") if "xmm_u" in FILTER_REGISTRY else "XMM-Newton/OM"

        assert facility_epic == "XMM-Newton/EPIC"
        # Make sure they're not confused (if OM filter exists)
        if "xmm_u" in FILTER_REGISTRY:
            assert facility_epic != facility_om

    @pytest.mark.contract
    def test_facility_for_all_synthetic(self, synthetic_band_name):
        """Every synthetic band has a valid facility from _infer_facility."""
        from tengri.observation.filters import _infer_facility

        facility = _infer_facility(synthetic_band_name)
        assert facility is not None
        assert facility != "Other"
        # Should match the band's declared facility
        assert facility == SYNTHETIC_BAND_REGISTRY[synthetic_band_name].facility


class TestErrorHandling:
    """Error conditions and messages."""

    def test_unknown_synthetic_band_raises(self):
        """load_synthetic_band() raises KeyError for unknown names."""
        with pytest.raises(KeyError, match="Unknown synthetic band"):
            load_synthetic_band("nonexistent_band")

    def test_unknown_filter_lists_all_options(self):
        """Unknown filter error mentions all discovery options."""
        with pytest.raises(KeyError) as exc_info:
            load_filter("definitely_not_a_band")
        msg = str(exc_info.value)
        # Should mention how to discover bands
        assert "list" in msg.lower() or "available" in msg.lower()
