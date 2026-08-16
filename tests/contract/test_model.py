# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the SEDModel class."""

import pytest

pytestmark = pytest.mark.contract

from tengri.parameters.translate import PARAM_MAP, _build_param_map


class TestLegacyParamMap:
    """Tests for the legacy module-level PARAM_MAP (backward compat)."""

    def test_contains_sfh_params(self):
        assert "sfh_alpha" in PARAM_MAP
        assert "sfh_beta" in PARAM_MAP

    def test_unit_conversion_tau_peak(self):
        _, scale, offset = PARAM_MAP["sfh_tau_peak_gyr"]
        assert scale == 1e9  # Gyr → yr
        assert offset == 0.0

    def test_unit_conversion_psd_tau(self):
        _, scale, offset = PARAM_MAP["psd_tau_myr"]
        assert scale == 1e6  # Myr → yr
        assert offset == 0.0

    def test_metallicity_solar_offset(self):
        _, scale, offset = PARAM_MAP["met_logzsol"]
        assert scale == 1.0
        # log10(Zsun) ≈ -1.848 (Asplund 2009)
        assert abs(offset - (-1.8477116556169435)) < 1e-10


class TestDynamicParamMap:
    """Tests for the dynamic _build_param_map function."""

    def test_tsnorm_has_sfh_params(self):
        pm = _build_param_map(["tsnorm"])
        assert "sfh_tsnorm_log_total_mass" in pm
        assert "sfh_tsnorm_peak_lbt_gyr" in pm

    def test_tsnorm_has_non_sfh_params(self):
        pm = _build_param_map(["tsnorm"])
        assert "met_logzsol" in pm
        assert "dust_tau_bc" in pm

    def test_tsnorm_field_has_psd_params(self):
        pm = _build_param_map(["tsnorm", "field"])
        assert "sfh_field_psd_sigma" in pm
        assert "sfh_field_psd_tau_myr" in pm

    def test_dpl_has_alpha_beta(self):
        pm = _build_param_map(["dpl"])
        assert "sfh_dpl_alpha" in pm
        assert "sfh_dpl_beta" in pm
        assert "sfh_dpl_tau_gyr" in pm

    def test_unit_conversions(self):
        pm = _build_param_map(["tsnorm"])
        _, scale, _ = pm["sfh_tsnorm_peak_lbt_gyr"]
        assert scale == 1e9  # Gyr → yr
