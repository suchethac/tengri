# SPDX-License-Identifier: BSD-3-Clause
"""Tests for parameter registration of newly-wired models.

Verifies that THEMIS, BOSA, and patchy IGM parameters appear in
the Parameters registry when their respective models are enabled.
"""

import pytest

pytestmark = pytest.mark.contract

from tengri.parameters.parameters import Parameters


class TestThemisWiring:
    """Verify THEMIS dust emission params appear in registry."""

    def test_qhac_registered(self):
        """dust_qhac should be in registry when dust_emission='themis'."""
        spec = Parameters(mean_sfh_type="dpl", dust_emission="themis")
        assert "dust_qhac" in spec._param_registry

    def test_qhac_default(self):
        """Default dust_qhac should be 0.17."""
        spec = Parameters(mean_sfh_type="dpl", dust_emission="themis")
        assert spec._defaults["dust_qhac"].value == 0.17


class TestBosaWiring:
    """Verify BOSA dust emission params appear in registry."""

    def test_log_ssfr_registered(self):
        """dust_log_ssfr should be in registry when dust_emission='bosa'."""
        spec = Parameters(mean_sfh_type="dpl", dust_emission="bosa")
        assert "dust_log_ssfr" in spec._param_registry

    def test_log_ssfr_default(self):
        """Default dust_log_ssfr should be -10.0."""
        spec = Parameters(mean_sfh_type="dpl", dust_emission="bosa")
        assert spec._defaults["dust_log_ssfr"].value == -10.0


class TestPatchyIGMWiring:
    """Verify patchy IGM params appear when igm_patchy=True."""

    def test_params_registered_when_enabled(self):
        """igm_x_HI and igm_bubble_mpc should be in registry."""
        spec = Parameters(mean_sfh_type="dpl", igm_patchy=True)
        assert "igm_x_HI" in spec._param_registry
        assert "igm_bubble_mpc" in spec._param_registry

    def test_params_absent_when_disabled(self):
        """IGM patchy params should NOT be in registry when default."""
        spec = Parameters(mean_sfh_type="dpl")
        assert "igm_x_HI" not in spec._param_registry
        assert "igm_bubble_mpc" not in spec._param_registry

    def test_igm_defaults(self):
        """Default values: x_HI=0 (fully ionized), bubble=10 Mpc."""
        spec = Parameters(mean_sfh_type="dpl", igm_patchy=True)
        assert spec._defaults["igm_x_HI"].value == 0.0
        assert spec._defaults["igm_bubble_mpc"].value == 10.0

    def test_igm_patchy_attribute(self):
        """Parameters should store igm_patchy as attribute."""
        spec = Parameters(mean_sfh_type="dpl", igm_patchy=True)
        assert spec.igm_patchy is True

        spec2 = Parameters(mean_sfh_type="dpl")
        assert spec2.igm_patchy is False


class TestDocstringUpdated:
    """Verify new models appear in dust_emission options."""

    def test_new_models_in_docstring(self):
        """Parameters docstring should mention astrodust, bosa, themis."""
        doc = Parameters.__doc__ or ""
        for model in ["astrodust", "bosa", "themis"]:
            assert model in doc.lower(), f"'{model}' not in Parameters docstring"
