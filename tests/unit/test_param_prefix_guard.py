"""Unit test for parameter prefix guard (NAMING_CONTRACT §3.2).

Verifies that all free parameters in preset configurations comply with the
naming contract requiring domain prefixes (sfh_, met_, dust_, etc.).
"""

import sys
from pathlib import Path

import pytest

# Add tools to path to import the guard
tools_path = Path(__file__).parent.parent.parent / "tools"
sys.path.insert(0, str(tools_path))

from check_param_prefixes import check_preset, is_valid_param_name

import tengri.presets as presets


class TestParamPrefixValidation:
    """Tests for parameter name validation against NAMING_CONTRACT."""

    def test_valid_names(self):
        """Test that valid parameter names pass validation."""
        valid_names = [
            "redshift",
            "sfh_dpl_alpha",
            "sfh_field_psd_sigma",
            "met_logzsol",
            "dust_tau_bc",
            "neb_logU",
            "agn_log_lbol",
            "eline_sigma_kms",
            "noise_f_cal",
            "radio_q_ir",
            "xray_gamma_agn",
            "shock_frac",
            "chem_alpha_fe",
            "igm_tau_igm",
            "dla_logN_dla",
        ]
        for name in valid_names:
            assert is_valid_param_name(name), f"'{name}' should be valid"

    def test_invalid_names(self):
        """Test that invalid parameter names fail validation."""
        invalid_names = ["psd_xi", "psd_sigma", "dpl_alpha", "tau_bc"]
        for name in invalid_names:
            assert not is_valid_param_name(name), f"'{name}' should be invalid"

    @pytest.mark.parametrize(
        "preset_fn",
        [
            presets.starforming,
            presets.quiescent,
            presets.high_z,
            presets.photoz,
            presets.jwst_spec,
            presets.agn_host,
        ],
    )
    def test_preset_compliance(self, preset_fn):
        """Test that all presets comply with parameter naming contract."""
        params, _ = preset_fn()
        violations = check_preset(preset_fn.__name__, params, None)
        assert len(violations) == 0, f"Preset '{preset_fn.__name__}' has violations: {violations}"
