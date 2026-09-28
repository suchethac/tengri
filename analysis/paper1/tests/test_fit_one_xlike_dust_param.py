# SPDX-License-Identifier: BSD-3-Clause
"""Test dust_parameter_name() resolves X-like configurations.

Verifies that fit_one.dust_parameter_name() correctly returns the dust parameter
for both grid (I-VI) and X-like configurations, and that the function raises a
clear error if an unknown configuration is passed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PAPER1_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PAPER1_DIR.parent))
sys.path.insert(0, str(PAPER1_DIR))

from fit_one import dust_parameter_name

pytestmark = pytest.mark.unit


class TestDustParameterName:
    """Test dust_parameter_name for grid and X-like configurations."""

    def test_grid_config_dust_params(self):
        """Grid configurations (I-VI) return correct dust parameter names."""
        assert dust_parameter_name("I") == "dust_tau_diff"
        assert dust_parameter_name("II") == "dust_tau_v"
        assert dust_parameter_name("III") == "dust_tau_diff"
        assert dust_parameter_name("IV") == "dust_tau_diff"
        assert dust_parameter_name("V") == "dust_tau_v"
        assert dust_parameter_name("VI") == "dust_tau_diff"

    def test_xlike_config_dust_params(self):
        """X-like configurations return correct dust parameter names."""
        assert dust_parameter_name("bagpipes_like") == "dust_tau_v"
        assert dust_parameter_name("beagle_like") == "dust_tau_diff"
        assert dust_parameter_name("cigale_like") == "dust_tau_diff"
        assert dust_parameter_name("dense_basis_like") == "dust_tau_v"
        assert dust_parameter_name("prospector_like") == "dust_tau_diff"

    def test_unknown_config_raises_keyerror(self):
        """Unknown configuration raises KeyError with helpful message."""
        with pytest.raises(KeyError, match="No dust_param declared"):
            dust_parameter_name("unknown_config")
