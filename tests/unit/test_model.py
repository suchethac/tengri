"""Unit tests for the Model class."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from diffsed.distributions import Uniform, Fixed
from diffsed.param_spec import ParamSpec
from diffsed.model import Model, PARAM_MAP, MockData


class TestParamMap:
    def test_all_params_covered(self):
        from diffsed.param_spec import VALID_PARAM_NAMES
        for name in VALID_PARAM_NAMES:
            assert name in PARAM_MAP, f"Missing mapping for {name}"

    def test_unit_conversion_tau_peak(self):
        _, scale = PARAM_MAP["sfh_tau_peak_gyr"]
        assert scale == 1e9  # Gyr → yr

    def test_unit_conversion_psd_tau(self):
        _, scale = PARAM_MAP["psd_tau_myr"]
        assert scale == 1e6  # Myr → yr


class TestModelConstruction:
    """Tests that don't need SSP data."""

    def test_param_map_completeness(self):
        from diffsed.param_spec import VALID_PARAM_NAMES
        assert set(PARAM_MAP.keys()) == VALID_PARAM_NAMES
