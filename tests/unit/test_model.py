"""Unit tests for the Model class."""

import jax

jax.config.update("jax_enable_x64", True)

from diffsed.model import PARAM_MAP


class TestParamMap:
    def test_all_params_covered(self):
        from diffsed.param_spec import VALID_PARAM_NAMES

        for name in VALID_PARAM_NAMES:
            assert name in PARAM_MAP, f"Missing mapping for {name}"

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


class TestModelConstruction:
    """Tests that don't need SSP data."""

    def test_param_map_completeness(self):
        from diffsed.param_spec import VALID_PARAM_NAMES

        assert set(PARAM_MAP.keys()) == VALID_PARAM_NAMES
