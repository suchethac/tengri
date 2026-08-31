# SPDX-License-Identifier: BSD-3-Clause
"""Tests for model configuration serialization round-trip invariant.

Tests that ``build(...) -> model.config -> to_yaml() -> from_yaml() -> model.config``
produces equivalent configurations for real-world model structures.
"""

import json
import pathlib
import tempfile

import pytest

from tengri import (
    DEFAULT,
    FREE,
    Fixed,
    Uniform,
    load_filter_set,
    load_ssp_data,
)
from tengri.config.serialize import (
    deserialize_config,
    serialize_config,
)
from tengri.forward.sed_model import SEDModel

pytestmark = pytest.mark.contract


@pytest.fixture
def ssp_data():
    """Load SSP data for testing."""
    try:
        return load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
    except FileNotFoundError:
        pytest.skip("SSP data not available")


@pytest.fixture
def filters():
    """Minimal filter set for testing."""
    return load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])


class TestRoundTripInvariant:
    """Test the round-trip invariant for various model structures."""

    def test_two_screen_dust_different_laws(self, ssp_data, filters):
        """Test round-trip with two-screen dust model with different law_bc and law_diff.

        This tests the case mentioned in #75: a dust model where law_bc and law_diff
        are specified separately (not with a unified 'law'), which is where a serializer
        can break if it doesn't properly handle the nested structure.
        """
        model1 = SEDModel.build(
            ssp_data=ssp_data,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law_bc": "calzetti",
                "law_diff": "power_law",
                "all_params": Fixed(DEFAULT),
            },
            filters=filters,
            redshift=Fixed(0.1),
        )

        # Round-trip: config -> yaml -> from_yaml -> config
        config1 = model1.config
        yaml_str = model1.to_yaml()
        model2 = SEDModel.from_yaml(yaml_str, ssp_data=ssp_data, filters=filters)
        config2 = model2.config

        # Configs must be equivalent
        assert config1 == config2
        assert model1.spec.free_params == model2.spec.free_params

    def test_dust_emission_free_eta_balance(self, ssp_data, filters):
        """Test round-trip with dust_emission model with free eta_balance parameter.

        This tests dust_emission as a separate group (post-split grammar) with
        a free parameter, which exercises parameter serialization within nested groups.
        """
        model1 = SEDModel.build(
            ssp_data=ssp_data,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "single_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={
                "type": "dale2014",
                "eta_balance": Uniform(0.3, 0.7),
                "all_params": Fixed(DEFAULT),
            },
            filters=filters,
            redshift=Fixed(0.1),
        )

        # Round-trip
        yaml_str = model1.to_yaml()
        model2 = SEDModel.from_yaml(yaml_str, ssp_data=ssp_data, filters=filters)

        # Verify the free parameter is preserved
        assert "dust_eta_balance" in model2.spec.free_params
        assert "dust_eta_balance" in model1.spec.free_params
        assert model1.spec.free_params == model2.spec.free_params

    def test_dust_emission_serialization(self, ssp_data, filters):
        """Test round-trip with separate dust_emission group (post-split grammar).

        This tests the post-split grammar where dust_attenuation and dust_emission
        are separate groups, which is where a serializer can fail if it doesn't
        recursively handle multiple top-level groups properly.
        """
        model1 = SEDModel.build(
            ssp_data=ssp_data,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "single_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
            filters=filters,
            redshift=Fixed(0.1),
        )

        # Round-trip
        config1 = model1.config
        json_str = model1.to_json()
        config_dict = json.loads(json_str)
        model2 = SEDModel.from_dict(config_dict, ssp_data=ssp_data, filters=filters)
        config2 = model2.config

        # Both dust groups must be preserved
        assert config1 == config2
        assert config1["dust_attenuation"]["type"] == config2["dust_attenuation"]["type"]
        assert config1["dust_emission"]["type"] == config2["dust_emission"]["type"]

    def test_round_trip_preserves_priors(self, ssp_data, filters):
        """Test that complex priors are correctly serialized and deserialized."""
        model1 = SEDModel.build(
            ssp_data=ssp_data,
            sfh={"type": "dpl", "all_params": FREE, "alpha": Uniform(0.5, 2.0)},
            filters=filters,
            redshift=Fixed(0.1),
        )

        # Serialize and deserialize
        yaml_str = model1.to_yaml()
        model2 = SEDModel.from_yaml(yaml_str, ssp_data=ssp_data, filters=filters)

        # Check that free params and their priors match
        assert model1.spec.free_params == model2.spec.free_params
        # Verify the Uniform prior for alpha was preserved
        assert model1.config["sfh"]["alpha"] == model2.config["sfh"]["alpha"]

    def test_agn_nested_sub_blocks(self, ssp_data, filters):
        """Test round-trip with AGN composable model with nested sub-blocks.

        This tests the most complex nesting case: AGN model with disc and torus
        as separate nested sub-blocks, each with their own parameters. This is
        the case most likely to break a serializer with depth management.
        """
        model1 = SEDModel.build(
            ssp_data=ssp_data,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn={
                "type": "composable",
                "disc": {"type": "powerlaw", "all_params": Fixed(DEFAULT)},
                "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
                "all_params": Fixed(DEFAULT),
            },
            filters=filters,
            redshift=Fixed(0.1),
        )

        # Round-trip via YAML
        yaml_str = model1.to_yaml()
        model2 = SEDModel.from_yaml(yaml_str, ssp_data=ssp_data, filters=filters)

        # Round-trip via JSON
        json_str = model1.to_json()
        model3 = SEDModel.from_json(json_str, ssp_data=ssp_data, filters=filters)

        # Verify nested structure is preserved in all cases
        assert (
            model1.config["agn"]["type"]
            == model2.config["agn"]["type"]
            == model3.config["agn"]["type"]
        )
        assert (
            model1.config["agn"]["disc"]["type"]
            == model2.config["agn"]["disc"]["type"]
            == model3.config["agn"]["disc"]["type"]
        )
        assert (
            model1.config["agn"]["torus"]["type"]
            == model2.config["agn"]["torus"]["type"]
            == model3.config["agn"]["torus"]["type"]
        )

        # Verify the full config structure matches
        assert model1.config == model2.config
        assert model1.config == model3.config


class TestSerializationErrorHandling:
    """Test error handling in serialization (strict mode only)."""

    def test_malformed_prior_raises(self):
        """Test that malformed prior dict raises ConfigError."""
        from tengri.config.exceptions import ConfigError

        malformed = {"__prior__": "Uniform", "low": 0, "high": 1}  # Wrong keys
        with pytest.raises(ConfigError):
            deserialize_config(malformed)

    def test_unknown_distribution_with_suggestion(self):
        """Test that typo suggestions are provided for unknown distributions."""
        from tengri.config.exceptions import ConfigError

        # Typo: "Unifrom" instead of "Uniform"
        malformed = {"__prior__": "Unifrom", "lo": 0, "hi": 1}
        with pytest.raises(ConfigError) as exc_info:
            deserialize_config(malformed)

        # Check that suggestion is in error message
        error_text = str(exc_info.value)
        # Should suggest "Uniform" as a close match
        assert "Uniform" in error_text or "did you mean" in error_text.lower()

    def test_config_from_nonexistent_file(self, ssp_data):
        """Test error when loading config from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            SEDModel.from_file(
                "/nonexistent/path/config.yaml",
                ssp_data=ssp_data,
            )

    @pytest.mark.parametrize("legacy", ["FIXED", "fixed", "Fixed"])
    def test_legacy_fixed_string_raises_config_error(self, legacy):
        """The retired FIXED sentinel's wire form (any case) is a loud error.

        The FIXED sentinel was removed pre-1.0; a config written by an older
        tengri that serialized it as the bare string "FIXED" must not
        silently resolve to something else -- it must name the replacement.
        """
        from tengri.config.exceptions import ConfigError

        with pytest.raises(ConfigError, match="the FIXED sentinel was removed"):
            deserialize_config({"redshift": legacy})


class TestFixedDefaultWireForm:
    """``Fixed(DEFAULT)``'s wire form, ``{"__fixed_default__": true}``, round-trips."""

    def test_fixed_default_serializes_to_wire_form(self):
        assert serialize_config({"x": Fixed(DEFAULT)}) == {"x": {"__fixed_default__": True}}

    def test_wire_form_deserializes_to_fixed_default(self):
        restored = deserialize_config({"x": {"__fixed_default__": True}})
        assert restored["x"] == Fixed(DEFAULT)

    def test_fixed_default_round_trips_through_serialize_deserialize(self):
        config = {"sfh": {"type": "dpl", "all_params": Fixed(DEFAULT)}}
        restored = deserialize_config(serialize_config(config))
        assert restored == config

    def test_free_still_round_trips_as_the_string_free(self):
        """FREE's wire form is unchanged by the FIXED sentinel's removal."""
        assert serialize_config({"x": FREE}) == {"x": "FREE"}
        assert deserialize_config({"x": "FREE"})["x"] is FREE


class TestFileIOFormats:
    """Test file I/O for different formats."""

    def test_save_and_load_yaml(self, ssp_data, filters):
        """Test saving and loading YAML config file."""
        model1 = SEDModel.build(
            ssp_data=ssp_data,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            filters=filters,
            redshift=Fixed(0.1),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = pathlib.Path(tmpdir) / "config.yaml"
            model1.to_yaml(yaml_path)
            assert yaml_path.exists()

            model2 = SEDModel.from_file(yaml_path, ssp_data=ssp_data, filters=filters)
            assert model1.spec.free_params == model2.spec.free_params

    def test_save_and_load_json(self, ssp_data, filters):
        """Test saving and loading JSON config file."""
        model1 = SEDModel.build(
            ssp_data=ssp_data,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            filters=filters,
            redshift=Fixed(0.1),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = pathlib.Path(tmpdir) / "config.json"
            model1.to_json(json_path)
            assert json_path.exists()

            model2 = SEDModel.from_file(json_path, ssp_data=ssp_data, filters=filters)
            assert model1.spec.free_params == model2.spec.free_params
