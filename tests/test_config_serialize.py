# SPDX-License-Identifier: BSD-3-Clause
"""Tests for model configuration serialization and deserialization.

Tests the round-trip invariant: ``build(...) -> model.config -> to_yaml() ->
from_file() -> model.config`` should produce equivalent configurations.
"""

import json
import pathlib
import tempfile

import pytest

from tengri import (
    FREE,
    FIXED,
    Fixed,
    Uniform,
    load_filter_set,
    load_ssp_data,
)
from tengri.config.serialize import (
    deserialize_config,
    dict_to_distribution,
    distribution_to_dict,
    serialize_config,
)
from tengri.forward.sed_model import SEDModel
from tengri.parameters.priors import Gaussian, LogUniform

pytestmark = pytest.mark.contract


@pytest.fixture
def ssp_data():
    """Load SSP data for testing."""
    # Use a fixture from conftest instead if available; fallback to a known file
    try:
        return load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
    except FileNotFoundError:
        pytest.skip("SSP data not available")


@pytest.fixture
def filters():
    """Minimal filter set for testing."""
    return load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])


class TestDistributionSerialization:
    """Test serialization of individual distributions."""

    def test_uniform_roundtrip(self):
        """Test Uniform distribution serialization."""
        dist = Uniform(0, 1)
        serialized = distribution_to_dict(dist)
        assert serialized["__prior__"] == "Uniform"
        assert serialized["lo"] == 0
        assert serialized["hi"] == 1

        # Deserialize
        reconstructed = dict_to_distribution(serialized)
        assert isinstance(reconstructed, Uniform)
        assert reconstructed.lo == dist.lo
        assert reconstructed.hi == dist.hi

    def test_gaussian_roundtrip(self):
        """Test Gaussian distribution serialization."""
        dist = Gaussian(mu=0.5, sigma=0.1)
        serialized = distribution_to_dict(dist)
        assert serialized["__prior__"] == "Gaussian"

        reconstructed = dict_to_distribution(serialized)
        assert isinstance(reconstructed, Gaussian)

    def test_fixed_roundtrip(self):
        """Test Fixed distribution serialization."""
        dist = Fixed(0.5)
        serialized = distribution_to_dict(dist)
        assert serialized == {"__fixed__": 0.5}

        # Fixed dicts need full deserialization, not dict_to_distribution
        reconstructed = deserialize_config(serialized)
        assert isinstance(reconstructed, Fixed)
        assert reconstructed.value == 0.5

    def test_fixed_string_roundtrip(self):
        """Test Fixed with string value serialization."""
        dist = Fixed("solar")
        serialized = distribution_to_dict(dist)
        assert serialized == {"__fixed__": "solar"}

        # Fixed dicts need full deserialization
        reconstructed = deserialize_config(serialized)
        assert reconstructed.value == "solar"

    def test_unknown_distribution_error(self):
        """Test error handling for unknown distribution."""
        malformed = {"__prior__": "Unifrom"}  # Misspelled
        with pytest.raises(Exception):  # ConfigError
            dict_to_distribution(malformed, strict=True)


class TestSentinelSerialization:
    """Test serialization of FREE and FIXED sentinels."""

    def test_free_serialization(self):
        """Test FREE sentinel serialization."""
        config = {"all_params": FREE}
        serialized = serialize_config(config)
        assert serialized["all_params"] == "FREE"

        deserialized = deserialize_config(serialized)
        assert deserialized["all_params"] is FREE

    def test_fixed_sentinel_serialization(self):
        """Test FIXED sentinel serialization."""
        config = {"all_params": FIXED}
        serialized = serialize_config(config)
        assert serialized["all_params"] == "FIXED"

        deserialized = deserialize_config(serialized)
        assert deserialized["all_params"] is FIXED

    def test_case_insensitive_sentinel(self):
        """Test that sentinels are case-insensitive on read."""
        config = {"all_params": "free"}
        deserialized = deserialize_config(config)
        assert deserialized["all_params"] is FREE

        config = {"all_params": "FIXED"}
        deserialized = deserialize_config(config)
        assert deserialized["all_params"] is FIXED


class TestNestedConfigSerialization:
    """Test serialization of nested group configurations."""

    def test_simple_group_config(self):
        """Test serialization of a simple group config."""
        config = {
            "sfh": {"type": "dpl", "all_params": FREE},
            "redshift": Fixed(0.1),
        }
        serialized = serialize_config(config)
        assert serialized["sfh"]["type"] == "dpl"
        assert serialized["sfh"]["all_params"] == "FREE"
        assert serialized["redshift"] == {"__fixed__": 0.1}

        deserialized = deserialize_config(serialized)
        assert deserialized["sfh"]["all_params"] is FREE
        assert deserialized["redshift"].value == 0.1

    def test_nested_prior_config(self):
        """Test serialization of config with nested priors."""
        config = {
            "sfh": {"type": "dpl", "beta": Uniform(1, 3)},
            "dust_emission": {
                "type": "dale2014",
                "eta": LogUniform(0.01, 1),
            },
        }
        serialized = serialize_config(config)
        assert serialized["sfh"]["beta"]["__prior__"] == "Uniform"
        assert serialized["dust_emission"]["eta"]["__prior__"] == "LogUniform"

        deserialized = deserialize_config(serialized)
        assert isinstance(deserialized["sfh"]["beta"], Uniform)
        assert isinstance(deserialized["dust_emission"]["eta"], LogUniform)


class TestModelConfigRoundTrip:
    """Test round-trip invariant: build -> config -> to_yaml -> from_yaml."""

    def test_roundtrip_simple_model(self, ssp_data, filters):
        """Test simple model round-trip."""
        # Build model with explicit groups (skip neb due to wNE SSP warning)
        model1 = SEDModel.build(
            ssp_data=ssp_data,
            sfh={"type": "dpl", "all_params": FIXED},
            dust={"type": "two_component", "law": "calzetti", "all_params": FIXED},
            redshift=Fixed(0.1),
            filters=filters,
        )

        # Get config and serialize
        config1 = model1.config
        yaml_str = model1.to_yaml()

        # Deserialize and rebuild
        model2 = SEDModel.from_yaml(yaml_str, ssp_data=ssp_data, filters=filters)
        config2 = model2.config

        # Configs should be equivalent
        assert config1 == config2
        assert model1.spec.free_params == model2.spec.free_params

    def test_roundtrip_with_free_params(self, ssp_data, filters):
        """Test round-trip with free parameters."""
        model1 = SEDModel.build(
            ssp_data=ssp_data,
            sfh={"type": "dpl", "all_params": FREE},
            dust={"type": "two_component", "law": "calzetti", "tau_bc": Fixed(0.5)},
            redshift=Uniform(0.0, 1.0),
            filters=filters,
        )

        yaml_str = model1.to_yaml()
        model2 = SEDModel.from_yaml(yaml_str, ssp_data=ssp_data, filters=filters)

        # Check that free params match
        assert model1.spec.free_params == model2.spec.free_params

    def test_roundtrip_with_json(self, ssp_data, filters):
        """Test round-trip using JSON format."""
        model1 = SEDModel.build(
            ssp_data=ssp_data,
            sfh={"type": "dpl", "all_params": FIXED},
            redshift=Fixed(0.1),
            filters=filters,
        )

        json_str = model1.to_json()
        config_dict = json.loads(json_str)

        model2 = SEDModel.from_dict(config_dict, ssp_data=ssp_data, filters=filters)

        assert model1.spec.free_params == model2.spec.free_params


class TestFileIO:
    """Test file I/O for config serialization."""

    def test_save_and_load_yaml(self, ssp_data, filters):
        """Test saving and loading YAML config."""
        model1 = SEDModel.build(
            ssp_data=ssp_data,
            sfh={"type": "dpl", "all_params": FIXED},
            redshift=Fixed(0.1),
            filters=filters,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = pathlib.Path(tmpdir) / "config.yaml"
            model1.to_yaml(yaml_path)
            assert yaml_path.exists()

            model2 = SEDModel.from_file(yaml_path, ssp_data=ssp_data, filters=filters)
            assert model1.spec.free_params == model2.spec.free_params

    def test_save_and_load_json(self, ssp_data, filters):
        """Test saving and loading JSON config."""
        model1 = SEDModel.build(
            ssp_data=ssp_data,
            sfh={"type": "dpl", "all_params": FIXED},
            redshift=Fixed(0.1),
            filters=filters,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = pathlib.Path(tmpdir) / "config.json"
            model1.to_json(json_path)
            assert json_path.exists()

            model2 = SEDModel.from_file(json_path, ssp_data=ssp_data, filters=filters)
            assert model1.spec.free_params == model2.spec.free_params


class TestErrorHandling:
    """Test error handling in serialization/deserialization."""

    def test_malformed_prior_dict_strict(self):
        """Test error on malformed prior dict in strict mode."""
        malformed = {"__prior__": "Uniform", "low": 0, "high": 1}  # Wrong keys
        with pytest.raises(Exception):  # ConfigError
            dict_to_distribution(malformed, strict=True)

    def test_unknown_distribution_suggestion(self):
        """Test that suggestions are provided for misspelled distributions."""
        malformed = {"__prior__": "Unifrom"}  # Typo
        try:
            dict_to_distribution(malformed, strict=True)
            assert False, "Should have raised ConfigError"
        except Exception as e:
            # Check that suggestion is in error message
            assert "Uniform" in str(e) or "did you mean" in str(e)

    def test_config_from_nonexistent_file(self, ssp_data):
        """Test error when loading config from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            SEDModel.from_file(
                "/nonexistent/path/config.yaml",
                ssp_data=ssp_data,
            )
