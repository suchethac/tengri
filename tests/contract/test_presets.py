# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for tengri.presets module.

Tests the factory functions for common galaxy type presets.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract
import tengri.presets as _presets

_REQUIRED = ("starforming", "quiescent", "high_z", "photoz", "jwst_spec", "agn_host", "describe")
_missing = [name for name in _REQUIRED if not hasattr(_presets, name)]
if _missing:
    pytest.skip(
        f"Galaxy-type presets not yet implemented (missing: {', '.join(_missing)}). "
        "See tengri.presets — only `synthesizer_default` is currently registered.",
        allow_module_level=True,
    )

from tengri.config.settings import SEDModelConfig
from tengri.parameters.parameters import Parameters
from tengri.presets import (
    agn_host,
    describe,
    high_z,
    jwst_spec,
    list_presets,
    photoz,
    quiescent,
    starforming,
)
from tengri.registry import _RegistryTable


class TestListPresets:
    """Test the package preset registry menu (galaxy-type presets registered)."""

    def test_list_presets_nonempty(self):
        """list_presets() returns a non-empty registry table (#1574)."""
        presets = list_presets()
        assert isinstance(presets, _RegistryTable)
        assert len(presets) > 0

    def test_list_presets_contains_expected(self):
        """All six galaxy-type presets are registered in the package menu."""
        names = set(list_presets().names())
        expected = {"starforming", "quiescent", "high_z", "photoz", "jwst_spec", "agn_host"}
        assert expected.issubset(names), f"Missing: {expected - names}"

    def test_list_presets_entries_carry_metadata(self):
        """Registry entries expose short_doc and citations for the menu."""
        entry = next(row for row in list_presets() if row["name"] == "starforming")
        assert entry["short_doc"]
        assert "Calzetti_2000" in entry["citations"]


class TestEachPresetReturnsValidTuple:
    """Test that each preset returns a valid (Parameters, SEDModelConfig) tuple."""

    @pytest.mark.parametrize(
        "preset_func",
        [starforming, quiescent, high_z, photoz, jwst_spec, agn_host],
    )
    def test_preset_returns_tuple(self, preset_func):
        """Each preset returns a 2-tuple."""
        result = preset_func()
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.parametrize(
        "preset_func",
        [starforming, quiescent, high_z, photoz, jwst_spec, agn_host],
    )
    def test_preset_returns_parameters_and_model_config(self, preset_func):
        """First element is Parameters, second is SEDModelConfig."""
        params, config = preset_func()
        assert isinstance(params, Parameters), f"{preset_func.__name__} returned wrong types"
        assert isinstance(config, SEDModelConfig), f"{preset_func.__name__} returned wrong types"


class TestRedshiftPassthrough:
    """Test that redshift argument is correctly applied."""

    @pytest.mark.parametrize(
        "preset_func,redshift_val",
        [
            (starforming, 0.5),
            (starforming, 1.5),
            (quiescent, 0.3),
            (quiescent, 1.0),
            (high_z, 5.0),
            (high_z, 8.0),
            (photoz, 0.5),
            (photoz, 5.0),
            (jwst_spec, 2.0),
            (jwst_spec, 10.0),
            (agn_host, 0.3),
            (agn_host, 1.5),
        ],
    )
    def test_redshift_fixed_when_provided(self, preset_func, redshift_val):
        """The redshift is fixed to the value passed, not merely fixed.

        The table above gives every preset two redshifts precisely so the pair
        can tell them apart -- but asserting only ``is_fixed`` made the two rows
        identical in what they claimed, and a preset that ignored the argument
        and pinned ``Fixed(0.0)`` passed both. A wrong-but-fixed redshift is a
        silent 1e17 flux error (NAMING_CONTRACT §4b.2), so the value is the
        part worth asserting.
        """
        params, _ = preset_func(redshift=redshift_val)
        z_dist = params._distributions.get("redshift")
        assert z_dist is not None
        assert z_dist.is_fixed, f"{preset_func.__name__} did not fix redshift"
        assert z_dist.value == pytest.approx(redshift_val), (
            f"{preset_func.__name__} fixed redshift to {z_dist.value}, not {redshift_val}"
        )

    @pytest.mark.parametrize(
        "preset_func",
        [starforming, quiescent, high_z, photoz, jwst_spec, agn_host],
    )
    def test_redshift_free_when_none(self, preset_func):
        """When redshift=None, it is a free parameter in the preset."""
        params, _ = preset_func(redshift=None)
        z_dist = params._distributions.get("redshift")
        assert z_dist is not None
        assert not z_dist.is_fixed, f"{preset_func.__name__} did not leave redshift free"


class TestDescribe:
    """Test describe() function."""

    @pytest.mark.parametrize(
        "preset_name",
        ["starforming", "quiescent", "high_z", "photoz", "jwst_spec", "agn_host"],
    )
    def test_describe_returns_string(self, preset_name):
        """describe() returns non-empty string for each valid preset."""
        desc = describe(preset_name)
        assert isinstance(desc, string_types := (str,))
        assert len(desc) > 0

    @pytest.mark.parametrize(
        "preset_name",
        ["starforming", "quiescent", "high_z", "photoz", "jwst_spec", "agn_host"],
    )
    def test_describe_contains_preset_name(self, preset_name):
        """describe() output mentions the preset name or concept."""
        desc = describe(preset_name)
        # Rough check: name or concept should appear (case-insensitive)
        # high_z may appear as "high-z" or "high_z" or be conceptually present
        desc_lower = desc.lower()
        name_variants = [
            preset_name.lower(),
            preset_name.replace("_", "-").lower(),
        ]
        assert any(variant in desc_lower for variant in name_variants)

    def test_describe_unknown_raises(self):
        """describe() raises ValueError for unknown preset name."""
        with pytest.raises(ValueError, match="Unknown preset"):
            describe("nonexistent_preset")

    def test_describe_error_message_lists_valid(self):
        """ValueError message lists valid presets."""
        with pytest.raises(ValueError) as exc_info:
            describe("invalid")
        msg = str(exc_info.value)
        # Should mention available presets
        for name in ["starforming", "quiescent", "high_z", "photoz", "jwst_spec", "agn_host"]:
            assert name in msg


class TestPresetsConsistency:
    """Test internal consistency of preset outputs."""

    @pytest.mark.parametrize(
        "preset_func",
        [starforming, quiescent, high_z, photoz, jwst_spec, agn_host],
    )
    def test_preset_free_params_nonempty(self, preset_func):
        """Each preset has at least one free parameter."""
        params, _ = preset_func()
        assert len(params.free_params) > 0

    @pytest.mark.parametrize(
        "preset_func",
        [starforming, quiescent, high_z, photoz, jwst_spec, agn_host],
    )
    def test_preset_can_sample(self, preset_func):
        """Each preset can be sampled (assumes SSP data not required for constructor)."""
        import jax.random

        params, _ = preset_func()
        key = jax.random.PRNGKey(0)
        # Sampling may fail if SSP data is required; skip gracefully
        try:
            sample = params.sample(key)
            assert sample is not None
            assert len(sample) > 0
        except Exception as e:
            # SSP data may not be available; skip
            if "SSP" in str(e) or "data" in str(e).lower():
                pytest.skip(f"Sampling requires SSP data: {e}")
            raise
