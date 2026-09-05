# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for the ``Fixed(DEFAULT)`` build-grammar token.

``DEFAULT`` (in ``tengri.parameters.sentinels``, alongside ``FREE``) is legal
only as the argument of ``Fixed(...)``. ``Fixed(DEFAULT)`` spells "pin this
one parameter at the registry default" -- the per-parameter equivalent of
what the ``'all_params': Fixed(DEFAULT)`` wildcard does for every unaddressed
parameter in a group. It resolves through the exact same canonical-table
resolver (``_default_fixed_value`` in ``groups.py``, the #412 fix) the
wildcard uses, never a second path.

Pre-1.0 clean break: the bare ``FIXED`` sentinel this file originally
compared ``Fixed(DEFAULT)`` against side by side has been REMOVED from the
library; its replacement everywhere is ``Fixed(DEFAULT)``. The dual-window
tests that once built both spellings to prove they agreed (formerly marked
``# TEMPORARY dual-window test``) now assert ``Fixed(DEFAULT)``'s resolved
values directly against the registry.
"""

from __future__ import annotations

import copy
import pickle

import pytest

from tengri import DEFAULT, Fixed, ParameterError, builders, parse_groups
from tengri.config.exceptions import ConfigError
from tengri.config.serialize import deserialize_config, serialize_config
from tengri.parameters.groups import _CANONICAL_FIXED_DEFAULTS

pytestmark = pytest.mark.contract


# ── (a) Fixed(DEFAULT) pins the registry default, at every grammar site ────


class TestFixedDefaultPinsTheRegistryDefault:
    """Fixed(DEFAULT) pins a parameter at its registry default, wherever the
    grammar accepts it: per-parameter, the group wildcard, and top-level."""

    def test_per_param_sfh_dpl(self):
        """A per-parameter Fixed(DEFAULT) pins at the declared ``default=``,
        not the prior midpoint (``sfh_dpl_alpha``'s Uniform(0.1, 5.0) midpoint
        is 2.55; its curated ``default=`` is 1.5)."""
        spec = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "alpha": Fixed(DEFAULT)},
            redshift=Fixed(0.1),
        )
        assert spec.get_distribution("sfh_dpl_alpha").value == 1.5
        assert spec._group_provenance["sfh_dpl_alpha"] == "user_fixed"

    def test_per_param_met_delta(self):
        """The #412 met_logzsol pin: Fixed(DEFAULT) resolves to 0.0 (solar)."""
        spec = parse_groups(met={"type": "delta", "logzsol": Fixed(DEFAULT)}, redshift=Fixed(0.1))
        assert spec.get_distribution("met_logzsol").value == 0.0

    def test_wildcard_met_delta(self):
        """'all_params': Fixed(DEFAULT) pins every declared param at its registry default."""
        spec = parse_groups(
            met={"type": "delta", "all_params": Fixed(DEFAULT)}, redshift=Fixed(0.1)
        )
        expected = {"met_logzsol": 0.0, "met_alpha_fe": 0.0, "met_logzsol_scatter": 0.1}
        for name, value in expected.items():
            assert spec.get_distribution(name).value == value, name
            assert spec._group_provenance[name] == "wildcard_fixed", name

    def test_top_level_redshift(self):
        """redshift=Fixed(DEFAULT) pins at the registry default for redshift."""
        spec = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)}, redshift=Fixed(DEFAULT)
        )
        assert spec.get_distribution("redshift").value == 0.1
        assert spec._group_provenance["redshift"] == "user_fixed"


# ── (b) The #412 pin: canonical table, not the prior midpoint ──────────────


class TestCanonicalTablePin:
    """Fixed(DEFAULT) must route through _CANONICAL_FIXED_DEFAULTS, not the midpoint.

    #412: the prior-midpoint fallback for met_logzsol (Uniform(-2.0, 0.2) ->
    -0.9) silently injected a ~0.85 dex metallicity offset. The fix was a
    curated table (``_CANONICAL_FIXED_DEFAULTS`` in groups.py) consulted by
    ``_default_fixed_value``. Fixed(DEFAULT) must resolve through that same
    table, not fall back to the prior midpoint.
    """

    def test_met_logzsol_pins_at_solar_not_midpoint(self):
        spec = parse_groups(met={"type": "delta", "logzsol": Fixed(DEFAULT)}, redshift=Fixed(0.1))
        assert spec.get_distribution("met_logzsol").value == 0.0
        # And that literal 0.0 is exactly what the curated table says, not a
        # coincidence of the prior's shape.
        assert _CANONICAL_FIXED_DEFAULTS["met_logzsol"] == 0.0


# ── (c) Reader guards, repr, equality, identity, pickle, deepcopy ──────────


class TestUnresolvedTokenReaders:
    """A bare Fixed(DEFAULT) used as a distribution (parser never touched it)."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda f: f.value,
            lambda f: f.default,
            lambda f: f.bounds,
            lambda f: f.sample(None),
            lambda f: f.unstandardize(0.0),
        ],
        ids=["value", "default", "bounds", "sample", "unstandardize"],
    )
    def test_reader_raises(self, call):
        """Every value-reading accessor raises rather than leaking the sentinel."""
        with pytest.raises(ValueError, match="Fixed\\(DEFAULT\\)"):
            call(Fixed(DEFAULT))

    def test_repr_prints_fixed_default(self):
        assert repr(Fixed(DEFAULT)) == "Fixed(DEFAULT)"

    def test_is_fixed_is_still_true(self):
        """is_fixed does not read _value, so it stays True even unresolved."""
        assert Fixed(DEFAULT).is_fixed is True

    def test_equality_and_identity(self):
        assert Fixed(DEFAULT) == Fixed(DEFAULT)
        assert Fixed(DEFAULT) != Fixed(0.3)
        assert Fixed(0.3) != Fixed(DEFAULT)

    def test_survives_pickle(self):
        restored = pickle.loads(pickle.dumps(Fixed(DEFAULT)))
        assert restored._value is DEFAULT
        assert restored == Fixed(DEFAULT)

    def test_survives_deepcopy(self):
        restored = copy.deepcopy(Fixed(DEFAULT))
        assert restored._value is DEFAULT
        assert restored == Fixed(DEFAULT)


# ── (d) Bare DEFAULT is not a legal parameter value ─────────────────────────


class TestBareDefaultRaises:
    """DEFAULT outside Fixed(...) is not a grammar directive -- it must raise."""

    def test_per_param(self):
        with pytest.raises(ParameterError, match="Fixed\\(DEFAULT\\)"):
            parse_groups(met={"type": "delta", "logzsol": DEFAULT}, redshift=Fixed(0.1))

    def test_wildcard_slot(self):
        with pytest.raises(ParameterError, match="Fixed\\(DEFAULT\\)"):
            parse_groups(met={"type": "delta", "all_params": DEFAULT}, redshift=Fixed(0.1))

    def test_top_level_redshift(self):
        with pytest.raises(ParameterError, match="Fixed\\(DEFAULT\\)"):
            parse_groups(sfh={"type": "dpl", "all_params": Fixed(DEFAULT)}, redshift=DEFAULT)


# ── (e) Serialization ───────────────────────────────────────────────────────


class TestSerialization:
    """Round-trip of a raw group-dict spec containing Fixed(DEFAULT)."""

    def test_group_dict_round_trip(self):
        """serialize_config/deserialize_config preserve the unresolved token.

        This is the pre-parse group-dict serializer path (``model.config`` /
        recipe serialization), not a full SEDModel round-trip.
        """
        spec = {
            "sfh": {"type": "dpl", "log_total_mass": Fixed(DEFAULT)},
            "redshift": Fixed(0.1),
        }
        serialized = serialize_config(spec)
        assert serialized["sfh"]["log_total_mass"] == {"__fixed_default__": True}

        deserialized = deserialize_config(serialized)
        restored = deserialized["sfh"]["log_total_mass"]
        assert isinstance(restored, Fixed)
        assert restored._value is DEFAULT

        # And it resolves through parse_groups to the registry default (the
        # declared ``default=`` for sfh_dpl_log_total_mass is 10.0).
        default_spec = parse_groups(**deserialized)
        assert default_spec.get_distribution("sfh_dpl_log_total_mass").value == 10.0

    def test_legacy_fixed_string_now_raises_config_error(self):
        """The removed FIXED sentinel's wire form ("FIXED") is a loud error.

        Pre-1.0 clean break: a config serialized by an older tengri that wrote
        the bare string "FIXED" must not silently resolve to something else --
        it must name the replacement, ``Fixed(DEFAULT)``.
        """
        with pytest.raises(ConfigError, match="the FIXED sentinel was removed"):
            deserialize_config({"sfh": {"all_params": "FIXED"}})


# ── (f) Builders ─────────────────────────────────────────────────────────


class TestBuilders:
    """builders.*(all_params=...) accepts Fixed(DEFAULT); it is the factory default."""

    def test_all_params_fixed_default_is_the_factory_default(self):
        """The wildcard emits verbatim, and pins every declared param at its
        registry default when parsed (matching the declared ``default=`` for
        each, not the prior midpoint)."""
        default_dict = builders.sfh.dpl(all_params=Fixed(DEFAULT))
        assert default_dict == {"type": "dpl", "all_params": Fixed(DEFAULT)}
        # The signature default (no all_params= given at all) is the same token.
        assert builders.sfh.dpl() == default_dict

        spec = parse_groups(sfh=default_dict, redshift=Fixed(0.1))
        expected = {
            "sfh_dpl_alpha": 1.5,
            "sfh_dpl_beta": 1.0,
            "sfh_dpl_tau_gyr": 3.0,
            "sfh_dpl_log_total_mass": 10.0,
        }
        for name, value in expected.items():
            assert spec.get_distribution(name).value == value, name

    def test_concrete_fixed_value_still_rejected(self):
        """Fixed(1.5) is not a valid wildcard value -- only FREE/Fixed(DEFAULT)."""
        with pytest.raises(ValueError, match=r"expected FREE or Fixed\(DEFAULT\)"):
            builders.sfh.dpl(all_params=Fixed(1.5))

    def test_bare_default_rejected_with_fixed_default_pointer(self):
        with pytest.raises(ValueError, match="Fixed\\(DEFAULT\\)"):
            builders.sfh.dpl(all_params=DEFAULT)


# ── (g) Fixed() with no arguments is still a TypeError ──────────────────────


def test_fixed_with_no_arguments_is_still_a_typeerror():
    with pytest.raises(TypeError):
        Fixed()
