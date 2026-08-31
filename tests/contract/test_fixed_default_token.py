# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for the ``Fixed(DEFAULT)`` build-grammar token.

``DEFAULT`` (added alongside ``FREE``/``FIXED`` in
``tengri.parameters.sentinels``) is legal only as the argument of
``Fixed(...)``. ``Fixed(DEFAULT)`` is a fully-supported, ADDITIVE spelling of
"pin this one parameter at the registry default" -- the per-parameter
equivalent of what the ``'all_params': FIXED`` wildcard already does for
every unaddressed parameter in a group. It resolves through the exact same
canonical-table resolver (``_default_fixed_value`` in ``groups.py``, the
#412 fix) as wildcard-FIXED, never a second path.

This wave is additive: bare ``FIXED`` keeps working everywhere, unchanged. A
later wave retires it in favor of ``Fixed(DEFAULT)``; until then this file
deliberately builds both spellings side by side so a reviewer can see they
agree. Those FIXED-side comparisons are marked ``# TEMPORARY dual-window
test`` and should be deleted (not just left to bit-rot) once FIXED is
retired.
"""

from __future__ import annotations

import copy
import pickle

import pytest

from tengri import DEFAULT, FIXED, Fixed, ParameterError, builders, parse_groups
from tengri.config.serialize import deserialize_config, serialize_config
from tengri.parameters.groups import _CANONICAL_FIXED_DEFAULTS

pytestmark = pytest.mark.contract


# ── (a) Equivalence with FIXED, at every grammar site ──────────────────────


class TestEquivalenceWithFixed:
    """Fixed(DEFAULT) must resolve identically to FIXED, everywhere FIXED works."""

    def test_per_param_sfh_dpl(self):
        """A per-parameter Fixed(DEFAULT) pins the same value FIXED would."""
        default_spec = parse_groups(
            sfh={"type": "dpl", "all_params": FIXED, "alpha": Fixed(DEFAULT)},
            redshift=Fixed(0.1),
        )
        # TEMPORARY dual-window test: rewrite against Fixed(DEFAULT)-only in the removal wave
        fixed_spec = parse_groups(
            sfh={"type": "dpl", "all_params": FIXED, "alpha": FIXED},
            redshift=Fixed(0.1),
        )
        assert (
            default_spec.get_distribution("sfh_dpl_alpha").value
            == fixed_spec.get_distribution("sfh_dpl_alpha").value
        )
        assert (
            default_spec._group_provenance["sfh_dpl_alpha"]
            == fixed_spec._group_provenance["sfh_dpl_alpha"]
            == "user_fixed"
        )

    def test_per_param_met_delta(self):
        """The #412 met_logzsol pin, spelled Fixed(DEFAULT) instead of FIXED."""
        default_spec = parse_groups(
            met={"type": "delta", "logzsol": Fixed(DEFAULT)}, redshift=Fixed(0.1)
        )
        # TEMPORARY dual-window test: rewrite against Fixed(DEFAULT)-only in the removal wave
        fixed_spec = parse_groups(met={"type": "delta", "logzsol": FIXED}, redshift=Fixed(0.1))
        assert (
            default_spec.get_distribution("met_logzsol").value
            == fixed_spec.get_distribution("met_logzsol").value
        )

    def test_wildcard_met_delta(self):
        """'all_params': Fixed(DEFAULT) collapses a group exactly like 'all_params': FIXED."""
        default_spec = parse_groups(
            met={"type": "delta", "all_params": Fixed(DEFAULT)}, redshift=Fixed(0.1)
        )
        # TEMPORARY dual-window test: rewrite against Fixed(DEFAULT)-only in the removal wave
        fixed_spec = parse_groups(met={"type": "delta", "all_params": FIXED}, redshift=Fixed(0.1))
        for name in ("met_logzsol", "met_alpha_fe", "met_logzsol_scatter"):
            assert (
                default_spec.get_distribution(name).value
                == fixed_spec.get_distribution(name).value
            ), name
            assert default_spec._group_provenance[name] == fixed_spec._group_provenance[name], name

    def test_top_level_redshift(self):
        """redshift=Fixed(DEFAULT) pins the same value redshift=FIXED would."""
        default_spec = parse_groups(
            sfh={"type": "dpl", "all_params": FIXED}, redshift=Fixed(DEFAULT)
        )
        # TEMPORARY dual-window test: rewrite against Fixed(DEFAULT)-only in the removal wave
        fixed_spec = parse_groups(sfh={"type": "dpl", "all_params": FIXED}, redshift=FIXED)
        assert (
            default_spec.get_distribution("redshift").value
            == fixed_spec.get_distribution("redshift").value
        )
        assert (
            default_spec._group_provenance["redshift"]
            == fixed_spec._group_provenance["redshift"]
            == "user_fixed"
        )


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
            parse_groups(sfh={"type": "dpl", "all_params": FIXED}, redshift=DEFAULT)


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

        # And it resolves through parse_groups exactly like the original.
        default_spec = parse_groups(**deserialized)
        fixed_spec = parse_groups(
            sfh={"type": "dpl", "log_total_mass": FIXED}, redshift=Fixed(0.1)
        )
        assert (
            default_spec.get_distribution("sfh_dpl_log_total_mass").value
            == fixed_spec.get_distribution("sfh_dpl_log_total_mass").value
        )

    def test_legacy_fixed_string_still_deserializes(self):
        """The pre-existing "FIXED" string arm is untouched this wave."""
        deserialized = deserialize_config({"sfh": {"all_params": "FIXED"}})
        assert deserialized["sfh"]["all_params"] is FIXED


# ── (f) Builders ─────────────────────────────────────────────────────────


class TestBuilders:
    """builders.*(all_params=...) accepts Fixed(DEFAULT) alongside FIXED."""

    def test_all_params_fixed_default_equivalent_to_fixed(self):
        default_dict = builders.sfh.dpl(all_params=Fixed(DEFAULT))
        # TEMPORARY dual-window test: rewrite against Fixed(DEFAULT)-only in the removal wave
        fixed_dict = builders.sfh.dpl(all_params=FIXED)

        default_spec = parse_groups(sfh=default_dict, redshift=Fixed(0.1))
        fixed_spec = parse_groups(sfh=fixed_dict, redshift=Fixed(0.1))
        for name in ("sfh_dpl_alpha", "sfh_dpl_beta", "sfh_dpl_tau_gyr", "sfh_dpl_log_total_mass"):
            assert (
                default_spec.get_distribution(name).value
                == fixed_spec.get_distribution(name).value
            ), name

    def test_concrete_fixed_value_still_rejected(self):
        """Fixed(1.5) is not a valid wildcard value -- only FREE/FIXED/Fixed(DEFAULT)."""
        with pytest.raises(ValueError, match="expected FREE or FIXED"):
            builders.sfh.dpl(all_params=Fixed(1.5))

    def test_bare_default_rejected_with_fixed_default_pointer(self):
        with pytest.raises(ValueError, match="Fixed\\(DEFAULT\\)"):
            builders.sfh.dpl(all_params=DEFAULT)


# ── (g) Fixed() with no arguments is still a TypeError ──────────────────────


def test_fixed_with_no_arguments_is_still_a_typeerror():
    with pytest.raises(TypeError):
        Fixed()
